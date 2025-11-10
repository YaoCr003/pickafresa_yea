"""
Fruit center estimation in camera coordinates using RealSense depth + YOLO.

This module provides two principal functionalities:
1) API: call functions to obtain per-fruit transforms T_cam_fruit (translation-only)
     with metadata, sorted by proximity.
2) Direct run tool: interactive CLI to select camera intrinsics, model, dataset
     (for classes), thresholds, and show a visual window with overlays and controls.

Design notes:
- We compute the fruit center 3D position by back-projecting the 2D center
    of each detection using the RealSense depth. This yields translation-only
    transforms in the camera frame; orientation is not estimated.
- PnP is NOT required for this per-fruit position. PnP needs known 3D-2D
    correspondences in the object's coordinate frame; typical bbox corners + depth
    give 3D points already in the camera frame, which does not satisfy PnP's
    requirements for solving the object/camera pose.
- The API reads defaults from a YAML in pickafresa_vision/configs/, and the
    interactive tool persists user choices to both the JSON config store and the
    YAML so API consumers have stable defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import importlib
import time

# Ensure repository root is on sys.path for absolute package imports when run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

# RealSense verification tools (for cached profiles and robust initialization)
REALSENSE_TESTING_PATH = REPO_ROOT / "pickafresa_vision" / "realsense_testing"
if str(REALSENSE_TESTING_PATH) not in sys.path:
    sys.path.insert(0, str(REALSENSE_TESTING_PATH))

try:
    from realsense_verify_color import get_camera_serial  # type: ignore
    from realsense_verify_full import (  # type: ignore
        get_best_full_profile,
        load_working_profiles,
    )
    HAVE_REALSENSE_VERIFICATION = True
except Exception:
    HAVE_REALSENSE_VERIFICATION = False

# Local utilities
from pickafresa_vision.vision_tools.config_store import (
    load_config,
    save_config,
    get_namespace,
    update_namespace,
)

# Inference API (YOLO)
def _import_inference_api():
    mod = None
    for name in (
        "pickafresa_vision.vision_nodes.inference_bbox",
        "pickafresa_vision.vision_nodes.inference_node_bbox",
    ):
        try:
            mod = importlib.import_module(name)
            return mod
        except Exception:
            continue
    return None
def _import_inference_api():
    mod = None
    for name in (
        "pickafresa_vision.vision_nodes.inference_bbox",
        "pickafresa_vision.vision_nodes.inference_node_bbox",
    ):
        try:
            mod = importlib.import_module(name)
            return mod
        except Exception:
            continue
    return None

_INF_MOD = _import_inference_api()
HAVE_INFERENCE_API = _INF_MOD is not None


# ----------------------------- Data models (API) -----------------------------

# Strawberry physical dimensions (mm)
STRAWBERRY_WIDTH_MM = 32.5
STRAWBERRY_HEIGHT_MM = 34.3

@dataclass(frozen=True)
class FruitCenterResult:
    """Result for a single detected fruit with full 6DOF pose.

    Attributes:
    - index: detection index in the frame (sorted by confidence desc from model)
    - label: class label (string)
    - class_id: numeric class id
    - confidence: detection confidence [0,1]
    - bbox_xyxy: tuple (x1,y1,x2,y2)
    - pos_cam: np.ndarray shape (3,) in meters, [x,y,z] - position only
    - rvec: rotation vector from PnP (3,) if solved, else None
    - rotation_matrix: 3x3 rotation matrix from PnP if solved, else None
    - T_cam_fruit: 4x4 homogeneous transform (full rotation + translation from PnP)
    """

    index: int
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    pos_cam: np.ndarray
    rvec: Optional[np.ndarray]
    rotation_matrix: Optional[np.ndarray]
    T_cam_fruit: np.ndarray


@dataclass
class Detection:
    """Lightweight container for a YOLO bounding box."""

    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))


class DepthSampler:
    """Samples a depth neighborhood around a pixel and returns the median."""

    def __init__(self, window: int = 5, min_valid: int = 3) -> None:
        if window <= 0 or window % 2 == 0:
            raise ValueError("window size must be a positive odd integer")
        self.window = window
        self.min_valid = max(1, min_valid)

    def sample(self, depth_frame: rs.depth_frame, pixel: Tuple[int, int]) -> Optional[float]:
        cx, cy = pixel
        width, height = depth_frame.get_width(), depth_frame.get_height()
        half = self.window // 2
        values: List[float] = []

        for yy in range(max(0, cy - half), min(height, cy + half + 1)):
            for xx in range(max(0, cx - half), min(width, cx + half + 1)):
                dist = depth_frame.get_distance(xx, yy)
                if dist > 0 and math.isfinite(dist):
                    values.append(dist)

        if len(values) < self.min_valid:
            return None

        return float(np.median(values))


def _depth_colormap_adaptive(depth_frame: rs.depth_frame) -> np.ndarray:
    """Create an adaptive colorized depth map for visualization.

    Scales based on non-zero depth values per-frame for better contrast.
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    if depth_image.size == 0:
        return np.zeros((depth_frame.get_height(), depth_frame.get_width(), 3), dtype=np.uint8)
    # Mask out zeros
    mask = depth_image > 0
    if not np.any(mask):
        scale = 1.0
    else:
        dmin = float(np.percentile(depth_image[mask], 5))
        dmax = float(np.percentile(depth_image[mask], 95))
        if dmax <= dmin:
            dmin, dmax = float(depth_image[mask].min()), float(depth_image[mask].max())
        scale = 255.0 / max(1e-6, (dmax - dmin))
        depth_image = np.clip((depth_image - dmin) * scale, 0, 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return depth_colormap


def pixel_to_camera(
    pixel: Tuple[int, int],
    depth: float,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Back-project a pixel and depth to a 3D point in the camera frame."""

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u, v = pixel
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z], dtype=np.float32)


def _T_from_pos(p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.asarray(p, dtype=np.float32).reshape(3)
    return T


def _build_object_points_strawberry() -> np.ndarray:
    """Build 4 corner points for a strawberry in its own coordinate frame.
    
    Origin at center, Z=0 plane, corners at +/- half-width and half-height.
    Returns shape (4, 3) in meters.
    """
    hw = (STRAWBERRY_WIDTH_MM / 2.0) / 1000.0  # meters
    hh = (STRAWBERRY_HEIGHT_MM / 2.0) / 1000.0
    # Top-left, top-right, bottom-right, bottom-left (clockwise from TL)
    return np.array([
        [-hw, -hh, 0.0],
        [+hw, -hh, 0.0],
        [+hw, +hh, 0.0],
        [-hw, +hh, 0.0],
    ], dtype=np.float32)


def _solve_fruit_pnp(
    bbox_xyxy: Tuple[int, int, int, int],
    depth_frame: rs.depth_frame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    depth_sampler: DepthSampler,
    debug: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Solve PnP for a single fruit detection.
    
    Returns (rvec, tvec, rotation_matrix, T_cam_fruit) if successful, else None.
    """
    x1, y1, x2, y2 = bbox_xyxy
    # Image corners (TL, TR, BR, BL)
    corners_2d = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)
    
    # Sample depth at each corner
    depths = []
    for cx, cy in corners_2d:
        d = depth_sampler.sample(depth_frame, (int(cx), int(cy)))
        if d is None:
            if debug:
                print(f"  [PnP DEBUG] Corner ({int(cx)},{int(cy)}) has no valid depth")
            return None
        depths.append(d)
    
    # Verify depth consistency (reject if corners vary >30% from median)
    median_depth = float(np.median(depths))
    depth_variations = [abs(d - median_depth) / max(median_depth, 1e-6) for d in depths]
    max_variation = max(depth_variations)
    
    # Adaptive threshold: more lenient at close range where depth gradients are steeper
    # Close range (<0.5m): 50% tolerance
    # Mid range (0.5-2m): 40% tolerance  
    # Far range (>2m): 30% tolerance
    if median_depth < 0.5:
        threshold = 0.50
    elif median_depth < 2.0:
        threshold = 0.40
    else:
        threshold = 0.30
    
    if debug:
        print(f"  [PnP DEBUG] bbox={bbox_xyxy}, depths={[f'{d:.3f}' for d in depths]}, median={median_depth:.3f}m")
        print(f"  [PnP DEBUG] variations={[f'{v:.1%}' for v in depth_variations]}, max={max_variation:.1%}, threshold={threshold:.1%}")
    
    if max_variation > threshold:
        if debug:
            print(f"  [PnP DEBUG] REJECTED: depth variation {max_variation:.1%} exceeds {threshold:.1%} threshold")
        return None
    
    # Object points (strawberry corners in its frame)
    obj_pts = _build_object_points_strawberry()
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        corners_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        if debug:
            print(f"  [PnP DEBUG] cv2.solvePnP FAILED")
        return None
    
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation_matrix.astype(np.float32)
    T[:3, 3] = tvec.ravel().astype(np.float32)
    
    if debug:
        print(f"  [PnP DEBUG] SUCCESS: tvec={tvec.ravel()}")
    
    return rvec, tvec, rotation_matrix.astype(np.float32), T


# ----------------------------- Intrinsics loading ----------------------------


def load_camera_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from YAML or FileStorage."""

    if yaml:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        camera_matrix = _extract_camera_matrix(data, path)
        dist_coeffs = _extract_dist_coeffs(data)
        return camera_matrix, dist_coeffs

    return _load_intrinsics_with_filestorage(path)


def _extract_camera_matrix(data: Dict, origin: Path) -> np.ndarray:
    # Prefer explicit OpenCV-style matrix
    cm = data.get("camera_matrix")
    if cm is not None:
        if isinstance(cm, dict) and "data" in cm:
            arr = np.array(cm["data"], dtype=np.float32).reshape(int(cm.get("rows", 3)), int(cm.get("cols", 3)))
        else:
            arr = np.array(cm, dtype=np.float32).reshape(3, 3)
        if arr.shape != (3, 3):
            raise ValueError(f"camera_matrix in {origin} must be 3x3")
        return arr

    # Fallback to scalar intrinsics
    intr = data.get("camera_intrinsics")
    if isinstance(intr, dict):
        fx = float(intr.get("fx"))
        fy = float(intr.get("fy"))
        cx = float(intr.get("cx"))
        cy = float(intr.get("cy"))
        arr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return arr

    raise KeyError(f"camera_matrix or camera_intrinsics not found in {origin}")


def _extract_dist_coeffs(data: Dict) -> np.ndarray:
    """Return distortion coefficients as an (N,1) float32 array.

    Supports keys: "dist_coeffs" (OpenCV style), "distortion_coefficients" (rows/cols/data),
    or "distortion_parameters" as individual named coefficients.
    """
    if "dist_coeffs" in data:
        coeffs = data.get("dist_coeffs")
        if isinstance(coeffs, dict) and "data" in coeffs:
            arr = np.array(coeffs["data"], dtype=np.float32)
        else:
            arr = np.array(coeffs, dtype=np.float32)
        return arr.reshape((-1, 1))

    if "distortion_coefficients" in data:
        node = data["distortion_coefficients"]
        if isinstance(node, dict) and "data" in node:
            arr = np.array(node["data"], dtype=np.float32)
            return arr.reshape((-1, 1))

    # Named parameters map: k1,k2,p1,p2,k3 (OpenCV plumb_bob order)
    if "distortion_parameters" in data and isinstance(data["distortion_parameters"], dict):
        dp = data["distortion_parameters"]
        k1 = float(dp.get("k1", 0.0))
        k2 = float(dp.get("k2", 0.0))
        p1 = float(dp.get("p1", 0.0))
        p2 = float(dp.get("p2", 0.0))
        k3 = float(dp.get("k3", 0.0))
        arr = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        return arr.reshape((-1, 1))

    # Default to zero-distortion
    return np.zeros((5, 1), dtype=np.float32)


def _load_intrinsics_with_filestorage(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)

    camera_node = storage.getNode("camera_matrix")
    dist_node = storage.getNode("dist_coeffs")
    camera_matrix = camera_node.mat()
    dist_coeffs = dist_node.mat() if not dist_node.empty() else np.zeros((5, 1), dtype=np.float32)
    storage.release()

    if camera_matrix is None or camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix in FileStorage must be 3x3")
    return camera_matrix.astype(np.float32), dist_coeffs.astype(np.float32).reshape((-1, 1))


# ----------------------------- API implementation ----------------------------

API_DEFAULTS_YAML = REPO_ROOT / "pickafresa_vision" / "configs" / "fruit_center_estimator.yaml"
MODELS_DIR = REPO_ROOT / "pickafresa_vision" / "models"
DATASETS_DIR = REPO_ROOT / "pickafresa_vision" / "datasets"
CALIB_DIR = REPO_ROOT / "pickafresa_vision" / "camera_calibration"


def _load_yaml(path: Path) -> Dict:
    if not yaml or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: Dict) -> None:
    if not yaml:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        pass


def _discover_intrinsics() -> List[Path]:
    return sorted(CALIB_DIR.glob("calib*.yaml")) if CALIB_DIR.exists() else []


def _discover_models() -> List[Path]:
    return sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []


def _discover_datasets() -> List[Path]:
    if not DATASETS_DIR.exists():
        return []
    return [d for d in DATASETS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]


def _load_class_names_from_dataset(dataset_path: Path) -> List[str]:
    names: List[str] = []
    if not yaml:
        return names
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return names
    try:
        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("names", [])
        if isinstance(raw, list):
            names = [str(x) for x in raw]
        elif isinstance(raw, dict):
            names = [str(v) for _, v in sorted(raw.items())]
    except Exception:
        names = []
    return names


class FruitCenterEstimator:
    """Function-based API for estimating fruit centers in camera coordinates.

    Typical usage:
        api = FruitCenterEstimator()  # loads defaults from YAML if present
        results = api.estimate_once(camera_mode="realsense", model_conf=0.25, target_class="ripe")
        # results: List[FruitCenterResult] sorted by proximity (z ascending)
    """

    def __init__(self, defaults_yaml: Optional[Path] = None) -> None:
        self.defaults_yaml = defaults_yaml or API_DEFAULTS_YAML
        self.defaults = _load_yaml(self.defaults_yaml)
        self._model = None
        self._model_path: Optional[Path] = None
        self._intrinsics_path: Optional[Path] = None
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._depth_sampler = DepthSampler(window=int(self.defaults.get("depth_window", 5)),
                                           min_valid=int(self.defaults.get("min_valid_depth_samples", 3)))

    def _ensure_model(self, model_conf: float) -> None:
        if self._model is not None:
            return
        if not HAVE_INFERENCE_API:
            raise ImportError("inference API not available")
        # Resolve model path
        mp = self.defaults.get("model_path")
        if mp:
            cand = (REPO_ROOT / mp) if not mp.startswith("/") else Path(mp)
        else:
            models = _discover_models()
            cand = models[0] if models else None
        if not cand or not Path(cand).exists():
            raise FileNotFoundError("No YOLO model found. Place a .pt in pickafresa_vision/models or set in YAML.")
        self._model_path = Path(cand)
        self._model = getattr(_INF_MOD, "load_model")(str(self._model_path))
        # Store last used conf
        self.defaults["confidence"] = float(model_conf)

    def _ensure_intrinsics(self) -> None:
        if self._camera_matrix is not None:
            return
        ip = self.defaults.get("intrinsics_path")
        if ip:
            cand = (REPO_ROOT / ip) if not ip.startswith("/") else Path(ip)
        else:
            intrinsics = _discover_intrinsics()
            cand = intrinsics[0] if intrinsics else None
        if not cand or not Path(cand).exists():
            raise FileNotFoundError("No intrinsics YAML found in pickafresa_vision/camera_calibration.")
        self._intrinsics_path = Path(cand)
        self._camera_matrix, self._dist_coeffs = load_camera_intrinsics(self._intrinsics_path)

    def estimate_on_frame(
        self,
        frame_bgr: np.ndarray,
        depth_frame: rs.depth_frame,
        model_conf: Optional[float] = None,
        target_class: Optional[str] = None,
        max_det: int = 300,
        debug: bool = False,
    ) -> List[FruitCenterResult]:
        """Estimate fruit 6DOF poses on a single aligned color+depth frame.

        Returns results sorted by increasing z (proximity).
        """
        self._ensure_intrinsics()
        conf = float(model_conf if model_conf is not None else self.defaults.get("confidence", 0.25))
        self._ensure_model(conf)

        # Run detection
        dets, bboxes = getattr(_INF_MOD, "infer")(self._model, frame_bgr, conf=conf, bbox_format="xyxy", normalized=False)

        results: List[FruitCenterResult] = []
        cam_mtx = self._camera_matrix  # type: ignore[assignment]
        dist_cf = self._dist_coeffs  # type: ignore[assignment]
        assert cam_mtx is not None and dist_cf is not None

        for idx, (d, bbox) in enumerate(zip(dets, bboxes)):
            # Filter by class if requested
            label = getattr(d, "clazz", str(getattr(d, "class_id", "0")))
            class_id = int(getattr(d, "class_id", -1))
            if target_class and label != target_class and str(class_id) != target_class:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            if debug:
                print(f"\n[Detection {idx}] {label} conf={getattr(d, 'confidence', 0.0):.2f} bbox={x1},{y1},{x2},{y2}")
            
            # Try PnP solver for 6DOF pose
            pnp_result = _solve_fruit_pnp((x1, y1, x2, y2), depth_frame, cam_mtx, dist_cf, self._depth_sampler, debug=debug)
            if pnp_result is not None:
                rvec, tvec, rot_mat, T = pnp_result
                pos = tvec.ravel()
                if debug:
                    print(f"  → Using PnP result (6DOF pose)")
            else:
                # Fallback: translation-only from bbox center depth
                if debug:
                    print(f"  → PnP failed, using fallback (translation-only)")
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                depth_m = self._depth_sampler.sample(depth_frame, (cx, cy))
                if depth_m is None:
                    if debug:
                        print(f"  → Fallback FAILED: no valid depth at center ({cx},{cy})")
                    continue
                pos = pixel_to_camera((cx, cy), depth_m, cam_mtx)
                rvec = None
                rot_mat = None
                T = _T_from_pos(pos)
                if debug:
                    print(f"  → Fallback SUCCESS: center_depth={depth_m:.3f}m, pos={pos}")
            
            results.append(FruitCenterResult(
                index=idx,
                label=str(label),
                class_id=class_id,
                confidence=float(getattr(d, "confidence", 0.0)),
                bbox_xyxy=(x1, y1, x2, y2),
                pos_cam=pos,
                rvec=rvec,
                rotation_matrix=rot_mat,
                T_cam_fruit=T,
            ))

        # Sort by proximity (z ascending)
        results.sort(key=lambda r: float(r.pos_cam[2]))
        return results

    def estimate_once(
        self,
        camera_mode: str = "realsense",
        model_conf: Optional[float] = None,
        target_class: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> List[FruitCenterResult]:
        """Acquire one frame from the camera and return per-fruit transforms.

        camera_mode: "realsense" is supported for depth-based estimation.
        "opencv" (no depth) will return an empty list.
        """
        if camera_mode != "realsense":
            # Depth required for 3D back-projection
            return []

        width, height = resolution or tuple(self.defaults.get("resolution", [640, 480]))
        use_fps = int(fps or self.defaults.get("fps", 30))

        with RealSenseStream((int(width), int(height)), use_fps) as stream:
            ok = False
            # Warmup a couple frames for filters/auto-exposure
            for _ in range(2):
                try:
                    _ = stream.get_aligned_frames()
                except Exception:
                    pass
                time.sleep(0.02)
            try:
                frame_bgr, depth_frame = stream.get_aligned_frames()
                ok = True
            except Exception:
                ok = False
            if not ok:
                return []
            return self.estimate_on_frame(frame_bgr, depth_frame, model_conf=model_conf, target_class=target_class)

    def save_defaults(self, patch: Dict) -> None:
        self.defaults.update(patch)
        # Convert absolute paths to repo-relative when inside repo
        for k in ("model_path", "dataset_path", "intrinsics_path"):
            v = self.defaults.get(k)
            if isinstance(v, str):
                p = Path(v)
                try:
                    self.defaults[k] = str(p.relative_to(REPO_ROOT))
                except Exception:
                    self.defaults[k] = str(p)
        _save_yaml(self.defaults_yaml, self.defaults)


def load_world_points(path: Path) -> Dict[str, np.ndarray]:
    """Load known 3D world points keyed by label."""

    if yaml:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return _extract_world_points(data, path)

    return _load_world_points_filestorage(path)


def _extract_world_points(data: Dict, origin: Path) -> Dict[str, np.ndarray]:
    node = data.get("world_points")
    if node is None:
        raise KeyError(f"world_points not found in {origin}")

    points: Dict[str, np.ndarray] = {}
    if isinstance(node, dict):
        for label, coords in node.items():
            points[label] = np.asarray(coords, dtype=np.float32).reshape(3)
    elif isinstance(node, list):
        for item in node:
            label = item.get("label")
            coords = item.get("position")
            if label is None or coords is None:
                continue
            points[label] = np.asarray(coords, dtype=np.float32).reshape(3)
    else:
        raise TypeError("world_points must be a dict or list")
    return points


def _load_world_points_filestorage(path: Path) -> Dict[str, np.ndarray]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)

    node = storage.getNode("world_points")
    if node.empty():
        storage.release()
        raise KeyError("world_points node missing in FileStorage")

    points: Dict[str, np.ndarray] = {}
    for entry in node:
        label = entry.getNode("label").string()
        coords = entry.getNode("position").mat()
        if not label or coords is None:
            continue
        points[label] = coords.astype(np.float32).reshape(3)
    storage.release()
    return points


def solve_pnp(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[Tuple[int, int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    use_ransac: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run OpenCV PnP and return (rvec, tvec, inliers)."""

    if len(object_points) < 4 or len(image_points) < 4:
        return None

    obj = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)

    if use_ransac:
        result = cv2.solvePnPRansac(
            obj,
            img,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0,
            confidence=0.995,
            iterationsCount=200,
        )
        success = result[0]
        rvec, tvec = result[1], result[2]
        inliers = result[3] if len(result) > 3 and result[3] is not None else np.arange(len(obj)).reshape(-1, 1)
    else:
        success, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        inliers = np.arange(len(obj)).reshape(-1, 1)

    if not success:
        return None
    return rvec, tvec, inliers


class YOLODetectionProvider:
    """Wraps a YOLO model to produce Detection objects from RGB frames."""

    def __init__(
        self,
        model_path: Path,
        target_class: Optional[str],
        confidence: float,
    ) -> None:
        # Try the unified inference API first
        self.use_inference_api = False
        self.model = None
        if HAVE_INFERENCE_API:
            try:
                self.model = getattr(_INF_MOD, "load_model")(str(model_path))
                self.use_inference_api = True
            except Exception:
                self.use_inference_api = False
                self.model = None
        if not self.use_inference_api:
            if YOLO is None:
                raise ImportError("ultralytics is required for YOLO-based detection")
            self.model = YOLO(str(model_path))
        self.target_class = target_class
        self.confidence = confidence

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if self.use_inference_api:
            try:
                dets, bboxes = getattr(_INF_MOD, "infer")(self.model, frame, conf=self.confidence, bbox_format="xyxy", normalized=False, display=False)
                for d in dets:
                    label = getattr(d, "clazz", str(getattr(d, "class_id", "0")))
                    if self.target_class and label != self.target_class and str(getattr(d, "class_id", label)) != self.target_class:
                        continue
                    x1, y1, x2, y2 = d.bbox_xyxy
                    detections.append(
                        Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            label=str(label),
                            confidence=float(d.confidence),
                        )
                    )
                return detections
            except Exception:
                # Fallback to direct Ultralytics call below
                self.use_inference_api = False
        # Direct Ultralytics fallback
        results = self.model(frame, verbose=False, conf=self.confidence)
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                cls_idx = int(box.cls.item())
                label = getattr(self.model, "names", {}).get(cls_idx, str(cls_idx))
                if self.target_class and label != self.target_class and str(cls_idx) != self.target_class:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=str(label),
                        confidence=conf,
                    )
                )
        return detections


class JSONDetectionProvider:
    """Loads detections from a JSON sequence for offline testing."""

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError("JSON detections must be a list of frames")
        self.frames = data
        self.index = 0

    def __call__(self, _: np.ndarray) -> List[Detection]:
        if self.index >= len(self.frames):
            return []

        frame_data = self.frames[self.index]
        self.index += 1
        detections: List[Detection] = []
        for item in frame_data:
            bbox = tuple(int(v) for v in item["bbox"])
            detections.append(
                Detection(
                    bbox=bbox,
                    label=str(item.get("label", "strawberry")),
                    confidence=float(item.get("confidence", 1.0)),
                    track_id=item.get("track_id"),
                )
            )
        return detections


class RealSenseStream:
    """Context manager around an aligned RealSense color + depth stream with profile verification."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        auto_detect: bool = True,
    ) -> None:
        self.resolution = resolution
        self.fps = fps
        self.auto_detect = auto_detect
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.0
        self._filters: List = []
        self.selected_profile: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
        self._color_is_rgb: bool = False

    def _cleanup_existing_contexts(self) -> None:
        """Force cleanup of any existing RealSense contexts (macOS workaround)."""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                try:
                    dev.hardware_reset()
                except Exception:
                    pass
            # Give hardware time to reset
            time.sleep(0.5)
        except Exception:
            pass

    def _reset_device_state(self, device_serial: Optional[str] = None) -> None:
        """Reset RealSense device hardware state (macOS workaround)."""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                if device_serial is None or dev.get_info(rs.camera_info.serial_number) == device_serial:
                    try:
                        dev.hardware_reset()
                    except Exception:
                        pass
        except Exception:
            pass

    def _force_device_cleanup(self, device_serial: Optional[str] = None) -> None:
        """Aggressively cleanup device handles (macOS workaround)."""
        import subprocess
        
        # First, try to stop pipeline if it exists
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        
        # Hardware reset
        self._reset_device_state(device_serial)
        
        # Kill macOS processes that might be holding the device
        try:
            subprocess.run(
                ['sudo', '-n', 'killall', '-9', 'VDCAssistant', 'AppleCameraAssistant'],
                capture_output=True,
                timeout=2
            )
        except Exception:
            pass

    def _prompt_select_cached_full_profile(
        self,
        cached: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        saved_pref: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Prompt user to select a full (color+depth) profile from cache.

        If saved_pref is present and matches one of the cached entries, use it automatically.
        """
        if not cached:
            return None
        # Auto-use saved preference when available
        if saved_pref and saved_pref in cached:
            c, d = saved_pref
            print(
                f"Using saved full profile preference: Color {c[0]}x{c[1]}@{c[2]} | Depth {d[0]}x{d[1]}@{d[2]}"
            )
            return saved_pref

        # If differing fps pairs exist, prompt; otherwise just use the first
        fps_pairs = {(c[2], d[2]) for c, d in cached}
        if len(fps_pairs) <= 1 and len(cached) == 1:
            return cached[0]

        print("\n=== Select Cached Full Profile (Color | Depth) ===")
        for i, (c, d) in enumerate(cached, 1):
            print(f"{i}. Color {c[0]}x{c[1]} @ {c[2]} | Depth {d[0]}x{d[1]} @ {d[2]}")
        while True:
            choice = input(f"Select profile (1-{len(cached)}) or 'q' to cancel: ").strip().lower()
            if choice == 'q':
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cached):
                    return cached[idx]
            except Exception:
                pass
            print("Please enter a valid option.")

    def __enter__(self) -> "RealSenseStream":
        width, height = self.resolution
        color_fps = self.fps
        depth_fps = self.fps

        # Auto-detect and select best profile if enabled
        if self.auto_detect and HAVE_REALSENSE_VERIFICATION:
            try:
                serial = get_camera_serial()
            except Exception:
                serial = None
            
            # Load saved preference from config
            cfg = load_config()
            ns = get_namespace(cfg, "bbox_depth_auto_pnp_calc")
            saved_full_pref = None
            if ns.get("preferred_full_profile"):
                pref = ns["preferred_full_profile"]
                if isinstance(pref, dict) and "color" in pref and "depth" in pref:
                    try:
                        saved_full_pref = (
                            tuple(pref["color"]),
                            tuple(pref["depth"])
                        )
                    except Exception:
                        pass
            
            # Try cached profiles first
            cached = load_working_profiles(serial) if serial else None
            selected: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
            
            if cached:
                # Prompt user to select from cached profiles
                selected = self._prompt_select_cached_full_profile(cached, saved_full_pref)
                
                if selected:
                    (width, height, color_fps), (dw, dh, depth_fps) = selected
                    self.selected_profile = selected
                    self.resolution = (width, height)
                    self.fps = color_fps
                    print(f"[OK] Selected profile: Color {width}x{height}@{color_fps} | Depth {dw}x{dh}@{depth_fps}")
                    
                    # Save selected profile as preference
                    try:
                        update_namespace(cfg, "bbox_depth_auto_pnp_calc", {
                            "preferred_full_profile": {
                                "color": [width, height, color_fps],
                                "depth": [dw, dh, depth_fps],
                            }
                        })
                    except Exception:
                        pass
            
            if not selected:
                # Fall back to best_full_profile detection
                try:
                    pair = get_best_full_profile(mode="independent", verbose=False, use_cache=True, validate_cached=False)
                    if pair:
                        (width, height, color_fps), (dw, dh, depth_fps) = pair
                        self.selected_profile = pair
                        self.resolution = (width, height)
                        self.fps = color_fps
                        print(f"[OK] Detected profile: Color {width}x{height}@{color_fps} | Depth {dw}x{dh}@{depth_fps}")
                except Exception:
                    pass

        # Small settle delay after verification
        time.sleep(0.3)

        # macOS-specific: Ensure any previous pipeline instances are fully released
        print("Initializing RealSense camera...")
        self._cleanup_existing_contexts()
        time.sleep(1.0)  # Extended delay for macOS camera process cleanup

        # Start with enhanced retry logic for macOS
        profile = None
        last_error = None
        color_format = rs.format.bgr8
        self._color_is_rgb = False
        for attempt in range(5):  # Increased from 3 to 5 attempts
            # Ensure we have a fresh pipeline after cleanup
            if self.pipeline is None:
                self.pipeline = rs.pipeline()

            # Build a fresh config each attempt (so we can switch formats if needed)
            config = rs.config()
            device_serial = None
            try:
                if HAVE_REALSENSE_VERIFICATION:
                    device_serial = get_camera_serial()
                    if device_serial:
                        config.enable_device(device_serial)
                        if attempt == 0:
                            print(f"[OK] Binding to device: {device_serial}")
            except Exception:
                pass

            # Configure streams
            config.enable_stream(rs.stream.color, width, height, color_format, color_fps)
            if self.selected_profile:
                dw, dh, dfps = self.selected_profile[1]
                config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            else:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)

            try:
                # On macOS, reset device before each retry attempt
                if attempt > 0:
                    print(f"⚠ Retry attempt {attempt}/4 - resetting device...")
                    self._reset_device_state(device_serial)
                    time.sleep(1.5)  # Longer delay for hardware reset

                profile = self.pipeline.start(config)
                print(f"[OK] RealSense pipeline started successfully")
                break
            except RuntimeError as e:
                last_error = e
                error_msg = str(e).lower()

                # Switch color format if resolution negotiation fails or power-state issues arise
                if ("failed to resolve" in error_msg) or ("set power state" in error_msg) or ("bgr8" in error_msg and "format" in error_msg):
                    if color_format != rs.format.rgb8:
                        print("⚠ Pipeline start error suggests format issue; switching to RGB8 and retrying...")
                        color_format = rs.format.rgb8
                        self._color_is_rgb = True

                # Check for specific error types
                if "already opened" in error_msg or "device is busy" in error_msg:
                    print(f"⚠ Device busy (attempt {attempt + 1}/5) - attempting cleanup...")
                    # Device is locked, try harder cleanup
                    self._force_device_cleanup(device_serial)
                    # Force recreation of pipeline next loop
                    self.pipeline = None
                    time.sleep(2.0)  # Give device time to fully release
                elif "no device connected" in error_msg:
                    print(f"⚠ Device not detected (attempt {attempt + 1}/5) - waiting...")
                    # Force recreation of pipeline next loop
                    self.pipeline = None
                    time.sleep(1.5)
                else:
                    print(f"⚠ Pipeline start failed (attempt {attempt + 1}/5): {e}")
                    # Force recreation of pipeline next loop
                    self.pipeline = None
                    time.sleep(1.0)

                if attempt == 4:
                    raise RuntimeError(
                        f"Failed to start RealSense pipeline after 5 attempts.\n"
                        f"Last error: {last_error}\n\n"
                        f"Troubleshooting steps:\n"
                        f"1. Unplug and replug the RealSense camera\n"
                        f"2. Kill any processes using the camera:\n"
                        f"   sudo killall -9 VDCAssistant AppleCameraAssistant\n"
                        f"3. Check if realsense_guard.sh is running:\n"
                        f"   ps aux | grep realsense_guard\n"
                        f"4. Restart the realsense_guard service if needed"
                    )
            except Exception as e:
                last_error = e
                print(f"⚠ Unexpected error (attempt {attempt + 1}/5): {type(e).__name__}: {e}")
                # Force recreation of pipeline next loop
                self.pipeline = None
                if attempt == 4:
                    raise RuntimeError(f"Failed to start RealSense pipeline after 5 attempts: {e}")
                time.sleep(1.0)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        # Setup depth filters
        try:
            self._filters = [
                rs.disparity_transform(True),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.disparity_transform(False),
                rs.hole_filling_filter(),
            ]
            spatial: rs.spatial_filter = self._filters[1]
            spatial.set_option(rs.option.holes_fill, 3)
            temporal: rs.temporal_filter = self._filters[2]
            temporal.set_option(rs.option.alpha, 0.5)
        except Exception:
            self._filters = []
        
        # Prime the pipeline with a short warmup to avoid initial timeouts on macOS
        try:
            for _ in range(5):
                _ = self.pipeline.wait_for_frames(timeout_ms=1000)
                time.sleep(0.02)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup with enhanced device release for macOS."""
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("[OK] RealSense pipeline stopped")
            except Exception as e:
                print(f"⚠ Error stopping pipeline: {e}")
            
            # macOS-specific: Give hardware time to power down cleanly
            # This is critical to prevent device lock on next run
            time.sleep(0.5)
            
            try:
                # Force hardware reset to ensure clean state
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    try:
                        dev.hardware_reset()
                    except Exception:
                        pass
                time.sleep(0.3)
            except Exception:
                pass
        
        self.pipeline = None
        self.align = None
        self._filters = []

    def get_aligned_frames(self) -> Tuple[np.ndarray, rs.depth_frame]:
        if not self.pipeline or not self.align:
            raise RuntimeError("RealSenseStream must be used as a context manager")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as e:
            raise RuntimeError(f"Failed to get frames: {e}")

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to retrieve frames from RealSense pipeline")

        # Apply depth filtering
        try:
            f = depth_frame
            for flt in self._filters:
                f = flt.process(f)
            depth_frame = f.as_depth_frame()  # type: ignore[attr-defined]
        except Exception:
            pass

        color_image = np.asanyarray(color_frame.get_data())
        if self._color_is_rgb:
            try:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        return color_image, depth_frame


# ----------------------------- Interactive CLI ------------------------------

def _prompt_select(title: str, options: List[Path]) -> Optional[Path]:
    print(f"\n=== {title} ===")
    if not options:
        print("No options available.")
        return None
    for i, p in enumerate(options, 1):
        print(f"{i}. {p.name}")
    while True:
        s = input(f"Select (1-{len(options)}) or 'q' to cancel: ").strip().lower()
        if s == 'q':
            return None
        try:
            idx = int(s) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except Exception:
            pass
        print("Please enter a valid option.")


def _prompt_target_class(classes: List[str]) -> Optional[str]:
    if not classes:
        return None
    print("\n=== Classes (blank = all) ===")
    for i, c in enumerate(classes, 1):
        print(f"{i}. {c}")
    s = input("Choose class by number or name (blank for all): ").strip()
    if not s:
        return None
    try:
        idx = int(s) - 1
        if 0 <= idx < len(classes):
            return classes[idx]
    except Exception:
        pass
    # Fallback to direct name match
    if s in classes:
        return s
    print("Unknown selection, using all classes.")
    return None


def _draw_hud(frame: np.ndarray, info: Dict, controls: List[str], font_scale: float = 0.35) -> None:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    box_w = min(420, w - 10)
    box_h = int(24 * font_scale / 0.35 * (len(info) + len(controls) + 2))
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = int(28 * font_scale / 0.35)
    for k, v in info.items():
        cv2.putText(frame, f"{k}: {v}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        y += int(22 * font_scale / 0.35)
    y += int(6 * font_scale / 0.35)
    for line in controls:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
        y += int(20 * font_scale / 0.35)


def run_interactive() -> None:
    """Interactive tool for estimating fruit 6DOF poses with visual output."""
    print("=" * 60)
    print("Fruit Center Estimator (camera frame)")
    print("Team YEA, 2025")
    print("=" * 60)

    # Load previous JSON config
    cfg = load_config()
    ns = get_namespace(cfg, "bbox_depth_auto_pnp_calc")

    # Select intrinsics, model, dataset
    intrinsics = _discover_intrinsics()
    models = _discover_models()
    datasets = _discover_datasets()

    intr_path = Path(ns.get("intrinsics_path", "")) if ns.get("intrinsics_path") else None
    if not intr_path or not intr_path.exists():
        intr_path = _prompt_select("Camera Intrinsics", intrinsics)
        if not intr_path:
            print("No intrinsics selected; exiting.")
            return

    model_path = Path(ns.get("model_path", "")) if ns.get("model_path") else None
    if not model_path or not model_path.exists():
        model_path = _prompt_select("YOLO Model (.pt)", models)
        if not model_path:
            print("No model selected; exiting.")
            return

    dataset_path = Path(ns.get("dataset_path", "")) if ns.get("dataset_path") else None
    if not dataset_path or not dataset_path.exists():
        dataset_path = _prompt_select("Dataset (for classes)", datasets)

    classes = _load_class_names_from_dataset(dataset_path) if dataset_path else []
    target_class = ns.get("target_class") or _prompt_target_class(classes)

    try:
        conf_default = float(ns.get("confidence", 0.25))
    except Exception:
        conf_default = 0.25
    conf_in = input(f"Confidence threshold [default {conf_default:.2f}]: ").strip()
    try:
        conf = float(conf_in) if conf_in else conf_default
    except Exception:
        conf = conf_default

    # Display preferences
    depth_overlay = bool(ns.get("depth_overlay", True))
    depth_alpha = float(ns.get("depth_alpha", 0.35))
    debug_mode = bool(ns.get("debug_mode", False))

    # Persist both JSON and API YAML defaults
    update_namespace(cfg, "bbox_depth_auto_pnp_calc", {
        "intrinsics_path": str(intr_path),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path) if dataset_path else "",
        "target_class": target_class or "",
        "confidence": conf,
        "depth_overlay": depth_overlay,
        "depth_alpha": depth_alpha,
        "debug_mode": debug_mode,
    })

    api = FruitCenterEstimator()
    api.save_defaults({
        "intrinsics_path": str(intr_path),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path) if dataset_path else "",
        "target_class": target_class or "",
        "confidence": conf,
    })

    # Load intrinsics and model once
    try:
        api._ensure_intrinsics()
        api._ensure_model(conf)
    except Exception as e:
        print(f"Setup error: {e}")
        return

    # Stream and visualize
    res = tuple(ns.get("resolution", [640, 480]))
    fps = int(ns.get("fps", 30))
    
    # For terminal matrix output throttling
    last_print_time = 0.0
    
    with RealSenseStream(resolution=(int(res[0]), int(res[1])), fps=fps, auto_detect=True) as stream:
        print(f"Depth scale: {stream.depth_scale:.6f} meters/unit")
        paused = False
        
        while True:
            if not paused:
                try:
                    color_image, depth_frame = stream.get_aligned_frames()
                except Exception as e:
                    print(f"Stream error: {e}")
                    break

                # Estimate per-frame
                results = api.estimate_on_frame(color_image, depth_frame, model_conf=conf, target_class=target_class, debug=debug_mode)

                # Compute font scale based on window size
                h, w = color_image.shape[:2]
                font_scale = max(0.3, min(0.5, w / 1920.0 * 0.5))
                det_font_scale = 0.35  # Fixed smaller scale for detection overlays

                # Draw depth overlay
                vis = color_image.copy()
                if depth_overlay:
                    try:
                        dcm = _depth_colormap_adaptive(depth_frame)
                        vis = cv2.addWeighted(dcm, depth_alpha, vis, 1.0 - depth_alpha, 0)
                    except Exception:
                        pass

                # Draw detections, pos text, and coordinate frames
                for r in results:
                    x1, y1, x2, y2 = r.bbox_xyxy
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx = int((x1 + x2) * 0.5)
                    cy = int((y1 + y2) * 0.5)
                    cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
                    
                    # Position text
                    txt = f"{r.label} {r.confidence:.2f} | P=({r.pos_cam[0]:.3f},{r.pos_cam[1]:.3f},{r.pos_cam[2]:.3f})m"
                    (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, det_font_scale, 1)
                    ytxt = max(15, y1 - th - 6)
                    cv2.rectangle(vis, (x1, ytxt - th - bl), (x1 + tw + 6, ytxt + bl), (0, 0, 0), -1)
                    cv2.putText(vis, txt, (x1 + 3, ytxt), cv2.FONT_HERSHEY_SIMPLEX, det_font_scale, (255, 255, 255), 1)
                    
                    # Draw coordinate frame if PnP succeeded
                    if r.rvec is not None and api._camera_matrix is not None and api._dist_coeffs is not None:
                        try:
                            axis_length = 0.040  # 40mm
                            cv2.drawFrameAxes(
                                vis,
                                api._camera_matrix,
                                api._dist_coeffs,
                                r.rvec,
                                r.T_cam_fruit[:3, 3].reshape(3, 1),
                                axis_length,
                                2,
                            )
                        except Exception:
                            pass

                # Print closest fruit matrix to terminal every ~1 second
                current_time = time.time()
                if results and (current_time - last_print_time) >= 1.0:
                    closest = results[0]
                    print(f"\n[Closest fruit] {closest.label} conf={closest.confidence:.2f} z={closest.pos_cam[2]:.3f}m")
                    print("T_cam_fruit =")
                    for row in closest.T_cam_fruit:
                        print("  ", " ".join(f"{v: .4f}" for v in row))
                    last_print_time = current_time

                # HUD
                hud_info = {
                    "Model": Path(str(model_path)).name,
                    "Intrinsics": Path(str(intr_path)).name,
                    "Conf": f"{conf:.2f}",
                    "Depth overlay": "on" if depth_overlay else "off",
                    "Alpha": f"{depth_alpha:.2f}",
                    "Debug": "ON" if debug_mode else "off",
                    "Detections": str(len(results)),
                }
                controls = [
                    "[Q] Quit | [D] Toggle depth | [ [ ] ] Alpha | [+/-] Conf",
                    "[P] Pause | [X] Debug mode",
                ]
                _draw_hud(vis, hud_info, controls, font_scale=0.35)

                cv2.imshow("Fruit Center Estimator", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                depth_overlay = not depth_overlay
            elif key == ord('['):
                depth_alpha = max(0.0, depth_alpha - 0.05)
            elif key == ord(']'):
                depth_alpha = min(1.0, depth_alpha + 0.05)
            elif key == ord('+') or key == ord('='):
                conf = min(0.99, conf + 0.05)
            elif key == ord('-') or key == ord('_'):
                conf = max(0.01, conf - 0.05)
            elif key == ord('x'):
                debug_mode = not debug_mode
                print(f"\n[Debug mode {'ENABLED' if debug_mode else 'DISABLED'}]")
                # Update config
                update_namespace(cfg, "bbox_depth_auto_pnp_calc", {"debug_mode": debug_mode})
            elif key == ord('p'):
                paused = not paused

        cv2.destroyAllWindows()


def annotate_frame(
    frame: np.ndarray,
    detections: Iterable[Detection],
    depths: Dict[int, float],
    translations: Dict[int, np.ndarray],
) -> np.ndarray:
    """Overlay detection and depth information on the output frame."""

    vis = frame.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = det.center
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

        depth = depths.get(idx)
        text_lines = [f"{det.label} {det.confidence:.2f}"]
        if depth is not None:
            text_lines.append(f"depth={depth:.3f}m")
        if idx in translations:
            t = translations[idx]
            text_lines.append(f"P=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) m")

        y = max(15, y1 - 10)
        for line in text_lines:
            cv2.putText(vis, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 15
    return vis


def build_detection_provider(args: argparse.Namespace) -> Optional[Callable[[np.ndarray], List[Detection]]]:
    if args.model:
        return YOLODetectionProvider(Path(args.model), args.target_class, args.confidence)
    if args.detections:
        return JSONDetectionProvider(Path(args.detections))
    return None


def run(args: argparse.Namespace) -> None:
    # Resolve intrinsics path (interactive discovery if missing)
    intrinsics_path = Path(args.intrinsics) if args.intrinsics else _discover_or_prompt_intrinsics()
    cached_intrinsics_yaml = _maybe_yaml(intrinsics_path) if intrinsics_path.exists() else {}
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_path)
    world_points: Dict[str, np.ndarray] = {}

    # World points can be embedded in intrinsics YAML or separate file
    if args.world_points:
        world_points = load_world_points(Path(args.world_points))
    elif cached_intrinsics_yaml.get("world_points"):
        world_points = _extract_world_points(cached_intrinsics_yaml, intrinsics_path)
    else:
        # Optionally prompt user
        if _yes_no_input("Provide a world-points YAML for full PnP (optional)? [y/N]: "):
            wp = input("Enter path to world_points YAML: ").strip()
            if wp:
                try:
                    world_points = load_world_points(Path(wp))
                except Exception as e:
                    print(f"Could not load world points: {e}")

    # Build detection provider (interactive if needed)
    # Load persisted config defaults
    cfg = load_config()
    ns = get_namespace(cfg, "bbox_depth_auto_pnp_calc")

    provider = build_detection_provider(args)
    if provider is None:
        provider = _prompt_detection_provider(ns)
        if provider is None:
            print("No detection source configured; exiting.", file=sys.stderr)
            return

    # Sampler configuration (interactive if missing)
    depth_window = args.depth_window or int(ns.get("depth_window", _int_input("Depth median window (odd) [5]: ", 5)))
    min_valid = args.min_valid_depth_samples or int(ns.get("min_valid_depth_samples", _int_input("Min valid depth samples [3]: ", 3)))
    sampler = DepthSampler(window=depth_window, min_valid=min_valid)

    # Stream configuration
    width = args.width or int(ns.get("width", _int_input("Color/Depth width [640]: ", 640)))
    height = args.height or int(ns.get("height", _int_input("Color/Depth height [480]: ", 480)))
    fps = args.fps or int(ns.get("fps", _int_input("Stream FPS [30]: ", 30)))
    max_frames_default = int(ns.get("max_frames", 0))
    max_frames = args.max_frames if args.max_frames is not None else _int_input("Max frames (0 = infinite) [0]: ", max_frames_default)
    max_frames = None if max_frames == 0 else max_frames
    display = bool(args.display or ns.get("display", _yes_no_input("Display annotated frames? [Y/n]: ", default=True)))
    use_ransac = (not args.disable_ransac) if args.disable_ransac is not None else bool(ns.get("use_ransac", _yes_no_input("Use RANSAC for PnP? [Y/n]: ", default=True)))

    # Persist choices
    update_namespace(cfg, "bbox_depth_auto_pnp_calc", {
        "depth_window": depth_window,
        "min_valid_depth_samples": min_valid,
        "width": width,
        "height": height,
        "fps": fps,
        "max_frames": (0 if max_frames is None else max_frames),
        "display": display,
        "use_ransac": use_ransac,
    })

    with RealSenseStream((width, height), fps) as stream:
        print(f"Depth scale: {stream.depth_scale:.6f} meters/unit")
        frame_count = 0
        while max_frames is None or frame_count < max_frames:
            color_image, depth_frame = stream.get_aligned_frames()
            detections: List[Detection] = provider(color_image) if provider else []

            depths: Dict[int, float] = {}
            translations: Dict[int, np.ndarray] = {}
            object_points: List[np.ndarray] = []
            image_points: List[Tuple[int, int]] = []

            for idx, det in enumerate(detections):
                center = det.center
                depth = sampler.sample(depth_frame, center)
                if depth is None:
                    continue
                depths[idx] = depth

                point_cam = pixel_to_camera(center, depth, camera_matrix)
                translations[idx] = point_cam

                if det.label in world_points:
                    object_points.append(world_points[det.label])
                    image_points.append(center)

            pose = solve_pnp(object_points, image_points, camera_matrix, dist_coeffs, use_ransac=use_ransac)
            if pose:
                rvec, tvec, inliers = pose
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                euler = rotation_matrix_to_euler(rotation_matrix)
                print(f"[PnP] Camera pose w.r.t world: tvec={tvec.ravel()} (m), Euler(ZYX)={euler} (rad), inliers={len(inliers)}/{len(object_points)}")
            elif object_points:
                print("[PnP] Not enough correspondences or solvePnP failed.")

            # Output per-fruit transforms (fruit center in camera frame)
            for idx, point in translations.items():
                T = np.eye(4, dtype=np.float32)
                T[:3, 3] = point
                depth_val = depths.get(idx, float("nan"))
                print(f"Fruit[{idx}] label='{detections[idx].label}' depth={depth_val:.3f} m pos_cam={point} \nT_cam_fruit=\n{T}")

            # Optional JSON-lines export for downstream fusion
            export_path = ns.get("export_json_path", "")
            if export_path:
                try:
                    payload = {
                        "frame_index": frame_count,
                        "fruits": [
                            {
                                "index": int(idx),
                                "label": str(detections[idx].label),
                                "depth_m": float(depths.get(idx, float("nan"))),
                                "pos_cam": [float(x) for x in translations[idx].tolist()],
                                "T_cam_fruit": np.eye(4, dtype=float).tolist() if idx not in translations else _t_from_pos(translations[idx]).tolist(),
                            }
                            for idx in sorted(translations.keys())
                        ],
                    }
                    with open(export_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(payload) + "\n")
                except Exception as e:
                    print(f"Export error: {e}")

            if display:
                vis = annotate_frame(color_image, detections, depths, translations)
                cv2.imshow("PnP Estimation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

    if display:
        cv2.destroyAllWindows()


def _maybe_yaml(path: Path) -> Dict:
    if not yaml:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _t_from_pos(p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


# ---------- Interactive helpers and discovery ----------

DEFAULT_CALIB_DIR = REPO_ROOT / "pickafresa_vision" / "camera_calibration"


def _discover_or_prompt_intrinsics() -> Path:
    """Find latest 'calib*.yaml' in camera_calibration, else prompt user."""
    candidates = []
    if DEFAULT_CALIB_DIR.exists():
        for p in DEFAULT_CALIB_DIR.glob("calib*.yaml"):
            ts = _timestamp_from_name(p.name)
            candidates.append((ts, p))
    intr_path: Optional[Path] = None
    if candidates:
        # Sort by timestamp (fallback to mtime if None)
        def sort_key(item):
            ts, p = item
            return ts or datetime.fromtimestamp(p.stat().st_mtime)
        candidates.sort(key=sort_key, reverse=True)
        intr_path = candidates[0][1]
        print(f"Using intrinsics: {intr_path}")
    else:
        print(f"No 'calib*.yaml' found in {DEFAULT_CALIB_DIR}")
    while intr_path is None or not intr_path.exists():
        user = input("Enter path to camera intrinsics YAML: ").strip()
        cand = Path(user)
        if cand.exists():
            intr_path = cand
        else:
            print("Path not found. Please try again.")
    return intr_path


def _timestamp_from_name(name: str) -> Optional[datetime]:
    m = re.search(r"(\d{8}_\d{6})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _yes_no_input(prompt: str, default: bool = False) -> bool:
    resp = input(prompt).strip().lower()
    if resp == "":
        return default
    return resp in {"y", "yes"}


def _int_input(prompt: str, default: int) -> int:
    txt = input(prompt).strip()
    if txt == "":
        return default
    try:
        return int(txt)
    except Exception:
        return default


def _prompt_detection_provider(ns: Dict) -> Optional[Callable[[np.ndarray], List[Detection]]]:
    last_choice = str(ns.get("det_source", "1"))
    print("Choose detection source:\n  1) YOLOv11 model (.pt)\n  2) Offline detections JSON\n  3) None (exit)")
    choice = input(f"Select [1/2/3] [{last_choice}]: ").strip() or last_choice
    if choice == "1":
        model_path = input(f"Path to YOLO .pt [{ns.get('model_path','')}]: ").strip() or str(ns.get('model_path',''))
        target_class = input(f"Target class label or index (blank=all) [{ns.get('target_class','')}]: ").strip() or (ns.get('target_class') or None)
        try:
            conf = float(input(f"Confidence threshold [{ns.get('confidence',0.25)}]: ").strip() or ns.get('confidence',0.25))
        except Exception:
            conf = 0.25
        # persist
        cfg = load_config()
        update_namespace(cfg, "bbox_depth_auto_pnp_calc", {"det_source": "1", "model_path": model_path, "target_class": target_class or "", "confidence": conf})
        return YOLODetectionProvider(Path(model_path), target_class, conf)
    if choice == "2":
        json_path = input(f"Path to detections JSON [{ns.get('json_path','')}]: ").strip() or str(ns.get('json_path',''))
        cfg = load_config()
        update_namespace(cfg, "bbox_depth_auto_pnp_calc", {"det_source": "2", "json_path": json_path})
        return JSONDetectionProvider(Path(json_path))
    # ask export preference once
    cfg = load_config()
    ns = get_namespace(cfg, "bbox_depth_auto_pnp_calc")
    if not ns.get("export_json_path") and _yes_no_input("Export fruit transforms to JSON-lines file for fusion? [y/N]: "):
        default_path = str((REPO_ROOT / "fruits_stream.jsonl").resolve())
        path = input(f"Export path [{default_path}]: ").strip() or default_path
        update_namespace(cfg, "bbox_depth_auto_pnp_calc", {"export_json_path": path})
    return None


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convert a rotation matrix to ZYX Euler angles."""

    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0
    return x, y, z


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Deprecated: arguments are no longer used (interactive mode only)."""
    # Kept for backward compatibility with callers that still import parse_args
    return argparse.ArgumentParser(add_help=False).parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        run_interactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
