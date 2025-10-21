"""
PnP pose estimation for strawberry detections using RealSense depth.

The script ingests YOLO bounding boxes, samples depth around each detection,
and triangulates the fruit center in camera coordinates. When 3D world
references for the strawberries are available, it solves a full PnP to recover
the camera pose via OpenCV's ``solvePnP`` / ``solvePnPRansac`` routines.

Typical usage:
    python pnp_calc.py --intrinsics camera.yaml --model path/to/yolo.pt \
        --target-class strawberry --world-points orchard_layout.yaml --display

Camera intrinsics are expected in a YAML file containing either OpenCV's
FileStorage format or dictionaries with ``camera_matrix`` (3x3) and
``dist_coeffs`` (1xN). World point files may reuse the same YAML file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import importlib

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

# Local utilities
from pickafresa_vision.vision_tools.config_store import (
    load_config,
    save_config,
    get_namespace,
    update_namespace,
)
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
    """Context manager around an aligned RealSense color + depth stream."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ) -> None:
        self.resolution = resolution
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.0
        # Depth filters tuned for D435: Convert to disparity -> spatial -> temporal -> back -> hole fill
        self._filters: List = []

    def __enter__(self) -> "RealSenseStream":
        width, height = self.resolution
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        try:
            self._filters = [
                rs.disparity_transform(True),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.disparity_transform(False),
                rs.hole_filling_filter(),
            ]
            # Light tuning
            spatial: rs.spatial_filter = self._filters[1]
            spatial.set_option(rs.option.holes_fill, 3)
            temporal: rs.temporal_filter = self._filters[2]
            temporal.set_option(rs.option.alpha, 0.5)
        except Exception:
            self._filters = []
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None

    def get_aligned_frames(self) -> Tuple[np.ndarray, rs.depth_frame]:
        if not self.pipeline or not self.align:
            raise RuntimeError("RealSenseStream must be used as a context manager")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to retrieve frames from RealSense pipeline")

        # Apply depth filtering pipeline if available (improves D435 depth reliability)
        try:
            f = depth_frame
            for flt in self._filters:
                f = flt.process(f)
            depth_frame = f.as_depth_frame()  # type: ignore[attr-defined]
        except Exception:
            pass

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame


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
    """Optional CLI; if omitted, interactive prompts will be used."""
    parser = argparse.ArgumentParser(description="PnP pose estimation using RealSense depth and YOLO detections.", add_help=True)
    parser.add_argument("--intrinsics", help="Path to YAML with intrinsics. If omitted, auto-discover or prompt.")
    parser.add_argument("--world-points", help="Optional YAML with world_points for PnP correspondences.")
    parser.add_argument("--model", help="YOLO model path (requires ultralytics).")
    parser.add_argument("--target-class", help="Label or class index to filter YOLO detections.")
    parser.add_argument("--detections", help="JSON file with offline detections per frame.")
    parser.add_argument("--confidence", type=float, help="Minimum confidence for YOLO detections.")
    parser.add_argument("--depth-window", type=int, help="Square window size for depth median sampling.")
    parser.add_argument(
        "--min-valid-depth-samples",
        type=int,
        dest="min_valid_depth_samples",
        help="Minimum valid depth samples to accept a measurement.",
    )
    parser.add_argument("--width", type=int, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, help="Stream frame rate.")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for debugging.")
    parser.add_argument("--display", action="store_true", help="Show annotated frames.")
    parser.add_argument("--disable-ransac", action="store_true", help="Use solvePnP instead of solvePnPRansac.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
