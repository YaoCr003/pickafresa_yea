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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


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
    if "camera_matrix" not in data:
        raise KeyError(f"camera_matrix not found in {origin}")

    cm = data["camera_matrix"]
    if isinstance(cm, dict) and "data" in cm:
        arr = np.array(cm["data"], dtype=np.float32).reshape(int(cm.get("rows", 3)), int(cm.get("cols", 3)))
    else:
        arr = np.array(cm, dtype=np.float32).reshape(3, 3)

    if arr.shape != (3, 3):
        raise ValueError(f"camera_matrix in {origin} must be 3x3")
    return arr


def _extract_dist_coeffs(data: Dict) -> np.ndarray:
    coeffs = data.get("dist_coeffs", [0, 0, 0, 0, 0])
    if isinstance(coeffs, dict) and "data" in coeffs:
        arr = np.array(coeffs["data"], dtype=np.float32)
    else:
        arr = np.array(coeffs, dtype=np.float32)
    return arr.reshape((-1, 1))


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
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO-based detection")

        self.model = YOLO(str(model_path))
        self.target_class = target_class
        self.confidence = confidence

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False, conf=self.confidence)
        detections: List[Detection] = []

        for result in results:
            if not hasattr(result, "boxes"):
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                cls_idx = int(box.cls.item())
                label = self.model.names.get(cls_idx, str(cls_idx))

                if self.target_class and label != self.target_class and str(cls_idx) != self.target_class:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=label,
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
    intrinsics_path = Path(args.intrinsics)
    cached_intrinsics_yaml = _maybe_yaml(intrinsics_path)
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_path)
    world_points: Dict[str, np.ndarray] = {}

    if args.world_points:
        world_points = load_world_points(Path(args.world_points))
    elif cached_intrinsics_yaml.get("world_points"):
        world_points = _extract_world_points(cached_intrinsics_yaml, intrinsics_path)

    provider = build_detection_provider(args)
    sampler = DepthSampler(window=args.depth_window, min_valid=args.min_valid_depth_samples)

    if provider is None:
        print("Warning: no detection provider configured; pass --model or --detections.", file=sys.stderr)

    with RealSenseStream((args.width, args.height), args.fps) as stream:
        print(f"Depth scale: {stream.depth_scale:.6f} meters/unit")
        frame_count = 0
        while args.max_frames is None or frame_count < args.max_frames:
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

            pose = solve_pnp(object_points, image_points, camera_matrix, dist_coeffs, use_ransac=not args.disable_ransac)
            if pose:
                rvec, tvec, inliers = pose
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                print(
                    f"[PnP] tvec={tvec.ravel()} (m)  "
                    f"Euler={rotation_matrix_to_euler(rotation_matrix)} (rad)  "
                    f"inliers={len(inliers)}/{len(object_points)}"
                )
            elif object_points:
                print("[PnP] Not enough correspondences or solvePnP failed.")

            for idx, point in translations.items():
                print(f"Detection {idx}: depth={depths.get(idx, float('nan')):.3f} m, camera_point={point}")

            if args.display:
                vis = annotate_frame(color_image, detections, depths, translations)
                cv2.imshow("PnP Estimation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

    if args.display:
        cv2.destroyAllWindows()


def _maybe_yaml(path: Path) -> Dict:
    if not yaml:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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
    parser = argparse.ArgumentParser(description="PnP pose estimation using RealSense depth and YOLO detections.")
    parser.add_argument("--intrinsics", required=True, help="Path to YAML file with camera_matrix and dist_coeffs.")
    parser.add_argument("--world-points", help="Optional YAML with world_points for PnP correspondences.")
    parser.add_argument("--model", help="YOLO model path (requires ultralytics).")
    parser.add_argument("--target-class", help="Label or class index to filter YOLO detections.")
    parser.add_argument("--detections", help="JSON file with offline detections per frame.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum confidence for YOLO detections.")
    parser.add_argument("--depth-window", type=int, default=5, help="Square window size for depth median sampling.")
    parser.add_argument(
        "--min-valid-depth-samples",
        type=int,
        default=3,
        dest="min_valid_depth_samples",
        help="Minimum valid depth samples to accept a measurement.",
    )
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate.")
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
