"""
Example usage of the new FruitPoseEstimator (PnP) API.

This script demonstrates two paths:
  1) Live: capture one aligned color+depth frame from an Intel RealSense,
     run YOLO detections, then estimate 6-DoF fruit poses via PnP.
  2) Offline: load an RGB image and a depth map from disk, run detections,
     then estimate poses using a lightweight depth-frame adapter.

What the script prints:
  - Per-detection pretty output including success/failure, position, rvec/tvec,
    and the 4x4 transform T_cam_fruit.
  - Optionally, it can dump the raw API outputs (PoseEstimationResult.to_dict)
    to a JSON file for downstream analysis.

Configuration:
  - All parameters (model path, confidence, iou, etc.) are read from:
      pickafresa_vision/configs/objd_config.yaml
  - Camera intrinsics use source="auto" (YAML first, validate against RealSense).
    If auto fails in offline mode (no camera), this script falls back to YAML.

Usage (live):
  python pickafresa_vision/vision_testing/example_api_usage.py --mode live [--json-out path]

Usage (offline):
  python pickafresa_vision/vision_testing/example_api_usage.py --mode offline \
         --image path/to/rgb.png --depth-npy path/to/depth_meters.npy [--json-out path]

Notes:
  - Offline depth must be in meters and the same resolution as the RGB image.
  - Class filtering (target_classes) stays as configured in pnp_calc_config.yaml.
  
by: Aldrick T, 2025 
for Team YEA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path for absolute package imports
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports
from pickafresa_vision.vision_nodes.pnp_calc import FruitPoseEstimator, PoseEstimationResult
from pickafresa_vision.vision_nodes.inference_bbox import load_model, infer, Detection
from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture

try:
    import yaml
    HAVE_YAML = True
except Exception:
    yaml = None  # type: ignore
    HAVE_YAML = False


def _load_objd_config() -> dict:
    """Load object detection configuration from YAML."""
    if not HAVE_YAML:
        raise RuntimeError("pyyaml is required. Install with: pip install pyyaml")
    
    cfg_path = REPO_ROOT / "pickafresa_vision" / "configs" / "objd_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Object detection config not found: {cfg_path}")
    
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {cfg_path}: {e}")


def _resolve_model_path(model_path_str: Optional[str]) -> str:
    """Resolve model path from config (can be relative or absolute)."""
    if not model_path_str:
        raise ValueError("model_path not specified in objd_config.yaml")
    
    model_path = Path(model_path_str)
    
    # If relative, resolve from repo root
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path_str).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return str(model_path)


class NumpyDepthFrame:
    """Lightweight adapter to provide a RealSense-like depth_frame API over a numpy array.

    Expected depth map units: meters. Shape: (H, W) or (H, W, 1).
    Methods used by DepthSampler: get_width(), get_height(), get_distance(x,y).
    """

    def __init__(self, depth_meters: np.ndarray):
        if depth_meters.ndim == 3 and depth_meters.shape[2] == 1:
            depth_meters = depth_meters[:, :, 0]
        if depth_meters.ndim != 2:
            raise ValueError("Depth map must be 2D (H, W) or (H, W, 1)")
        self._depth = depth_meters.astype(np.float32, copy=False)

    def get_width(self) -> int:
        return int(self._depth.shape[1])

    def get_height(self) -> int:
        return int(self._depth.shape[0])

    def get_distance(self, x: int, y: int) -> float:
        if x < 0 or y < 0 or y >= self._depth.shape[0] or x >= self._depth.shape[1]:
            return 0.0
        val = float(self._depth[y, x])
        # Negative/NaN treated as invalid (0)
        if not np.isfinite(val) or val <= 0.0:
            return 0.0
        return val


def result_to_dict(r: PoseEstimationResult) -> Dict[str, Any]:
    return r.to_dict()


def pretty_print_results(results: List[PoseEstimationResult]) -> None:
    if not results:
        print("No poses estimated (no depth or no detections).")
        return
    for i, r in enumerate(results, 1):
        print("-" * 60)
        status = "OK" if r.success else "FAIL"
        print(f"[{i}] class='{r.class_name}' id={r.class_id} conf={r.confidence:.2f} status={status}")
        print(f"    bbox(cx,cy,w,h)={tuple(int(x) for x in r.bbox_cxcywh)}")
        if r.depth_samples is not None:
            print(f"    depth_samples(m)={['{:.3f}'.format(d) for d in r.depth_samples]} median={r.median_depth:.3f if r.median_depth is not None else None} var={r.depth_variance:.3f if r.depth_variance is not None else None}")
        if r.success and r.position_cam is not None and r.T_cam_fruit is not None:
            px, py, pz = r.position_cam.tolist()
            print(f"    pos_cam(m) = [{px:.4f}, {py:.4f}, {pz:.4f}]")
            if r.rvec is not None:
                print(f"    rvec = {r.rvec.ravel()}")
            if r.tvec is not None:
                print(f"    tvec = {r.tvec.ravel()}")
            print("    T_cam_fruit =")
            for row in r.T_cam_fruit:
                print("        ", " ".join(f"{v: .4f}" for v in row))
        elif not r.success:
            print(f"    error: {r.error_reason}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FruitPoseEstimator example (live/offline)")
    p.add_argument("--mode", choices=["live", "offline"], default="live", help="Capture mode")
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON results")
    # Offline-only inputs
    p.add_argument("--image", type=str, default=None, help="RGB image path (offline)")
    p.add_argument("--depth-npy", type=str, default=None, help="Depth map .npy in meters, same size as image (offline)")
    return p.parse_args()


def _load_image_rgb(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def run_live(config: dict, json_out: Optional[Path]) -> None:
    print("=" * 60)
    print("FruitPoseEstimator demo (LIVE)")
    print("=" * 60)

    # Extract config parameters
    model_path = _resolve_model_path(config.get("model_path"))
    inference_cfg = config.get("inference", {})
    conf_threshold = inference_cfg.get("confidence", 0.25)
    iou_threshold = inference_cfg.get("iou", 0.45)
    max_det = inference_cfg.get("max_detections", 300)

    # Load detector (outside loop to avoid reloading)
    model = load_model(model_path)
    
    # Initialize estimator (outside loop)
    estimator = FruitPoseEstimator()
    
    # Initialize camera once (outside loop to avoid power state issues)
    with RealSenseCapture() as camera:
        # Get intrinsics source from config file
        intrinsics_source = camera.config.get("intrinsics", {}).get("source", "auto")
        print(f"[INFO] Using intrinsics source: {intrinsics_source}")
        intrinsics = camera.get_intrinsics(source=intrinsics_source)
        
        # Loop until successful detection and pose estimation
        attempt = 0
        while True:
            attempt += 1
            print(f"\n[Attempt {attempt}] Capturing frame and processing...")
            
            # Capture a frame from the already-initialized camera
            color_image, depth_frame = camera.capture_frame()

            # Run detection - return cxcywh in pixel coords
            detections, bboxes_cxcywh = infer(
                model,
                color_image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                bbox_format="cxcywh",
                normalized=False,
            )
            
            print(f"[INFO] Found {len(detections)} detections")

            # Estimate poses
            results = estimator.estimate_poses(
                color_image=color_image,
                depth_frame=depth_frame,
                detections=detections,
                bboxes_cxcywh=bboxes_cxcywh,
                camera_matrix=intrinsics.to_matrix(),
                dist_coeffs=intrinsics.distortion_coeffs,
            )

            print("\nResults:")
            pretty_print_results(results)
            
            # Check if we have any successful pose estimations
            successful_results = [r for r in results if r.success]
            if successful_results:
                print(f"\n[OK] Successfully estimated {len(successful_results)} pose(s)!")
                
                # Save results to default captures folder if json_out not specified
                if json_out is None:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_out = REPO_ROOT / "pickafresa_vision" / "captures" / f"{timestamp}_data.json"
                
                payload = [result_to_dict(r) for r in results]
                json_out.parent.mkdir(parents=True, exist_ok=True)
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"[OK] Saved JSON results -> {json_out}")
                
                break  # Exit loop on success
            else:
                print(f"No successful pose estimations. Retrying...")
                import time
                time.sleep(1)  # Brief delay before retry


def run_offline(config: dict, image_path: Path, depth_npy: Path, json_out: Optional[Path]) -> None:
    print("=" * 60)
    print("FruitPoseEstimator demo (OFFLINE)")
    print("=" * 60)

    # Extract config parameters
    model_path = _resolve_model_path(config.get("model_path"))
    inference_cfg = config.get("inference", {})
    conf_threshold = inference_cfg.get("confidence", 0.25)
    iou_threshold = inference_cfg.get("iou", 0.45)
    max_det = inference_cfg.get("max_detections", 300)

    # Load image and depth map
    color_image = _load_image_rgb(image_path)
    depth_m = np.load(depth_npy)
    if depth_m.ndim == 3 and depth_m.shape[2] == 1:
        depth_m = depth_m[:, :, 0]
    if depth_m.ndim != 2:
        raise ValueError("depth-npy must be a 2D array (H, W) or (H, W, 1)")
    if depth_m.shape[:2] != color_image.shape[:2]:
        raise ValueError("depth and image resolutions must match")
    depth_frame = NumpyDepthFrame(depth_m)

    # Load intrinsics: attempt auto, fallback to YAML only if no camera
    from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture
    intrinsics = None
    try:
        with RealSenseCapture() as cam:
            # Use 'auto' if available
            intrinsics = cam.get_intrinsics(source="auto")
    except Exception:
        # Fallback to YAML-only without requiring camera
        try:
            cam = RealSenseCapture()
            intrinsics = cam._get_intrinsics_yaml()  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"Failed to get intrinsics (auto and yaml): {e}")

    # Load detector
    model = load_model(model_path)

    # Run detection - return cxcywh in pixel coords
    detections, bboxes_cxcywh = infer(
        model,
        color_image,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        bbox_format="cxcywh",
        normalized=False,
    )

    # Estimate poses
    estimator = FruitPoseEstimator()
    results = estimator.estimate_poses(
        color_image=color_image,
        depth_frame=depth_frame,
        detections=detections,
        bboxes_cxcywh=bboxes_cxcywh,
        camera_matrix=intrinsics.to_matrix(),
        dist_coeffs=intrinsics.distortion_coeffs,
    )

    print("\nResults:")
    pretty_print_results(results)

    if json_out is not None:
        payload = [result_to_dict(r) for r in results]
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON results -> {json_out}")


def main() -> None:
    args = _parse_args()

    # Load object detection configuration
    config = _load_objd_config()

    json_out = Path(args.json_out) if args.json_out else None

    if args.mode == "live":
        run_live(config, json_out)
    else:
        # Offline requires image and depth npy
        if not args.image or not args.depth_npy:
            raise SystemExit("--image and --depth-npy are required for --mode offline")
        run_offline(
            config,
            Path(args.image),
            Path(args.depth_npy),
            json_out,
        )


if __name__ == "__main__":
    main()
