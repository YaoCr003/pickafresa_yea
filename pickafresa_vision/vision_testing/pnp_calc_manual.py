"""
Interactive PnP testing utility that lets the user draw bounding boxes manually.

This script mirrors the pipeline from ``pnp_calc.py`` but replaces YOLO detections
with OpenCV ROI selections. It is useful for validating depth sampling, camera
intrinsics, and the PnP pose solver without loading a detector model.

Controls:
    d   Draw ROI on the current frame (uses ``cv2.selectROI``)
    r   Remove the most recently added ROI
    c   Clear all ROIs
    q   Quit
"""

from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path so 'pickafresa_vision' package can be imported
# This allows running this script directly (python pnp_calc_manual.py) from any location.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_vision.vision_nodes.bbox_depth_auto_pnp_calc import (
    Detection,
    DepthSampler,
    RealSenseStream,
    annotate_frame,
    load_camera_intrinsics,
    load_world_points,
    pixel_to_camera,
    rotation_matrix_to_euler,
    solve_pnp,
    _extract_world_points,
    _maybe_yaml,
    _discover_or_prompt_intrinsics,
    _yes_no_input,
    _int_input,
)
from pickafresa_vision.vision_tools.config_store import load_config, get_namespace, update_namespace


def run(args: argparse.Namespace) -> None:
    # Intrinsics discovery / prompt
    intrinsics_path = Path(args.intrinsics) if args.intrinsics else _discover_or_prompt_intrinsics()
    cached_intrinsics_yaml = _maybe_yaml(intrinsics_path)
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_path)
    world_points: Dict[str, np.ndarray] = {}

    # Persisted config defaults
    cfg = load_config()
    ns = get_namespace(cfg, "pnp_calc_manual")

    if args.world_points:
        world_points = load_world_points(Path(args.world_points))
    elif cached_intrinsics_yaml.get("world_points"):
        world_points = _extract_world_points(cached_intrinsics_yaml, intrinsics_path)
    else:
        if _yes_no_input("Provide a world-points YAML for full PnP (optional)? [y/N]: "):
            wp = input(f"Enter path to world_points YAML [{ns.get('world_points','')}]: ").strip() or str(ns.get('world_points',''))
            if wp:
                try:
                    world_points = load_world_points(Path(wp))
                    update_namespace(cfg, "pnp_calc_manual", {"world_points": wp})
                except Exception as e:
                    print(f"Could not load world points: {e}")

    # Sampler/stream interactive config
    depth_window = args.depth_window or int(ns.get("depth_window", _int_input("Depth median window (odd) [5]: ", 5)))
    min_valid = args.min_valid_depth_samples or int(ns.get("min_valid_depth_samples", _int_input("Min valid depth samples [3]: ", 3)))
    width = args.width or int(ns.get("width", _int_input("Color/Depth width [640]: ", 640)))
    height = args.height or int(ns.get("height", _int_input("Color/Depth height [480]: ", 480)))
    fps = args.fps or int(ns.get("fps", _int_input("Stream FPS [30]: ", 30)))
    max_frames_default = int(ns.get("max_frames", 0))
    max_frames = args.max_frames if args.max_frames is not None else _int_input("Max frames (0 = infinite) [0]: ", max_frames_default)
    max_frames = None if max_frames == 0 else max_frames
    use_ransac = (not args.disable_ransac) if args.disable_ransac is not None else bool(ns.get("use_ransac", _yes_no_input("Use RANSAC for PnP? [Y/n]: ", default=True)))

    # Persist
    update_namespace(cfg, "pnp_calc_manual", {
        "depth_window": depth_window,
        "min_valid_depth_samples": min_valid,
        "width": width,
        "height": height,
        "fps": fps,
        "max_frames": (0 if max_frames is None else max_frames),
        "use_ransac": use_ransac,
    })

    sampler = DepthSampler(window=depth_window, min_valid=min_valid)
    label_cycle = build_label_cycle(args, world_points)
    detections: List[Detection] = []

    print("Manual PnP test started.")
    print("Keys: d=draw ROI, r=remove last ROI, c=clear ROIs, q=quit.")
    if world_points:
        print(f"Loaded {len(world_points)} world points: {', '.join(world_points.keys())}")
    else:
        print("No world points provided; PnP will only estimate camera-frame points.")

    with RealSenseStream((width, height), fps) as stream:
        print(f"Depth scale: {stream.depth_scale:.6f} meters/unit")
        frame_count = 0
        while max_frames is None or frame_count < max_frames:
            color_image, depth_frame = stream.get_aligned_frames()

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

            pose = solve_pnp(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                use_ransac=use_ransac,
            )
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
                depth_val = depths.get(idx, float("nan"))
                det = detections[idx]
                T = np.eye(4, dtype=np.float32)
                T[:3, 3] = point
                print(f"ROI {idx} ({det.label}): depth={depth_val:.3f} m, camera_point={point} \nT_cam_fruit=\n{T}")

            vis = annotate_frame(color_image, detections, depths, translations)
            overlay_instructions(vis, detections, args.max_rois)
            cv2.imshow("PnP Manual Test", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("d"):
                if len(detections) >= args.max_rois:
                    print(f"Maximum of {args.max_rois} ROIs reached; press 'c' to clear.")
                else:
                    new_detection = select_detection(color_image, label_cycle)
                    if new_detection:
                        detections.append(new_detection)
            elif key == ord("r"):
                if detections:
                    removed = detections.pop()
                    print(f"Removed ROI for label '{removed.label}'.")
            elif key == ord("c"):
                if detections:
                    detections.clear()
                    print("Cleared all ROIs.")

            frame_count += 1

    cv2.destroyAllWindows()


def build_label_cycle(args: argparse.Namespace, world_points: Dict[str, np.ndarray]) -> Iterator[str]:
    if args.label_order:
        return cycle(args.label_order)
    if world_points:
        return cycle(sorted(world_points.keys()))
    return cycle([args.default_label])


def select_detection(frame: np.ndarray, labels: Iterator[str]) -> Optional[Detection]:
    label = next(labels)
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("ROI selection cancelled.")
        return None

    bbox = (x, y, x + w, y + h)
    print(f"Added ROI for label '{label}' at {bbox}.")
    return Detection(bbox=bbox, label=label, confidence=1.0)


def overlay_instructions(frame: np.ndarray, detections: List[Detection], max_rois: int) -> None:
    lines = [
        "d: draw ROI  r: remove last  c: clear  q: quit",
        f"Active ROIs: {len(detections)}/{max_rois}",
    ]

    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual ROI-based PnP testing using RealSense depth."
    )
    parser.add_argument("--intrinsics", help="Path to YAML with intrinsics. If omitted, auto-discover or prompt.")
    parser.add_argument("--world-points", help="Optional YAML with world_points for PnP correspondences.")
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
    parser.add_argument(
        "--label-order",
        nargs="+",
        help="Sequence of labels to assign in order for each new ROI (cycles when exhausted).",
    )
    parser.add_argument(
        "--default-label",
        default="strawberry",
        help="Label used when no label order is provided.",
    )
    parser.add_argument("--max-rois", type=int, default=6, help="Maximum number of concurrent ROIs.")
    parser.add_argument("--disable-ransac", action="store_true", help="Use solvePnP instead of solvePnPRansac.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
