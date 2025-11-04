#!/usr/bin/env python3
"""
Interactive Fruit Pose Estimation CLI Tool

Live camera preview with on-demand fruit pose estimation using:
- RealSense D435 camera for RGB + Depth
- YOLOv11 for strawberry detection  
- OpenCV PnP for 6DOF pose estimation

Features:
- Continuous live preview with RealSense camera
- Press 'c' to capture and process a frame
- Real-time detection and pose visualization
- Automatic data persistence (images + JSON)
- Terminal output of transformation matrices

Keyboard Controls:
    c - Capture frame and run detection + pose estimation
    q - Quit application
    s - Toggle saving to disk (default: ON)
    d - Toggle showing all detections including failures (default: OFF)
    h - Show help

Usage:
    python fruit_pose_cli.py [--model MODEL_PATH] [--intrinsics auto|realsense|yaml] [--no-save]

Team YEA, 2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture, CameraIntrinsics
from pickafresa_vision.vision_nodes.pnp_calc import FruitPoseEstimator, PoseEstimationResult
from pickafresa_vision.vision_nodes.inference_bbox import load_model, infer, Detection
from pickafresa_vision.vision_tools.data_persistence import DataSaver


class FruitPoseCLI:
    """
    Interactive CLI application for fruit pose estimation.
    """
    
    def __init__(
        self,
        model_path: Path,
        intrinsics_source: str = "auto",
        enable_save: bool = True
    ):
        """
        Initialize CLI application.
        
        Args:
            model_path: Path to YOLOv11 model (.pt file)
            intrinsics_source: "auto", "realsense", or "yaml"
            enable_save: Whether to save captures to disk by default
        """
        print("=" * 70)
        print("Fruit Pose Estimation - Interactive CLI")
        print("Team YEA, 2025")
        print("=" * 70)
        print()
        
        self.model_path = model_path
        self.intrinsics_source = intrinsics_source
        self.enable_save = enable_save
        self.show_all_detections = False
        self.live_overlay = True  # draw live detection bboxes on preview
        self.live_conf_threshold = 0.25
        # FPS tracking (EMA)
        self._fps_alpha = 0.12
        self.fps_ema = 0.0
        self._fps_last_time = time.time()
        
        # Initialize components
        print("[1/4] Loading YOLO model...")
        self.model = load_model(str(model_path))
        print(f"✓ Model loaded: {model_path.name}")
        
        print("[2/4] Initializing RealSense camera...")
        self.camera = RealSenseCapture()
        self.camera.start()
        print("✓ Camera started")
        
        print("[3/4] Loading camera intrinsics...")
        
        # If using 'auto' source, validate YAML intrinsics and fallback to RealSense SDK if corrupted
        if intrinsics_source == "auto":
            # First try YAML
            try:
                yaml_intrinsics = self.camera.get_intrinsics(source="yaml")
                # Validate: D435 should have fx/fy in range 600-900
                if yaml_intrinsics.fx > 2000 or yaml_intrinsics.fy > 2000 or yaml_intrinsics.fx < 400 or yaml_intrinsics.fy < 400:
                    print(f"⚠️  YAML intrinsics seem corrupted (fx={yaml_intrinsics.fx:.1f}, fy={yaml_intrinsics.fy:.1f})")
                    print("   Falling back to RealSense SDK intrinsics...")
                    self.intrinsics = self.camera.get_intrinsics(source="realsense")
                else:
                    self.intrinsics = yaml_intrinsics
            except Exception as e:
                print(f"⚠️  Failed to load YAML intrinsics: {e}")
                print("   Falling back to RealSense SDK intrinsics...")
                self.intrinsics = self.camera.get_intrinsics(source="realsense")
        else:
            # Explicit source requested
            self.intrinsics = self.camera.get_intrinsics(source=intrinsics_source)
        
        print(f"✓ Intrinsics loaded (fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f})")
        
        # Final validation warning
        if self.intrinsics.fx > 2000 or self.intrinsics.fy > 2000:
            print()
            print("⚠️  WARNING: Intrinsics still seem incorrect!")
            print(f"   fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            print(f"   Expected range for D435: fx/fy ≈ 600-900")
            print(f"   This will cause incorrect pose estimation!")
            print(f"   Try: --intrinsics realsense to force SDK intrinsics")
            print()
        
        print("[4/4] Initializing pose estimator...")
        self.pose_estimator = FruitPoseEstimator()
        print("✓ Pose estimator ready")
        
        # Data saver
        self.data_saver = DataSaver()
        
        # Statistics
        self.total_captures = 0
        self.total_detections = 0
        self.total_successful_poses = 0
        
        print()
        print("=" * 70)
        print("Ready! Press 'h' for help, 'c' to capture, 'q' to quit")
        print("=" * 70)
        print()

    def _bbox_median_depth(self, depth_frame, bbox_xyxy: Tuple[int, int, int, int]) -> Optional[float]:
        """Compute median depth in meters inside bbox (xyxy) on aligned depth_frame."""
        try:
            x1, y1, x2, y2 = bbox_xyxy
            w = depth_frame.get_width(); h = depth_frame.get_height()
            x1 = max(0, min(w - 1, int(x1))); x2 = max(0, min(w - 1, int(x2)))
            y1 = max(0, min(h - 1, int(y1))); y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                return None
            vals = []
            step_x = max(1, (x2 - x1) // 20)
            step_y = max(1, (y2 - y1) // 20)
            for yy in range(y1, y2 + 1, step_y):
                for xx in range(x1, x2 + 1, step_x):
                    d = depth_frame.get_distance(int(xx), int(yy))
                    if d > 0:
                        vals.append(d)
            if not vals:
                return None
            import numpy as _np
            return float(_np.median(vals))
        except Exception:
            return None

    def _draw_live_detections_overlay(
        self,
        frame_bgr: np.ndarray,
        color_image_rgb: np.ndarray,
        depth_frame
    ) -> None:
        """Run fast YOLO on current frame and draw bboxes + labels on frame_bgr."""
        try:
            # Ultralytics expects OpenCV BGR by default; our capture is RGB
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            dets, bboxes_xyxy = infer(
                self.model,
                color_image_bgr,
                conf=self.live_conf_threshold,
                bbox_format="xyxy",
                normalized=False,
            )
        except Exception as e:
            # Do not crash preview; just skip drawing for this frame
            return

        # Draw on BGR frame
        h, w = frame_bgr.shape[:2]
        for det, bb in zip(dets, bboxes_xyxy):
            x1, y1, x2, y2 = [int(v) for v in bb]
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{det.clazz} {det.confidence:.2f}"
            depth_m = self._bbox_median_depth(depth_frame, (x1, y1, x2, y2)) if depth_frame is not None else None
            if depth_m is not None:
                label += f" | {depth_m*1000:.0f}mm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def _print_help(self) -> None:
        """Print keyboard controls help."""
        print()
        print("=" * 70)
        print("KEYBOARD CONTROLS")
        print("=" * 70)
        print("  c - Capture frame and estimate poses")
        print("  s - Toggle save to disk (current: {})".format("ON" if self.enable_save else "OFF"))
        print("  d - Toggle show all detections (current: {})".format("ON" if self.show_all_detections else "OFF"))
        print("  b - Toggle live bbox overlay (current: {})".format("ON" if self.live_overlay else "OFF"))
        print("  h - Show this help")
        print("  q - Quit application")
        print("=" * 70)
        print()
    
    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw heads-up display on preview frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # HUD background
        cv2.rectangle(overlay, (5, 5), (min(450, w - 5), 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # HUD text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        texts = [
            f"Model: {self.model_path.name}",
            f"Intrinsics: {self.intrinsics_source}",
            f"Captures: {self.total_captures} | Detections: {self.total_detections} | Poses: {self.total_successful_poses}",
            f"FPS: {self.fps_ema:.1f} | Live conf: {self.live_conf_threshold:.2f}",
            f"Save: {'ON' if self.enable_save else 'OFF'} | Show All: {'ON' if self.show_all_detections else 'OFF'} | Live BBox: {'ON' if self.live_overlay else 'OFF'}",
            "Press 'c' to capture, 'h' for help, 'q' to quit"
        ]
        
        y = 25
        for text in texts:
            cv2.putText(frame, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += 20
    
    def _process_frame(
        self,
        color_image: np.ndarray,
        depth_frame
    ) -> Tuple[List[Detection], List[Tuple[float, float, float, float]], List[PoseEstimationResult]]:
        """
        Process a captured frame: run detection and pose estimation.
        
        Returns:
            (detections, bboxes_cxcywh, pose_results)
        """
        # Run YOLO inference (convert RGB->BGR for Ultralytics)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        detections, bboxes_cxcywh = infer(
            self.model,
            color_image_bgr,
            conf=0.25,
            bbox_format="cxcywh",
            normalized=False
        )
        
        if not detections:
            return [], [], []
        
        # Run pose estimation
        pose_results = self.pose_estimator.estimate_poses(
            color_image=color_image,
            depth_frame=depth_frame,
            detections=detections,
            bboxes_cxcywh=bboxes_cxcywh,
            camera_matrix=self.intrinsics.to_matrix(),
            dist_coeffs=self.intrinsics.distortion_coeffs
        )
        
        return detections, bboxes_cxcywh, pose_results
    
    def _print_results(self, results: List[PoseEstimationResult]) -> None:
        """Print pose estimation results to terminal."""
        print()
        print("=" * 70)
        print(f"POSE ESTIMATION RESULTS ({len(results)} detections)")
        print("=" * 70)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✓ Successful: {len(successful)} | ✗ Failed: {len(failed)}")
        print()
        
        # Print successful poses
        if successful:
            print("--- Successful Poses ---")
            for i, result in enumerate(successful, 1):
                x, y, z = result.position_cam
                print(f"{i}. {result.class_name} (conf={result.confidence:.3f})")
                print(f"   Position [m]: x={x:+.4f}, y={y:+.4f}, z={z:+.4f}")
                print(f"   Depth: median={result.median_depth:.3f}m, variance={result.depth_variance:.1%}")
                print(f"   T_cam_fruit:")
                for row in result.T_cam_fruit:
                    print(f"     [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f} {row[3]:+.4f}]")
                print()
        
        # Print failed poses
        if failed:
            print("--- Failed Poses ---")
            for i, result in enumerate(failed, 1):
                print(f"{i}. {result.class_name} (conf={result.confidence:.3f})")
                print(f"   Reason: {result.error_reason}")
                print()
        
        print("=" * 70)
        print()
    
    def _handle_capture(self, color_image: np.ndarray, depth_frame) -> None:
        """Handle a capture request: process and optionally save."""
        print(f"\n[Capture #{self.total_captures + 1}] Processing...")
        
        start_time = time.time()
        
        # Process frame
        detections, bboxes_cxcywh, results = self._process_frame(color_image, depth_frame)
        
        elapsed = time.time() - start_time
        
        # Update statistics
        self.total_captures += 1
        self.total_detections += len(detections)
        self.total_successful_poses += sum(1 for r in results if r.success)
        
        # Print results
        if results:
            self._print_results(results)
        else:
            print("No detections found.")
            print()
        
        # Save to disk if enabled
        if self.enable_save and results:
            timestamp = self.data_saver.save_capture(
                color_image=color_image,
                depth_frame=depth_frame,
                results=results,
                bboxes_cxcywh=bboxes_cxcywh,
                intrinsics=self.intrinsics,
                model_path=str(self.model_path),
                save_raw=True,
                save_annotated=True,
                save_json=True,
                show_all_detections=self.show_all_detections,
                extra_data={
                    "processing_time_seconds": elapsed,
                    "intrinsics_source": self.intrinsics_source
                }
            )
        
        print(f"Processing time: {elapsed:.3f}s")
        print()
    
    def run(self) -> None:
        """Run the interactive CLI application."""
        window_name = "Fruit Pose Estimation - Live Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Capture frame
                try:
                    color_image, depth_frame = self.camera.capture_frame()
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    time.sleep(0.1)
                    continue
                
                # Convert RGB to BGR for display
                display_frame = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR).copy()
                
                # Live detection overlay (optional)
                if self.live_overlay:
                    self._draw_live_detections_overlay(display_frame, color_image, depth_frame)

                # Draw HUD
                self._draw_hud(display_frame)
                
                # Show frame
                cv2.imshow(window_name, display_frame)

                # Update FPS
                now = time.time()
                dt = now - self._fps_last_time
                if dt > 0:
                    inst = 1.0 / dt
                    self.fps_ema = self._fps_alpha * inst + (1 - self._fps_alpha) * self.fps_ema
                self._fps_last_time = now
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    self._handle_capture(color_image, depth_frame)
                elif key == ord('s'):
                    self.enable_save = not self.enable_save
                    print(f"Save to disk: {'ON' if self.enable_save else 'OFF'}")
                elif key == ord('d'):
                    self.show_all_detections = not self.show_all_detections
                    print(f"Show all detections: {'ON' if self.show_all_detections else 'OFF'}")
                elif key == ord('b'):
                    self.live_overlay = not self.live_overlay
                    print(f"Live bbox overlay: {'ON' if self.live_overlay else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    # Increase live confidence threshold
                    self.live_conf_threshold = min(0.95, round(self.live_conf_threshold + 0.05, 2))
                    print(f"Live confidence threshold: {self.live_conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease live confidence threshold
                    self.live_conf_threshold = max(0.05, round(self.live_conf_threshold - 0.05, 2))
                    print(f"Live confidence threshold: {self.live_conf_threshold:.2f}")
                elif key == ord('h'):
                    self._print_help()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.camera.stop()
            
            print()
            print("=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Total Captures: {self.total_captures}")
            print(f"Total Detections: {self.total_detections}")
            print(f"Successful Poses: {self.total_successful_poses}")
            print(f"Success Rate: {100 * self.total_successful_poses / max(1, self.total_detections):.1f}%")
            print("=" * 70)
            print("Goodbye!")
            print()


def _discover_models() -> List[Path]:
    """Discover available YOLO model files."""
    models_dir = REPO_ROOT / "pickafresa_vision" / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.pt"))


def _prompt_select_model(models: List[Path]) -> Optional[Path]:
    """Interactively prompt user to select a model."""
    if not models:
        print("Error: No .pt model files found in pickafresa_vision/models/")
        return None
    
    print()
    print("Available Models:")
    print("-" * 70)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    print("-" * 70)
    
    while True:
        choice = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive fruit pose estimation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fruit_pose_cli.py
  python fruit_pose_cli.py --model path/to/model.pt
  python fruit_pose_cli.py --intrinsics yaml --no-save
        """
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to YOLOv11 model file (.pt). If not specified, will prompt."
    )
    
    parser.add_argument(
        "--intrinsics",
        choices=["auto", "realsense", "yaml"],
        default="auto",
        help="Intrinsics source: auto (YAML then RealSense), realsense (SDK only), yaml (file only)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable automatic saving of captures"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Select model
    model_path = args.model
    if model_path is None or not model_path.exists():
        models = _discover_models()
        model_path = _prompt_select_model(models)
        if model_path is None:
            return 1
    
    # Run CLI
    try:
        cli = FruitPoseCLI(
            model_path=model_path,
            intrinsics_source=args.intrinsics,
            enable_save=not args.no_save
        )
        cli.run()
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
