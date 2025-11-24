"""
Data Persistence Module for Fruit Pose Estimation

Handles saving of images and JSON metadata for pose estimation results.
Provides consistent file naming and organization for captured data.

File Naming Convention:
- YYYYMMDD_HHMMSS_raw.png: Raw RGB image from camera
- YYYYMMDD_HHMMSS_bbox.png: Annotated image with bounding boxes and pose info
- YYYYMMDD_HHMMSS_data.json: JSON metadata with detections, poses, and system info

All files are saved to pickafresa_vision/captures/ by default.

Usage:
    from pickafresa_vision.vision_tools.data_persistence import DataSaver
    
    # Initialize saver
    saver = DataSaver()
    
    # Save a capture session
    timestamp = saver.save_capture(
        color_image=color_img,
        depth_frame=depth_frame,
        results=pose_results,
        intrinsics=camera_intrinsics,
        model_path="/path/to/model.pt"
    )
    
    print(f"Saved to captures/{timestamp}_*")

@aldrick-t, 2025
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False
    from typing import Any as rs  # type: ignore

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DataSaver:
    """
    Handles saving of images and metadata for fruit pose estimation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize data saver.
        
        Args:
            output_dir: Directory to save files. If None, uses default captures/
        """
        if output_dir is None:
            output_dir = REPO_ROOT / "pickafresa_vision" / "captures"
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_timestamp(self) -> str:
        """Generate timestamp string for filenames: YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_raw_image(
        self,
        color_image: np.ndarray,
        timestamp: Optional[str] = None
    ) -> Tuple[str, Path]:
        """
        Save raw RGB image.
        
        Args:
            color_image: RGB numpy array (H, W, 3)
            timestamp: Optional timestamp string. If None, generates new one.
        
        Returns:
            (timestamp, filepath): timestamp used and path to saved file
        """
        if timestamp is None:
            timestamp = self._generate_timestamp()
        
        filename = f"{timestamp}_raw.png"
        filepath = self.output_dir / filename
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr_image)
        
        return timestamp, filepath
    
    def save_annotated_image(
        self,
        color_image: np.ndarray,
        results: List[Any],
        bboxes_cxcywh: List[Tuple[float, float, float, float]],
        timestamp: Optional[str] = None,
        show_all_detections: bool = False,
        depth_frame: Optional[Any] = None,
    ) -> Tuple[str, Path]:
        """
        Save annotated image with bounding boxes and pose information.
        
        Args:
            color_image: RGB numpy array (H, W, 3)
            results: List of PoseEstimationResult objects
            bboxes_cxcywh: List of bboxes in (cx, cy, w, h) format
            timestamp: Optional timestamp string. If None, generates new one.
            show_all_detections: If True, show failed detections in red
        
        Returns:
            (timestamp, filepath): timestamp used and path to saved file
        """
        if timestamp is None:
            timestamp = self._generate_timestamp()
        
        filename = f"{timestamp}_bbox.png"
        filepath = self.output_dir / filename
        
        # Create annotated image
        annotated = self._annotate_image(
            color_image,
            results,
            bboxes_cxcywh,
            show_all_detections,
            depth_frame=depth_frame,
        )
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr_image)
        
        return timestamp, filepath
    
    def _annotate_image(
        self,
        color_image: np.ndarray,
        results: List[Any],
        bboxes_cxcywh: List[Tuple[float, float, float, float]],
        show_all: bool,
        depth_frame: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Create annotated image with bounding boxes and pose info.
        
        Args:
            color_image: RGB numpy array
            results: List of PoseEstimationResult objects
            bboxes_cxcywh: List of bboxes in (cx, cy, w, h) format
            show_all: If True, show failed detections in red
        
        Returns:
            Annotated RGB image
        """
        img = color_image.copy()

        def cxcywh_to_xyxy(bb: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
            cx, cy, w, h = bb
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            return x1, y1, x2, y2

        def median_depth_in_bbox(df: Any, bb_xyxy: Tuple[int, int, int, int]) -> Optional[float]:
            if not HAVE_REALSENSE or df is None:
                return None
            try:
                x1, y1, x2, y2 = bb_xyxy
                # clamp to frame
                width = df.get_width()
                height = df.get_height()
                x1 = max(0, min(width - 1, x1))
                x2 = max(0, min(width - 1, x2))
                y1 = max(0, min(height - 1, y1))
                y2 = max(0, min(height - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    return None
                vals: List[float] = []
                # Sample a grid (to reduce cost) instead of every pixel
                step_x = max(1, (x2 - x1) // 20)
                step_y = max(1, (y2 - y1) // 20)
                for yy in range(y1, y2 + 1, step_y):
                    for xx in range(x1, x2 + 1, step_x):
                        d = df.get_distance(int(xx), int(yy))
                        if d > 0:
                            vals.append(d)
                if not vals:
                    return None
                return float(np.median(vals))
            except Exception:
                return None
        
        for result, bbox in zip(results, bboxes_cxcywh):
            # Support both dict and object result formats
            if isinstance(result, dict):
                success = result.get("success", False)
                class_name = result.get("class_name", "cls")
                conf = float(result.get("confidence", 0.0))
                median_depth = result.get("median_depth")
                position_cam = result.get("position_cam")
                error_reason = result.get("error_reason", "")
            else:
                success = bool(getattr(result, "success", False))
                class_name = getattr(result, "class_name", "cls")
                conf = float(getattr(result, "confidence", 0.0))
                median_depth = getattr(result, "median_depth", None)
                position_cam = getattr(result, "position_cam", None)
                error_reason = getattr(result, "error_reason", "")
            
            # If show_all is False and PnP failed, skip
            if not show_all and not success:
                continue
            
            # Determine color
            color = (0, 255, 0) if success else (255, 0, 0)

            # Convert cxcywh to xyxy for drawing
            x1, y1, x2, y2 = cxcywh_to_xyxy(bbox)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text (always show class + conf + depth if available)
            depth_m = None
            if median_depth is not None:
                depth_m = float(median_depth)
            elif depth_frame is not None:
                depth_m = median_depth_in_bbox(depth_frame, (x1, y1, x2, y2))

            depth_text = f" | {depth_m*1000:.0f}mm" if depth_m is not None else ""
            label = f"{class_name} {conf:.2f}{depth_text}"

            # Coordinates or reason line
            if success and position_cam is not None:
                x, y, z = position_cam
                coord_text = f"xyz:[{x:.3f},{y:.3f},{z:.3f}]m"
            else:
                coord_text = (error_reason[:30] if error_reason else "")
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw coordinates if successful
            if coord_text:
                (cw, ch), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y2 + 2), (x1 + cw + 2, y2 + ch + 8), color, -1)
                cv2.putText(img, coord_text, (x1 + 1, y2 + ch + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def save_metadata(
        self,
        results: List[Any],
        intrinsics: Any,
        color_image: np.ndarray,
        model_path: Optional[str] = None,
        timestamp: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Path]:
        """
        Save JSON metadata for a capture session.
        
        Args:
            results: List of PoseEstimationResult objects
            intrinsics: CameraIntrinsics object
            color_image: RGB image (used for resolution)
            model_path: Path to YOLO model used
            timestamp: Optional timestamp string. If None, generates new one.
            extra_data: Optional additional data to include in JSON
        
        Returns:
            (timestamp, filepath): timestamp used and path to saved file
        """
        if timestamp is None:
            timestamp = self._generate_timestamp()
        
        filename = f"{timestamp}_data.json"
        filepath = self.output_dir / filename
        
        # Build metadata dictionary
        metadata = {
            "timestamp": timestamp,
            "timestamp_iso": datetime.now().isoformat(),
            "resolution": {
                "width": color_image.shape[1],
                "height": color_image.shape[0]
            },
            "camera_intrinsics": intrinsics.to_dict() if hasattr(intrinsics, 'to_dict') else str(intrinsics),
            "model_path": model_path,
            "detections": [result.to_dict() for result in results],
            "summary": {
                "total_detections": len(results),
                "successful_poses": sum(1 for r in results if r.success),
                "failed_poses": sum(1 for r in results if not r.success),
            }
        }
        
        # Add extra data if provided
        if extra_data:
            metadata.update(extra_data)
        
        # Save JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return timestamp, filepath
    
    def save_capture(
        self,
        color_image: np.ndarray,
        depth_frame: Any,
        results: List[Any],
        bboxes_cxcywh: List[Tuple[float, float, float, float]],
        intrinsics: Any,
        model_path: Optional[str] = None,
        save_raw: bool = True,
        save_annotated: bool = True,
        save_json: bool = True,
        show_all_detections: bool = False,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete capture session (images + metadata).
        
        Args:
            color_image: RGB numpy array
            depth_frame: RealSense depth frame (not currently used, kept for future)
            results: List of PoseEstimationResult objects
            bboxes_cxcywh: List of bboxes in (cx, cy, w, h) format
            intrinsics: CameraIntrinsics object
            model_path: Path to YOLO model used
            save_raw: Whether to save raw image
            save_annotated: Whether to save annotated image
            save_json: Whether to save JSON metadata
            show_all_detections: If True, show failed detections in red
            extra_data: Optional additional data to include in JSON
        
        Returns:
            timestamp: Timestamp string used for all files
        """
        # Generate single timestamp for all files
        timestamp = self._generate_timestamp()
        
        saved_files = []
        
        # Save raw image
        if save_raw:
            _, path = self.save_raw_image(color_image, timestamp)
            saved_files.append(path.name)
        
        # Save annotated image
        if save_annotated:
            _, path = self.save_annotated_image(color_image, results, bboxes_cxcywh, timestamp, show_all_detections, depth_frame=depth_frame)
            saved_files.append(path.name)
        
        # Save JSON metadata
        if save_json:
            _, path = self.save_metadata(results, intrinsics, color_image, model_path, timestamp, extra_data)
            saved_files.append(path.name)
        
        print(f"[OK] Saved {len(saved_files)} files to captures/:")
        for filename in saved_files:
            print(f"  - {filename}")
        
        return timestamp


def load_metadata(filepath: Path) -> Dict[str, Any]:
    """
    Load metadata from a saved JSON file.
    
    Args:
        filepath: Path to JSON metadata file
    
    Returns:
        Dictionary with metadata
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_captures_by_timestamp(timestamp: str, captures_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Find all files associated with a timestamp.
    
    Args:
        timestamp: Timestamp string (YYYYMMDD_HHMMSS)
        captures_dir: Directory to search. If None, uses default.
    
    Returns:
        Dictionary with keys 'raw', 'bbox', 'data' and Path values (or None if not found)
    """
    if captures_dir is None:
        captures_dir = REPO_ROOT / "pickafresa_vision" / "captures"
    
    return {
        'raw': next(captures_dir.glob(f"{timestamp}_raw.png"), None),
        'bbox': next(captures_dir.glob(f"{timestamp}_bbox.png"), None),
        'data': next(captures_dir.glob(f"{timestamp}_data.json"), None),
    }


if __name__ == "__main__":
    print("Data persistence module loaded successfully")
    print(f"Default output directory: {REPO_ROOT / 'pickafresa_vision' / 'captures'}")
