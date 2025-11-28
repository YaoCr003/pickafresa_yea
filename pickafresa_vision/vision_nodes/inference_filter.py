"""
Multi-Frame Detection Filtering and Averaging

Reduces detection uncertainty through temporal filtering across multiple frames.
Implements IoU-based tracking, outlier rejection, and intelligent merging of
overlapping detections to improve bbox accuracy and reduce X-axis offset errors.

Key Features:
- Multi-frame capture and averaging (configurable frame count)
- IoU-based detection tracking across frames
- Outlier rejection using IQR, standard deviation, or RANSAC
- Intelligent handling of overlapping/intersecting detections
- Support for multiple non-intersecting detections

Usage:
    from pickafresa_vision.vision_nodes.inference_filter import DetectionFilter
    
    # Initialize filter
    filter = DetectionFilter(config_path="path/to/config.yaml")
    
    # Process multiple frames
    averaged_detections = filter.process_multi_frame(
        capture_func=lambda: camera.capture_frame(),
        inference_func=lambda frame: model.predict(frame)
    )

# @aldrick-t, 2025
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    yaml = None  # type: ignore

# Setup logging
LOG_DIR = REPO_ROOT / "pickafresa_vision" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "inference_filter.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),  # Overwrite on start
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection from one frame."""
    bbox_cxcywh: Tuple[float, float, float, float]  # (cx, cy, w, h)
    confidence: float
    class_name: str
    class_id: int
    frame_idx: int  # Which frame this detection came from
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert bbox to (x1, y1, x2, y2) format."""
        cx, cy, w, h = self.bbox_cxcywh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)
    
    def area(self) -> float:
        """Calculate bbox area."""
        _, _, w, h = self.bbox_cxcywh
        return w * h


@dataclass
class DetectionTrack:
    """Track of a detection across multiple frames."""
    detections: List[Detection] = field(default_factory=list)
    track_id: int = 0
    
    def add_detection(self, detection: Detection):
        """Add a detection to this track."""
        self.detections.append(detection)
    
    def get_average_bbox(self, outlier_method: str = "iqr", **kwargs) -> Tuple[float, float, float, float]:
        """
        Calculate average bbox with outlier rejection.
        
        Args:
            outlier_method: "iqr", "std_dev", or "ransac"
            **kwargs: Parameters for outlier rejection method
        
        Returns:
            Averaged (cx, cy, w, h)
        """
        if not self.detections:
            raise ValueError("No detections in track")
        
        # Extract bbox components
        cx_list = [d.bbox_cxcywh[0] for d in self.detections]
        cy_list = [d.bbox_cxcywh[1] for d in self.detections]
        w_list = [d.bbox_cxcywh[2] for d in self.detections]
        h_list = [d.bbox_cxcywh[3] for d in self.detections]
        
        # Apply outlier rejection
        if outlier_method == "iqr":
            cx_clean = self._reject_outliers_iqr(cx_list, kwargs.get("iqr_multiplier", 1.5))
            cy_clean = self._reject_outliers_iqr(cy_list, kwargs.get("iqr_multiplier", 1.5))
            w_clean = self._reject_outliers_iqr(w_list, kwargs.get("iqr_multiplier", 1.5))
            h_clean = self._reject_outliers_iqr(h_list, kwargs.get("iqr_multiplier", 1.5))
        
        elif outlier_method == "std_dev":
            cx_clean = self._reject_outliers_std(cx_list, kwargs.get("std_dev_threshold", 2.0))
            cy_clean = self._reject_outliers_std(cy_list, kwargs.get("std_dev_threshold", 2.0))
            w_clean = self._reject_outliers_std(w_list, kwargs.get("std_dev_threshold", 2.0))
            h_clean = self._reject_outliers_std(h_list, kwargs.get("std_dev_threshold", 2.0))
        
        elif outlier_method == "ransac":
            # RANSAC is more complex - use IQR as robust fallback
            logger.warning("RANSAC outlier rejection not yet implemented, using IQR")
            cx_clean = self._reject_outliers_iqr(cx_list, 1.5)
            cy_clean = self._reject_outliers_iqr(cy_list, 1.5)
            w_clean = self._reject_outliers_iqr(w_list, 1.5)
            h_clean = self._reject_outliers_iqr(h_list, 1.5)
        
        else:
            # No outlier rejection - use all values
            cx_clean, cy_clean, w_clean, h_clean = cx_list, cy_list, w_list, h_list
        
        # Calculate means
        cx_avg = float(np.mean(cx_clean))
        cy_avg = float(np.mean(cy_clean))
        w_avg = float(np.mean(w_clean))
        h_avg = float(np.mean(h_clean))
        
        logger.debug(f"Track {self.track_id}: Averaged {len(self.detections)} detections "
                    f"(kept {len(cx_clean)}/{len(cx_list)} after outlier rejection)")
        
        return (cx_avg, cy_avg, w_avg, h_avg)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence."""
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))
    
    def get_majority_class(self) -> Tuple[str, int]:
        """Get the most common class in this track."""
        if not self.detections:
            return ("unknown", -1)
        
        class_counts = {}
        for d in self.detections:
            key = (d.class_name, d.class_id)
            class_counts[key] = class_counts.get(key, 0) + 1
        
        majority_class = max(class_counts.items(), key=lambda x: x[1])
        return majority_class[0]
    
    @staticmethod
    def _reject_outliers_iqr(values: List[float], multiplier: float = 1.5) -> List[float]:
        """Reject outliers using IQR method."""
        if len(values) < 4:
            return values  # Need at least 4 values for quartiles
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        clean_values = [v for v in values if lower_bound <= v <= upper_bound]
        
        return clean_values if clean_values else values  # Return all if all rejected
    
    @staticmethod
    def _reject_outliers_std(values: List[float], threshold: float = 2.0) -> List[float]:
        """Reject outliers using standard deviation method."""
        if len(values) < 3:
            return values  # Need at least 3 values for meaningful std
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return values  # No variation
        
        clean_values = [v for v in values if abs(v - mean) <= threshold * std]
        
        return clean_values if clean_values else values  # Return all if all rejected


def calculate_iou(bbox1: Tuple[float, float, float, float],
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two bboxes in (cx, cy, w, h) format.
    
    Args:
        bbox1: (cx, cy, w, h)
        bbox2: (cx, cy, w, h)
    
    Returns:
        IoU value [0, 1]
    """
    # Convert to (x1, y1, x2, y2)
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    
    x1_1, y1_1 = cx1 - w1/2, cy1 - h1/2
    x2_1, y2_1 = cx1 + w1/2, cy1 + h1/2
    
    x1_2, y1_2 = cx2 - w2/2, cy2 - h2/2
    x2_2, y2_2 = cx2 + w2/2, cy2 + h2/2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class DetectionFilter:
    """Multi-frame detection filter with outlier rejection."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize detection filter.
        
        Args:
            config_path: Path to PnP calculation config (contains multi_frame_averaging settings)
        """
        if config_path is None:
            config_path = REPO_ROOT / "pickafresa_vision" / "configs" / "pnp_calc_config.yaml"
        
        logger.info("Initializing DetectionFilter...")
        logger.info(f"Config path: {config_path}")
        
        self.config = self._load_config(config_path)
        self.enabled = self.config.get("multi_frame_averaging", {}).get("enabled", False)
        self.num_frames = self.config.get("multi_frame_averaging", {}).get("num_frames", 10)
        self.outlier_method = self.config.get("multi_frame_averaging", {}).get("outlier_rejection", "iqr")
        
        # Tracking parameters
        self.iou_threshold = self.config.get("multi_frame_averaging", {}).get("tracking", {}).get("iou_threshold", 0.3)
        self.min_frames_visible = self.config.get("multi_frame_averaging", {}).get("tracking", {}).get("min_frames_visible", 5)
        
        # Overlap handling
        self.iou_merge_threshold = self.config.get("multi_frame_averaging", {}).get("overlap_handling", {}).get("iou_merge_threshold", 0.7)
        self.iou_separate_threshold = self.config.get("multi_frame_averaging", {}).get("overlap_handling", {}).get("iou_separate_threshold", 0.3)
        
        logger.info(f"Multi-frame averaging: {'enabled' if self.enabled else 'disabled'}")
        logger.info(f"Frames to average: {self.num_frames}")
        logger.info(f"Outlier rejection: {self.outlier_method}")
        logger.info("[OK] DetectionFilter initialized")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not HAVE_YAML:
            logger.warning("PyYAML not available, using default configuration")
            return {}
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config if config else {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def process_multi_frame(
        self,
        frames_and_detections: List[Tuple[Any, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple frames with detections and return averaged results.
        
        Args:
            frames_and_detections: List of (frame, detections) tuples where
                detections is a list of dicts with keys: bbox_cxcywh, confidence, class_name, class_id
        
        Returns:
            List of averaged detection dicts
        """
        if not self.enabled:
            # If multi-frame averaging disabled, return detections from first frame
            if frames_and_detections:
                return frames_and_detections[0][1]
            return []
        
        logger.info(f"Processing {len(frames_and_detections)} frames for multi-frame averaging")
        
        # Convert to Detection objects
        all_detections = []
        for frame_idx, (frame, detections) in enumerate(frames_and_detections):
            for det_dict in detections:
                detection = Detection(
                    bbox_cxcywh=tuple(det_dict["bbox_cxcywh"]),
                    confidence=det_dict["confidence"],
                    class_name=det_dict["class_name"],
                    class_id=det_dict["class_id"],
                    frame_idx=frame_idx
                )
                all_detections.append(detection)
        
        if not all_detections:
            logger.warning("No detections found across all frames")
            return []
        
        logger.info(f"Found {len(all_detections)} total detections across frames")
        
        # Track detections across frames
        tracks = self._track_detections(all_detections)
        
        # Filter tracks by minimum visibility
        valid_tracks = [t for t in tracks if len(t.detections) >= self.min_frames_visible]
        
        logger.info(f"Created {len(tracks)} tracks, {len(valid_tracks)} valid (>= {self.min_frames_visible} frames)")
        
        if not valid_tracks:
            logger.warning("No valid tracks found after filtering")
            return []
        
        # Handle overlapping tracks
        merged_tracks = self._handle_overlapping_tracks(valid_tracks)
        
        logger.info(f"After merging overlaps: {len(merged_tracks)} final tracks")
        
        # Convert tracks to averaged detections
        averaged_detections = []
        for track in merged_tracks:
            try:
                avg_bbox = track.get_average_bbox(
                    outlier_method=self.outlier_method,
                    iqr_multiplier=self.config.get("multi_frame_averaging", {}).get("iqr_multiplier", 1.5),
                    std_dev_threshold=self.config.get("multi_frame_averaging", {}).get("std_dev_threshold", 2.0)
                )
                avg_conf = track.get_average_confidence()
                class_name, class_id = track.get_majority_class()
                
                averaged_detections.append({
                    "bbox_cxcywh": list(avg_bbox),
                    "confidence": avg_conf,
                    "class_name": class_name,
                    "class_id": class_id,
                    "num_frames": len(track.detections)
                })
                
                logger.debug(f"Track {track.track_id}: Averaged {len(track.detections)} detections, "
                            f"conf={avg_conf:.3f}, class={class_name}")
            
            except Exception as e:
                logger.error(f"Failed to process track {track.track_id}: {e}")
                continue
        
        logger.info(f"Generated {len(averaged_detections)} averaged detections")
        return averaged_detections
    
    def _track_detections(self, detections: List[Detection]) -> List[DetectionTrack]:
        """
        Track detections across frames using IoU matching.
        
        Args:
            detections: List of all detections from all frames
        
        Returns:
            List of detection tracks
        """
        # Group detections by frame
        frames_dict = {}
        for det in detections:
            if det.frame_idx not in frames_dict:
                frames_dict[det.frame_idx] = []
            frames_dict[det.frame_idx].append(det)
        
        # Sort frames by index
        sorted_frame_indices = sorted(frames_dict.keys())
        
        # Initialize tracks with first frame
        tracks = []
        next_track_id = 0
        
        if sorted_frame_indices:
            for det in frames_dict[sorted_frame_indices[0]]:
                track = DetectionTrack(detections=[det], track_id=next_track_id)
                tracks.append(track)
                next_track_id += 1
        
        # Match detections in subsequent frames to tracks
        for frame_idx in sorted_frame_indices[1:]:
            frame_detections = frames_dict[frame_idx]
            
            # Calculate IoU between each detection and each track's latest detection
            unmatched_detections = set(range(len(frame_detections)))
            
            for track in tracks:
                if not track.detections:
                    continue
                
                latest_bbox = track.detections[-1].bbox_cxcywh
                best_match_idx = None
                best_iou = 0.0
                
                for det_idx in unmatched_detections:
                    det = frame_detections[det_idx]
                    iou = calculate_iou(latest_bbox, det.bbox_cxcywh)
                    
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_match_idx = det_idx
                
                if best_match_idx is not None:
                    track.add_detection(frame_detections[best_match_idx])
                    unmatched_detections.discard(best_match_idx)
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                track = DetectionTrack(detections=[frame_detections[det_idx]], track_id=next_track_id)
                tracks.append(track)
                next_track_id += 1
        
        return tracks
    
    def _handle_overlapping_tracks(self, tracks: List[DetectionTrack]) -> List[DetectionTrack]:
        """
        Handle overlapping tracks by merging or keeping separate based on IoU.
        
        Args:
            tracks: List of detection tracks
        
        Returns:
            List of tracks after handling overlaps
        """
        if len(tracks) <= 1:
            return tracks
        
        # Calculate average bbox for each track
        track_avg_bboxes = []
        for track in tracks:
            try:
                avg_bbox = track.get_average_bbox(outlier_method=self.outlier_method)
                track_avg_bboxes.append(avg_bbox)
            except Exception as e:
                logger.error(f"Failed to get average bbox for track {track.track_id}: {e}")
                track_avg_bboxes.append(None)
        
        # Build adjacency matrix of IoU values
        n = len(tracks)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            if track_avg_bboxes[i] is None:
                continue
            for j in range(i + 1, n):
                if track_avg_bboxes[j] is None:
                    continue
                iou = calculate_iou(track_avg_bboxes[i], track_avg_bboxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Merge tracks with high IoU
        merged = set()
        result_tracks = []
        
        for i in range(n):
            if i in merged or track_avg_bboxes[i] is None:
                continue
            
            # Find tracks to merge with this one
            to_merge = [i]
            for j in range(i + 1, n):
                if j not in merged and iou_matrix[i, j] >= self.iou_merge_threshold:
                    to_merge.append(j)
                    merged.add(j)
            
            if len(to_merge) == 1:
                # No merging needed
                result_tracks.append(tracks[i])
            else:
                # Merge multiple tracks
                merged_track = DetectionTrack(track_id=tracks[i].track_id)
                for idx in to_merge:
                    merged_track.detections.extend(tracks[idx].detections)
                    merged.add(idx)
                result_tracks.append(merged_track)
                logger.debug(f"Merged {len(to_merge)} tracks into track {merged_track.track_id}")
        
        return result_tracks
