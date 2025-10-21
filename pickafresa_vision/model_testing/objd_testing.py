"""
Real-time Object Detection Testing Tool for YOLOv11 Models

Interactive testing tool for evaluating object detection models on live video feeds.
Supports RealSense cameras (with optional depth overlay) and standard USB/built-in cameras.

Features:
- Model selection from available .pt files
- Camera source selection (RealSense or OpenCV VideoCapture)
- Real-time bounding box visualization with class-specific colors
- Depth information overlay for RealSense cameras
- Performance statistics (FPS, detection counts)
- Interactive controls for threshold adjustment and frame capture
- Configuration persistence across runs

Usage:
    python objd_testing.py

Keyboard Controls:
    q - Quit
    s - Save current frame
    p - Pause/Resume
    + - Increase confidence threshold by 0.05
    - - Decrease confidence threshold by 0.05
    r - Reset to default settings

Team YEA, 2025
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path for absolute package imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False
    print("Warning: pyrealsense2 not available. RealSense cameras will not be accessible.")

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    print("Warning: pyyaml not available. Will use fallback class names.")

from pickafresa_vision.vision_nodes.inference_bbox import load_model, infer
from pickafresa_vision.vision_tools.config_store import (
    load_config,
    save_config,
    get_namespace,
    update_namespace,
)

# Configuration namespace for this tool
CONFIG_NAMESPACE = "objd_testing"

# Default class colors (BGR format for OpenCV)
DEFAULT_CLASS_COLORS = {
    "ripe": (0, 255, 0),      # Green
    "unripe": (0, 255, 255),  # Yellow
    "flower": (255, 0, 0),    # Blue
}


@dataclass
class TestConfig:
    """Configuration for the testing session."""
    model_path: str
    dataset_path: str
    camera_type: str  # "realsense" or "opencv"
    camera_index: int
    enable_depth: bool
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 15


class FPSCounter:
    """Simple FPS counter with exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self.last_time = time.time()
    
    def update(self) -> float:
        current_time = time.time()
        delta = current_time - self.last_time
        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = self.alpha * instant_fps + (1 - self.alpha) * self.fps
        self.last_time = current_time
        return self.fps


class RealSenseCamera:
    """Wrapper for RealSense camera with depth support."""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        if not HAVE_REALSENSE:
            raise RuntimeError("pyrealsense2 is required for RealSense cameras")
        
        self.resolution = resolution
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.0
        self._filters: List = []
    
    def start(self):
        """Initialize and start the RealSense pipeline."""
        width, height = self.resolution
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)
        
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        # Setup depth filtering pipeline
        try:
            self._filters = [
                rs.disparity_transform(True),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.disparity_transform(False),
                rs.hole_filling_filter(),
            ]
            # Tune filters
            spatial: rs.spatial_filter = self._filters[1]
            spatial.set_option(rs.option.holes_fill, 3)
            temporal: rs.temporal_filter = self._filters[2]
            temporal.set_option(rs.option.alpha, 0.5)
        except Exception:
            self._filters = []
    
    def read(self) -> Tuple[bool, np.ndarray, Optional[rs.depth_frame]]:
        """Read aligned color and depth frames."""
        if not self.pipeline or not self.align:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return False, np.zeros((480, 640, 3), dtype=np.uint8), None
            
            # Apply depth filtering
            try:
                f = depth_frame
                for flt in self._filters:
                    f = flt.process(f)
                depth_frame = f.as_depth_frame()
            except Exception:
                pass
            
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image, depth_frame
        except Exception as e:
            print(f"Error reading RealSense frames: {e}")
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None
    
    def stop(self):
        """Stop the RealSense pipeline."""
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None


class OpenCVCamera:
    """Wrapper for standard OpenCV camera (USB/built-in)."""
    
    def __init__(self, camera_index: int, resolution: Tuple[int, int] = (640, 480)):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
    
    def start(self):
        """Initialize and start the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        width, height = self.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self) -> Tuple[bool, np.ndarray, Optional[rs.depth_frame]]:
        """Read a frame (no depth for OpenCV cameras)."""
        if not self.cap:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else np.zeros((480, 640, 3), dtype=np.uint8), None
    
    def stop(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
        self.cap = None


def load_class_names_from_yaml(dataset_path: Path) -> Dict[str, str]:
    """Load class names from dataset's data.yaml file."""
    if not HAVE_YAML:
        return {}
    
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"Warning: data.yaml not found at {yaml_path}")
        return {}
    
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        names = data.get("names", [])
        if isinstance(names, list):
            return {name: name for name in names}
        elif isinstance(names, dict):
            return {str(v): str(v) for v in names.values()}
        return {}
    except Exception as e:
        print(f"Error loading class names from YAML: {e}")
        return {}


def get_class_color(class_name: str, class_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Get BGR color for a class name."""
    return class_colors.get(class_name, (128, 128, 128))  # Default gray


def calculate_bbox_median_depth(
    depth_frame: rs.depth_frame,
    bbox: Tuple[float, float, float, float],
    depth_scale: float
) -> Optional[float]:
    """Calculate median depth value within a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Ensure bbox is within frame bounds
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract depth values in bbox region
    depth_values = []
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            depth = depth_frame.get_distance(x, y)
            if depth > 0:  # Valid depth
                depth_values.append(depth)
    
    if not depth_values:
        return None
    
    # Return median depth in meters
    return float(np.median(depth_values))


def draw_overlay_stats(
    frame: np.ndarray,
    fps: float,
    total_detections: int,
    class_counts: Dict[str, int],
    model_name: str,
    camera_info: str,
    conf_threshold: float,
    resolution: Tuple[int, int],
    paused: bool = False
) -> np.ndarray:
    """Draw statistics overlay on the frame."""
    overlay = frame.copy()
    
    # Semi-transparent background for text
    cv2.rectangle(overlay, (5, 5), (350, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    y_offset = 25
    line_height = 25
    
    # Draw stats
    stats = [
        f"Model: {model_name}",
        f"Camera: {camera_info}",
        f"Resolution: {resolution[0]}x{resolution[1]}",
        f"FPS: {fps:.1f}",
        f"Conf Threshold: {conf_threshold:.2f}",
        f"Total Detections: {total_detections}",
        "",
        "Per-class counts:",
    ]
    
    for class_name, count in sorted(class_counts.items()):
        stats.append(f"  {class_name}: {count}")
    
    if paused:
        stats.insert(0, "*** PAUSED ***")
    
    for i, line in enumerate(stats):
        y = y_offset + i * line_height
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Draw controls at bottom
    controls_y = frame.shape[0] - 60
    cv2.rectangle(frame, (5, controls_y - 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
    
    control_text = [
        "Controls: [Q]uit | [S]ave | [P]ause | [+/-] Conf | [R]eset",
    ]
    
    for i, line in enumerate(control_text):
        y = controls_y + 10 + i * 20
        cv2.putText(frame, line, (10, y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List,
    bboxes: List[Tuple[float, float, float, float]],
    class_colors: Dict[str, Tuple[int, int, int]],
    depth_frame: Optional[rs.depth_frame] = None,
    depth_scale: float = 0.0
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det, bbox in zip(detections, bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        class_name = det.clazz
        confidence = det.confidence
        
        # Get class-specific color
        color = get_class_color(class_name, class_colors)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name} {confidence:.2f}"
        
        # Add depth if available
        if depth_frame is not None:
            depth = calculate_bbox_median_depth(depth_frame, bbox, depth_scale)
            if depth is not None:
                label += f" | {depth*1000:.0f}mm"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w + 5, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return frame


def create_depth_colormap(depth_frame: rs.depth_frame) -> np.ndarray:
    """Create a colorized depth map for visualization."""
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    return depth_colormap


def prompt_model_selection(models_dir: Path) -> Optional[Path]:
    """Prompt user to select a model from available .pt files."""
    pt_files = sorted(models_dir.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt model files found in {models_dir}")
        return None
    
    print("\n=== Available Models ===")
    for i, model_file in enumerate(pt_files, 1):
        print(f"{i}. {model_file.name}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(pt_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(pt_files):
                return pt_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(pt_files)}")
        except (ValueError, KeyboardInterrupt):
            return None


def prompt_dataset_selection(datasets_dir: Path) -> Optional[Path]:
    """Prompt user to select a dataset folder for class names."""
    dataset_dirs = sorted([d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if not dataset_dirs:
        print(f"No dataset directories found in {datasets_dir}")
        return None
    
    print("\n=== Available Datasets ===")
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        print(f"{i}. {dataset_dir.name}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(dataset_dirs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(dataset_dirs):
                return dataset_dirs[idx]
            else:
                print(f"Please enter a number between 1 and {len(dataset_dirs)}")
        except (ValueError, KeyboardInterrupt):
            return None


def prompt_camera_selection() -> Tuple[Optional[str], Optional[int]]:
    """Prompt user to select camera type and index."""
    print("\n=== Camera Selection ===")
    print("1. RealSense Camera" + (" [Available]" if HAVE_REALSENSE else " [NOT AVAILABLE]"))
    print("2. OpenCV Camera (USB/Built-in)")
    
    while True:
        try:
            choice = input("\nSelect camera type (1-2) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None, None
            
            if choice == '1':
                if not HAVE_REALSENSE:
                    print("RealSense not available. Please install pyrealsense2.")
                    continue
                return "realsense", 0
            elif choice == '2':
                cam_idx = input("Enter camera index (default 0): ").strip()
                cam_idx = int(cam_idx) if cam_idx else 0
                return "opencv", cam_idx
            else:
                print("Please enter 1 or 2")
        except (ValueError, KeyboardInterrupt):
            return None, None


def prompt_depth_mode(camera_type: str) -> bool:
    """Prompt user to enable depth overlay (only for RealSense)."""
    if camera_type != "realsense":
        return False
    
    print("\n=== Depth Mode ===")
    choice = input("Enable depth overlay? (y/n, default=y): ").strip().lower()
    return choice != 'n'


def load_previous_config() -> Optional[TestConfig]:
    """Load previous configuration if available."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    if not ns:
        return None
    
    try:
        return TestConfig(
            model_path=ns.get("model_path", ""),
            dataset_path=ns.get("dataset_path", ""),
            camera_type=ns.get("camera_type", "opencv"),
            camera_index=ns.get("camera_index", 0),
            enable_depth=ns.get("enable_depth", False),
            conf_threshold=ns.get("conf_threshold", 0.25),
            iou_threshold=ns.get("iou_threshold", 0.45),
            resolution=tuple(ns.get("resolution", [640, 480])),
            fps=ns.get("fps", 30),
        )
    except Exception:
        return None


def save_current_config(test_config: TestConfig):
    """Save current configuration for future runs."""
    cfg = load_config()
    update_namespace(cfg, CONFIG_NAMESPACE, {
        "model_path": test_config.model_path,
        "dataset_path": test_config.dataset_path,
        "camera_type": test_config.camera_type,
        "camera_index": test_config.camera_index,
        "enable_depth": test_config.enable_depth,
        "conf_threshold": test_config.conf_threshold,
        "iou_threshold": test_config.iou_threshold,
        "resolution": list(test_config.resolution),
        "fps": test_config.fps,
    })


def interactive_setup() -> Optional[TestConfig]:
    """Interactive configuration setup with option to use previous settings."""
    models_dir = REPO_ROOT / "pickafresa_vision" / "models"
    datasets_dir = REPO_ROOT / "pickafresa_vision" / "datasets"
    
    # Check for previous config
    prev_config = load_previous_config()
    if prev_config and Path(prev_config.model_path).exists():
        print("\n=== Previous Configuration Found ===")
        print(f"Model: {Path(prev_config.model_path).name}")
        print(f"Dataset: {Path(prev_config.dataset_path).name}")
        print(f"Camera: {prev_config.camera_type} (index {prev_config.camera_index})")
        print(f"Depth: {'Enabled' if prev_config.enable_depth else 'Disabled'}")
        print(f"Confidence: {prev_config.conf_threshold}")
        
        use_prev = input("\nUse previous configuration? (y/n, default=y): ").strip().lower()
        if use_prev != 'n':
            return prev_config
    
    # New configuration
    print("\n=== New Configuration ===")
    
    # Model selection
    model_path = prompt_model_selection(models_dir)
    if model_path is None:
        return None
    
    # Dataset selection
    dataset_path = prompt_dataset_selection(datasets_dir)
    if dataset_path is None:
        return None
    
    # Camera selection
    camera_type, camera_index = prompt_camera_selection()
    if camera_type is None:
        return None
    
    # Depth mode
    enable_depth = prompt_depth_mode(camera_type)
    
    return TestConfig(
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        camera_type=camera_type,
        camera_index=camera_index,
        enable_depth=enable_depth,
    )


def main():
    """Main testing loop."""
    print("=" * 60)
    print("Object Detection Model Testing Tool")
    print("Team YEA, 2025")
    print("=" * 60)
    
    # Setup configuration
    config = interactive_setup()
    if config is None:
        print("\nSetup cancelled.")
        return
    
    # Save configuration
    save_current_config(config)
    
    # Load model
    print(f"\nLoading model: {Path(config.model_path).name}")
    try:
        model = load_model(config.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load class names and colors
    class_names = load_class_names_from_yaml(Path(config.dataset_path))
    class_colors = DEFAULT_CLASS_COLORS.copy()
    
    # Initialize camera
    print(f"\nInitializing camera: {config.camera_type}")
    try:
        if config.camera_type == "realsense":
            camera = RealSenseCamera(resolution=config.resolution, fps=config.fps)
        else:
            camera = OpenCVCamera(camera_index=config.camera_index, resolution=config.resolution)
        
        camera.start()
        print("Camera initialized successfully!")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # State variables
    paused = False
    frame_count = 0
    
    # Main loop
    print("\n" + "=" * 60)
    print("Starting real-time inference...")
    print("Press 'q' to quit, 's' to save frame, 'p' to pause")
    print("=" * 60 + "\n")
    
    window_name = "YOLOv11 Object Detection Testing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame, depth_frame = camera.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            if not paused:
                # Run inference
                detections, bboxes = infer(
                    model,
                    frame,
                    conf=config.conf_threshold,
                    iou=config.iou_threshold,
                    bbox_format="xyxy",
                    normalized=False,
                )
                
                # Count detections per class
                class_counts = {}
                for det in detections:
                    class_counts[det.clazz] = class_counts.get(det.clazz, 0) + 1
                
                # Update FPS
                fps = fps_counter.update()
                
                # Draw detections
                vis_frame = frame.copy()
                if config.enable_depth and depth_frame is not None:
                    depth_scale = camera.depth_scale if hasattr(camera, 'depth_scale') else 0.001
                    vis_frame = draw_detections(vis_frame, detections, bboxes, class_colors, depth_frame, depth_scale)
                else:
                    vis_frame = draw_detections(vis_frame, detections, bboxes, class_colors)
                
                # Draw stats overlay
                camera_info = f"{config.camera_type}:{config.camera_index}"
                if config.enable_depth:
                    camera_info += " (depth)"
                
                vis_frame = draw_overlay_stats(
                    vis_frame,
                    fps,
                    len(detections),
                    class_counts,
                    Path(config.model_path).name,
                    camera_info,
                    config.conf_threshold,
                    config.resolution,
                    paused
                )
                
                # Show depth colormap alongside if enabled
                if config.enable_depth and depth_frame is not None:
                    depth_colormap = create_depth_colormap(depth_frame)
                    # Resize to match frame height
                    h, w = vis_frame.shape[:2]
                    depth_colormap = cv2.resize(depth_colormap, (w // 2, h))
                    vis_frame = cv2.resize(vis_frame, (w // 2, h))
                    vis_frame = np.hstack((vis_frame, depth_colormap))
                
                frame_count += 1
            else:
                # Paused - just show the last frame with pause indicator
                vis_frame = draw_overlay_stats(
                    vis_frame,
                    fps,
                    len(detections),
                    class_counts,
                    Path(config.model_path).name,
                    camera_info,
                    config.conf_threshold,
                    config.resolution,
                    paused=True
                )
            
            # Display frame
            cv2.imshow(window_name, vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"objd_test_{timestamp}.jpg"
                save_path = REPO_ROOT / "pickafresa_vision" / "images" / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), vis_frame)
                print(f"Frame saved: {filename}")
            elif key == ord('p'):
                # Toggle pause
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                config.conf_threshold = min(0.95, config.conf_threshold + 0.05)
                print(f"Confidence threshold: {config.conf_threshold:.2f}")
                save_current_config(config)
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                config.conf_threshold = max(0.05, config.conf_threshold - 0.05)
                print(f"Confidence threshold: {config.conf_threshold:.2f}")
                save_current_config(config)
            elif key == ord('r'):
                # Reset to defaults
                config.conf_threshold = 0.25
                config.iou_threshold = 0.45
                print("Reset to default thresholds")
                save_current_config(config)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Done!")


if __name__ == "__main__":
    main()
