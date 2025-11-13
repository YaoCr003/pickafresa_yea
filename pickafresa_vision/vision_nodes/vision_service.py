"""
Always-On Vision Service with Persistent RealSense Pipeline

Provides a persistent vision system that keeps the RealSense camera pipeline open,
eliminating the risk of pipeline startup failures and improving response time.

Key Features:
- Persistent RealSense pipeline (no repeated open/close)
- Socket-based IPC for capture requests from robot system
- Live preview window with detection overlays, depth visualization, and 3D axes
- Multi-frame averaging support
- Automatic reconnection on pipeline failure
- Frame buffering for reliability

Architecture:
    Robot System <--Socket IPC--> Vision Service <--> RealSense Camera
                                         |
                                    Preview Window (optional)

Usage:
    # Start service (standalone)
    python -m pickafresa_vision.vision_nodes.vision_service --preview
    
    # Or import and use
    from pickafresa_vision.vision_nodes.vision_service import VisionService
    
    service = VisionService(config_path="configs/vision_service_config.yaml")
    service.start()

# @aldrick-t, 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports - Vision
from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture
from pickafresa_vision.vision_nodes.pnp_calc import FruitPoseEstimator
from pickafresa_vision.vision_nodes.inference_filter import DetectionFilter

# Local imports - Robot (for ROS2 logger)
try:
    from pickafresa_robot.robot_testing.ros2_logger import create_logger
    HAVE_ROS2_LOGGER = True
except ImportError:
    HAVE_ROS2_LOGGER = False
    create_logger = None  # type: ignore

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    yaml = None  # type: ignore

try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False
    YOLO = None  # type: ignore

# Setup logging
LOG_DIR = REPO_ROOT / "pickafresa_vision" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vision_service.log"

# Use ROS2-style logger if available, otherwise fallback to basic logging
if HAVE_ROS2_LOGGER and create_logger is not None:
    logger = create_logger(
        node_name="vision_service",
        log_dir=LOG_DIR,
        log_prefix="vision_service",
        console_level="INFO",
        file_level="DEBUG",
        use_timestamp=False,  # Use fixed filename
        overwrite_log=True    # Overwrite on each start
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [vision_service]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),  # Overwrite on start
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)


class VisionServiceError(Exception):
    """Base exception for vision service errors."""
    pass


class VisionService:
    """Always-on vision service with persistent RealSense pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None, enable_preview: bool = False):
        """
        Initialize vision service.
        
        Args:
            config_path: Path to vision service config file
            enable_preview: Override config to enable/disable preview
        """
        if config_path is None:
            config_path = REPO_ROOT / "pickafresa_vision" / "configs" / "vision_service_config.yaml"
        
        logger.info("="*60)
        logger.info("Initializing PickaFresa Vision Service")
        logger.info("="*60)
        logger.info(f"Config: {config_path}")
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Override preview setting if specified
        if enable_preview:
            self.config.setdefault("preview", {})["enabled"] = True
        
        # Service state
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        
        # Components (initialized in start())
        self.camera: Optional[RealSenseCapture] = None
        self.model: Optional[Any] = None
        self.pnp_estimator: Optional[FruitPoseEstimator] = None
        self.detection_filter: Optional[DetectionFilter] = None
        
        # IPC
        self.server_socket: Optional[socket.socket] = None
        self.server_thread: Optional[threading.Thread] = None
        
        # Preview
        self.preview_enabled = self.config.get("preview", {}).get("enabled", False)
        self.preview_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.perf_stats = {
            "capture_time": [],
            "inference_time": [],
            "pnp_time": [],
            "total_time": []
        }
        
        logger.info("[OK] VisionService object created")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not HAVE_YAML:
            logger.warning("PyYAML not available, using default configuration")
            return self._get_default_config()
        
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[OK] Loaded config from {self.config_path}")
            return config if config else self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "service": {
                "port": 5555,
                "host": "localhost",
                "timeout_sec": 30.0
            },
            "preview": {
                "enabled": False,
                "fps": 30
            },
            "inference": {
                "model_path": "pickafresa_vision/models/best.pt",
                "confidence_threshold": 0.25,
                "device": "cpu"
            },
            "pnp": {
                "config_path": "pickafresa_vision/configs/pnp_calc_config.yaml"
            }
        }
    
    def start(self) -> bool:
        """
        Start the vision service.
        
        Returns:
            True if started successfully
        """
        try:
            logger.info("Starting vision service...")
            
            # Initialize camera
            logger.info("Initializing RealSense camera...")
            self.camera = RealSenseCapture()
            self.camera.start()
            logger.info("[OK] Camera initialized")
            
            # Load inference model
            if HAVE_YOLO:
                model_path = self.config.get("inference", {}).get("model_path", "")
                full_model_path = REPO_ROOT / model_path if model_path else None
                
                if full_model_path and full_model_path.exists():
                    logger.info(f"Loading YOLO model from {full_model_path}...")
                    self.model = YOLO(str(full_model_path))
                    logger.info("[OK] Model loaded")
                else:
                    logger.warning(f"Model not found: {full_model_path}")
                    logger.warning("Service will run without inference (capture only)")
            else:
                logger.warning("ultralytics not available, running without inference")
            
            # Initialize PnP estimator
            pnp_config_path = self.config.get("pnp", {}).get("config_path", "")
            full_pnp_config_path = REPO_ROOT / pnp_config_path if pnp_config_path else None
            
            if full_pnp_config_path and full_pnp_config_path.exists():
                logger.info("Initializing PnP estimator...")
                self.pnp_estimator = FruitPoseEstimator(config_path=full_pnp_config_path)
                logger.info("[OK] PnP estimator initialized")
            else:
                logger.warning("PnP config not found, running without PnP")
            
            # Initialize detection filter
            if full_pnp_config_path:
                logger.info("Initializing detection filter...")
                self.detection_filter = DetectionFilter(config_path=full_pnp_config_path)
                logger.info("[OK] Detection filter initialized")
            
            # Set running flag BEFORE starting threads
            self.running = True
            self.start_time = time.time()
            
            # Start IPC server
            self._start_ipc_server()
            
            # Start preview if enabled
            if self.preview_enabled:
                self._start_preview()
            
            logger.info("[OK] Vision Service Started Successfully")
            logger.info(f"  - Camera: Active")
            logger.info(f"  - Model: {'Loaded' if self.model else 'Not available'}")
            logger.info(f"  - PnP: {'Enabled' if self.pnp_estimator else 'Disabled'}")
            logger.info(f"  - IPC Port: {self.config.get('service', {}).get('port', 5555)}")
            logger.info(f"  - Preview: {'Enabled' if self.preview_enabled else 'Disabled'}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start vision service: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the vision service."""
        logger.info("Stopping vision service...")
        
        self.running = False
        
        # Stop camera
        if self.camera:
            try:
                self.camera.stop()
                logger.info("[OK] Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
        
        # Stop IPC server
        if self.server_socket:
            try:
                self.server_socket.close()
                logger.info("[OK] IPC server stopped")
            except Exception as e:
                logger.error(f"Error stopping IPC server: {e}")
        
        # Stop preview
        if self.preview_thread and self.preview_thread.is_alive():
            try:
                self.preview_thread.join(timeout=2.0)
                logger.info("[OK] Preview stopped")
            except Exception as e:
                logger.error(f"Error stopping preview: {e}")
        
        logger.info("[OK] Vision service stopped")
    
    def _start_ipc_server(self):
        """Start IPC server for robot communication."""
        port = self.config.get("service", {}).get("port", 5555)
        host = self.config.get("service", {}).get("host", "localhost")
        
        logger.info(f"Starting IPC server on {host}:{port}...")
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((host, port))
            self.server_socket.listen(5)
            
            self.server_thread = threading.Thread(target=self._ipc_server_loop, daemon=True)
            self.server_thread.start()
            
            logger.info(f"[OK] IPC server listening on {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}")
            raise VisionServiceError(f"IPC server startup failed: {e}")
    
    def _ipc_server_loop(self):
        """IPC server loop to handle client requests."""
        logger.info("IPC server loop started")
        
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    logger.debug(f"Client connected: {client_addr}")
                    
                    # Handle request in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client_request,
                        args=(client_socket,),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
            except Exception as e:
                if self.running:
                    logger.error(f"IPC server error: {e}")
        
        logger.info("IPC server loop stopped")
    
    def _handle_client_request(self, client_socket: socket.socket):
        """
        Handle client requests on a persistent connection.
        
        Supports multiple requests on the same socket (keepalive).
        Connection is kept open until client disconnects or error occurs.
        
        Expected request format (JSON, newline-terminated):
        {
            "command": "capture",
            "multi_frame": false,
            "num_frames": 1
        }
        
        Response format (JSON, newline-terminated):
        {
            "success": true,
            "detections": [...],
            "timestamp": "...",
            "frame_count": 123
        }
        """
        try:
            # Keep connection alive for multiple requests
            while self.running:
                # Set timeout to detect disconnections
                client_socket.settimeout(60.0)
                
                try:
                    # Receive request (read until newline)
                    request_data = b''
                    while b'\n' not in request_data:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            # Client disconnected
                            logger.debug("Client disconnected")
                            return
                        request_data += chunk
                    
                    # Parse request
                    request_json = request_data.decode('utf-8').strip()
                    request = json.loads(request_json)
                    
                    logger.debug(f"Received request: {request.get('command', 'unknown')}")
                    
                    if request.get("command") == "capture":
                        # Execute capture
                        result = self.capture_and_process(
                            multi_frame=request.get("multi_frame", False),
                            num_frames=request.get("num_frames", 1)
                        )
                        
                        # Send response
                        response = json.dumps(result)
                        client_socket.sendall(response.encode('utf-8') + b'\n')
                        
                        logger.debug(f"Sent response: {result.get('success', False)}")
                    
                    elif request.get("command") == "status":
                        # Return service status
                        status = {
                            "success": True,
                            "alive": True,  # Service is alive if responding
                            "ready": self.running,  # Service is ready if running
                            "running": self.running,
                            "frame_count": self.frame_count,
                            "uptime_sec": time.time() - self.start_time if self.start_time else 0
                        }
                        response = json.dumps(status)
                        client_socket.sendall(response.encode('utf-8') + b'\n')
                    
                    else:
                        # Unknown command
                        error_response = {
                            "success": False,
                            "error": f"Unknown command: {request.get('command')}"
                        }
                        response = json.dumps(error_response)
                        client_socket.sendall(response.encode('utf-8') + b'\n')
                
                except socket.timeout:
                    # Client idle for too long, close connection
                    logger.debug("Client connection timeout")
                    break
                except json.JSONDecodeError as e:
                    # Invalid JSON, send error and continue
                    logger.error(f"Invalid JSON from client: {e}")
                    error_response = {
                        "success": False,
                        "error": f"Invalid JSON: {str(e)}"
                    }
                    try:
                        response = json.dumps(error_response)
                        client_socket.sendall(response.encode('utf-8') + b'\n')
                    except:
                        break
        
        except Exception as e:
            logger.error(f"Error handling client connection: {e}")
            try:
                error_response = {
                    "success": False,
                    "error": str(e)
                }
                response = json.dumps(error_response)
                client_socket.sendall(response.encode('utf-8') + b'\n')
            except:
                pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def capture_and_process(
        self,
        multi_frame: bool = False,
        num_frames: int = 1
    ) -> Dict[str, Any]:
        """
        Capture frame(s), run inference, and compute PnP.
        
        Args:
            multi_frame: Use multi-frame averaging
            num_frames: Number of frames to capture (if multi_frame=True)
        
        Returns:
            Dictionary with results
        """
        try:
            start_time = time.time()
            
            # Capture frame(s)
            if multi_frame and num_frames > 1:
                frames_and_detections = []
                for i in range(num_frames):
                    t0 = time.time()
                    color_frame, depth_frame = self.camera.capture_frame()
                    capture_time = time.time() - t0
                    
                    # Run inference
                    t0 = time.time()
                    detections = self._run_inference(color_frame)
                    inference_time = time.time() - t0
                    
                    frames_and_detections.append(((color_frame, depth_frame), detections))
                    
                    logger.debug(f"Frame {i+1}/{num_frames}: {len(detections)} detections")
                
                # Average detections
                if self.detection_filter:
                    detections = self.detection_filter.process_multi_frame(frames_and_detections)
                else:
                    # No filtering, use first frame
                    detections = frames_and_detections[0][1] if frames_and_detections else []
                
                # Use last frame for PnP
                color_frame, depth_frame = frames_and_detections[-1][0] if frames_and_detections else (None, None)
            
            else:
                # Single frame capture
                t0 = time.time()
                color_frame, depth_frame = self.camera.capture_frame()
                capture_time = time.time() - t0
                
                # Run inference
                t0 = time.time()
                detections = self._run_inference(color_frame)
                inference_time = time.time() - t0
            
            # Compute PnP for detections
            t0 = time.time()
            pnp_results = self._compute_pnp(color_frame, depth_frame, detections)
            pnp_time = time.time() - t0
            
            total_time = time.time() - start_time
            
            self.frame_count += 1
            
            # Track performance
            self.perf_stats["capture_time"].append(capture_time)
            self.perf_stats["inference_time"].append(inference_time)
            self.perf_stats["pnp_time"].append(pnp_time)
            self.perf_stats["total_time"].append(total_time)
            
            # Log performance every N frames
            log_every = self.config.get("logging", {}).get("performance", {}).get("log_every_n_frames", 30)
            if self.frame_count % log_every == 0:
                self._log_performance()
            
            return {
                "success": True,
                "detections": pnp_results,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "frame_count": self.frame_count,
                "processing_time_ms": total_time * 1000
            }
        
        except Exception as e:
            logger.error(f"Error in capture_and_process: {e}")
            return {
                "success": False,
                "error": str(e),
                "frame_count": self.frame_count
            }
    
    def _run_inference(self, color_frame) -> List[Dict[str, Any]]:
        """Run inference on color frame."""
        if not self.model:
            return []
        
        try:
            # RealSense returns RGB, but YOLO expects BGR (OpenCV format)
            # Convert RGB to BGR for inference
            if len(color_frame.shape) == 3 and color_frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            else:
                logger.warning(f"Unexpected color frame shape: {color_frame.shape}")
                bgr_frame = color_frame
            
            conf_threshold = self.config.get("inference", {}).get("confidence_threshold", 0.25)
            iou_threshold = self.config.get("inference", {}).get("iou_threshold", 0.45)
            
            # Run inference on BGR frame
            results = self.model.predict(
                bgr_frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Extract detections
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bbox in xywh format (YOLO format)
                        xywh = boxes.xywh[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                        
                        # Filter by target classes
                        target_classes = self.config.get("inference", {}).get("filter", {}).get("target_classes", ["ripe"])
                        if class_name not in target_classes:
                            continue
                        
                        detections.append({
                            "bbox_cxcywh": [float(x) for x in xywh],
                            "confidence": conf,
                            "class_name": class_name,
                            "class_id": cls_id
                        })
            
            return detections
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []
    
    def _compute_pnp(
        self,
        color_frame,
        depth_frame,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute PnP for detections."""
        if not self.pnp_estimator or not detections:
            return detections
        
        try:
            # Get camera intrinsics
            intrinsics = self.camera.get_intrinsics()
            
            # Sanity check: Validate intrinsics are reasonable for RealSense D435
            # fx/fy should typically be 400-2000 pixels for this camera model
            if intrinsics.fx > 2000 or intrinsics.fy > 2000 or intrinsics.fx < 400 or intrinsics.fy < 400:
                logger.warning(f"Intrinsics seem invalid (fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f})")
                logger.warning("Likely corrupted calibration file. Forcing RealSense SDK intrinsics...")
                intrinsics = self.camera.get_intrinsics(source="realsense")
                logger.info(f"Using SDK intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
            
            camera_matrix = intrinsics.to_matrix()
            dist_coeffs = intrinsics.distortion_coeffs
            
            # Prepare detection data for PnP estimator
            bboxes = [d["bbox_cxcywh"] for d in detections]
            
            # Convert dictionaries to simple objects for PnP estimator
            # (PnP estimator expects objects with .confidence, .clazz, .class_id attributes)
            class SimpleDetection:
                def __init__(self, d):
                    self.confidence = d["confidence"]
                    self.clazz = d["class_name"]
                    self.class_id = d["class_id"]
            
            detection_objects = [SimpleDetection(d) for d in detections]
            
            # Estimate poses
            pnp_results = self.pnp_estimator.estimate_poses(
                color_image=color_frame,
                depth_frame=depth_frame,
                detections=detection_objects,
                bboxes_cxcywh=bboxes,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            
            # Convert PoseEstimationResult objects to dicts
            results_dicts = []
            for result in pnp_results:
                result_dict = {
                    "bbox_cxcywh": list(result.bbox_cxcywh),
                    "confidence": result.confidence,
                    "class_name": result.class_name,
                    "class_id": result.class_id,
                    "success": result.success,
                    "T_cam_fruit": result.T_cam_fruit.tolist() if result.T_cam_fruit is not None else None,
                    "position_cam": result.position_cam.tolist() if result.position_cam is not None else None,
                    "rotation_matrix": result.rotation_matrix.tolist() if result.rotation_matrix is not None else None,
                    "rvec": result.rvec.tolist() if result.rvec is not None else None,
                    "tvec": result.tvec.tolist() if result.tvec is not None else None,
                    "error_reason": result.error_reason,
                    "depth_samples": result.depth_samples,
                    "median_depth": result.median_depth,
                    "depth_variance": result.depth_variance,
                    "sampling_strategy": result.sampling_strategy
                }
                results_dicts.append(result_dict)
            
            return results_dicts
        
        except Exception as e:
            logger.error(f"PnP computation error: {e}")
            return detections
    
    def _log_performance(self):
        """Log performance statistics."""
        if not any(self.perf_stats.values()):
            return
        
        logger.info("Performance Statistics:")
        for metric, values in self.perf_stats.items():
            if values:
                avg = np.mean(values) * 1000  # Convert to ms
                std = np.std(values) * 1000
                logger.info(f"  {metric}: {avg:.1f} ± {std:.1f} ms")
        
        # Calculate FPS
        if self.perf_stats["total_time"]:
            fps = 1.0 / np.mean(self.perf_stats["total_time"])
            logger.info(f"  FPS: {fps:.1f}")
        
        # Clear old stats (keep only recent)
        max_samples = 100
        for key in self.perf_stats:
            if len(self.perf_stats[key]) > max_samples:
                self.perf_stats[key] = self.perf_stats[key][-max_samples:]
    
    def _start_preview(self):
        """
        Start preview window.
        On macOS, this must be called from the main thread.
        On other platforms, it will run in a separate thread.
        """
        import platform
        
        if platform.system() == "Darwin":  # macOS
            logger.info("Preview will run on main thread (macOS requirement)")
            # Preview will be started by run_preview_loop() on main thread
        else:
            logger.info("Starting preview window in separate thread...")
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
            logger.info("[OK] Preview window started")
    
    def _preview_loop(self):
        """Preview window loop."""
        logger.info("Preview loop started")
        
        window_name = self.config.get("preview", {}).get("window_name", "Vision Service Preview")
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            logger.error(f"Failed to create window (GUI must run on main thread on macOS): {e}")
            logger.error("Use run_preview_loop() on main thread instead of _start_preview()")
            return
        
        target_fps = self.config.get("preview", {}).get("fps", 30)
        frame_delay_ms = int(1000 / target_fps)
        
        while self.running:
            try:
                # Capture frame
                color_frame, depth_frame = self.camera.capture_frame()
                
                # Run inference for preview
                detections = self._run_inference(color_frame)
                
                # Draw detections
                preview_frame = self._draw_preview(color_frame, depth_frame, detections)
                
                # Show frame
                cv2.imshow(window_name, preview_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord('q'):
                    logger.info("Preview: Quit requested")
                    self.stop()
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"Preview: {'Paused' if self.paused else 'Resumed'}")
            
            except Exception as e:
                logger.error(f"Preview error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        logger.info("Preview loop stopped")
    
    def run_preview_loop(self):
        """
        Run preview loop on the calling thread (must be main thread on macOS).
        
        This method blocks until the service is stopped or 'q' is pressed.
        Use this instead of start() when preview is enabled on macOS.
        """
        if not self.preview_enabled:
            logger.warning("Preview not enabled, nothing to run")
            return
        
        if not self.running:
            logger.error("Service not started, call start() first")
            return
        
        logger.info("Running preview loop on main thread...")
        self._preview_loop()
    
    def _draw_preview(
        self,
        color_frame,
        depth_frame,
        detections: List[Dict[str, Any]]
    ):
        """Draw preview frame with overlays."""
        # Clone frame for drawing
        preview = color_frame.copy()
        
        # Convert RGB to BGR for OpenCV display
        # RealSense returns RGB, but cv2.imshow expects BGR
        preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        
        # Draw detections
        for det in detections:
            cx, cy, w, h = det["bbox_cxcywh"]
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
            # Draw bbox (BGR format: Green = (0, 255, 0))
            color = (0, 255, 0)  # Green in BGR
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background for better visibility
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background (semi-transparent black)
            cv2.rectangle(preview, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 0, 0), -1)
            # Draw label text (white)
            cv2.putText(preview, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw info panel background (top-left corner)
        info_bg_height = 90
        cv2.rectangle(preview, (0, 0), (250, info_bg_height), (0, 0, 0), -1)
        
        # Draw FPS counter
        if self.perf_stats["total_time"]:
            fps = 1.0 / np.mean(self.perf_stats["total_time"][-30:]) if len(self.perf_stats["total_time"]) >= 30 else 0
            cv2.putText(preview, f"FPS: {fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw frame count
        cv2.putText(preview, f"Frame: {self.frame_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detection count
        cv2.putText(preview, f"Detections: {len(detections)}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return preview


def main():
    """Main entry point for standalone execution."""
    import platform
    
    parser = argparse.ArgumentParser(description="PickaFresa Always-On Vision Service")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--preview", action="store_true", help="Enable preview window")
    parser.add_argument("--port", type=int, help="IPC server port (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config) if args.config else None
    
    # Create and start service
    service = VisionService(config_path=config_path, enable_preview=args.preview)
    
    # Override port if specified
    if args.port:
        service.config.setdefault("service", {})["port"] = args.port
    
    if not service.start():
        logger.error("Failed to start service")
        return 1
    
    try:
        # On macOS with preview enabled, run preview on main thread
        if args.preview and platform.system() == "Darwin":
            logger.info("Running on macOS: preview will run on main thread")
            service.run_preview_loop()  # Blocks until stopped
        else:
            # Keep service running
            logger.info("Service running. Press Ctrl+C to stop.")
            while service.running:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        service.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
