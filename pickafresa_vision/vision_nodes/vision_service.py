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
from datetime import datetime
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
from pickafresa_vision.vision_tools.supabase_uploader import SupabaseUploader

# Local imports - Robot (for ROS2 logger)
try:
    from pickafresa_robot.robot_system.ros2_logger import create_logger
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

# Use ROS2-style logger if available
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
    # Fallback: configure root logger once
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [vision_service]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [vision_service]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
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
        self.model_path: Optional[Path] = None  # Track model path for HUD
        self.supabase_uploader: Optional[SupabaseUploader] = None  # Supabase cloud uploader
        
        # IPC
        self.server_socket: Optional[socket.socket] = None
        self.server_thread: Optional[threading.Thread] = None
        
        # Preview
        self.preview_enabled = self.config.get("preview", {}).get("enabled", False)
        self.preview_thread: Optional[threading.Thread] = None
        
        # Overlay state (runtime toggleable)
        self.show_depth_overlay = False  # Toggle depth colormap overlay
        self.show_hsv_overlay = False    # Toggle HSV filter mask overlay
        self.depth_overlay_alpha = 0.4   # Depth overlay transparency (0.0-1.0)
        self.hsv_overlay_alpha = 0.5     # HSV overlay transparency (0.0-1.0)
        self.depth_colormap = cv2.COLORMAP_JET  # Current colormap
        
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
                    self.model_path = full_model_path  # Store for HUD
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
            
            # Initialize Supabase uploader (if enabled)
            supabase_config = self.config.get("capture", {}).get("supabase", {})
            if supabase_config.get("enabled", False):
                logger.info("Initializing Supabase uploader...")
                self.supabase_uploader = SupabaseUploader(enabled=True)
                if self.supabase_uploader.is_enabled():
                    logger.info("[OK] Supabase uploader initialized")
                else:
                    logger.warning("Supabase uploader failed to initialize (check .env credentials)")
                    self.supabase_uploader = None
            else:
                logger.info("Supabase upload disabled in config")
            
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
                            num_frames=request.get("num_frames", 1),
                            is_ipc_request=True  # Enable logging for IPC requests
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
        num_frames: int = 1,
        is_ipc_request: bool = False
    ) -> Dict[str, Any]:
        """
        Capture frame(s), run inference, and compute PnP.
        
        Args:
            multi_frame: Use multi-frame averaging
            num_frames: Number of frames to capture (if multi_frame=True)
            is_ipc_request: True if called from IPC request (enables logging)
        
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
            
            # Only log performance for IPC requests, not for preview loop
            if is_ipc_request:
                log_every = self.config.get("logging", {}).get("performance", {}).get("log_every_n_frames", 30)
                if self.frame_count % log_every == 0:
                    self._log_performance()
            
            # Generate timestamp for saving
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save captures to disk and upload to Supabase (if configured)
            if is_ipc_request:
                save_config = self.config.get("capture", {}).get("save_captures", {})
                if save_config.get("enabled", False):
                    self._save_and_upload_capture(
                        color_frame=color_frame,
                        depth_frame=depth_frame,
                        pnp_results=pnp_results,
                        timestamp=timestamp
                    )
            
            return {
                "success": True,
                "detections": pnp_results,
                "timestamp": timestamp,
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
            # Get camera intrinsics from RealSense SDK directly
            # This avoids validation warnings from potentially corrupted calibration files
            intrinsics = self.camera.get_intrinsics(source="realsense")
            
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
    
    def _save_and_upload_capture(
        self,
        color_frame: np.ndarray,
        depth_frame: Any,
        pnp_results: List[Dict[str, Any]],
        timestamp: str
    ):
        """
        Save capture data to disk and upload to Supabase (if enabled).
        
        Args:
            color_frame: RGB image
            depth_frame: Depth frame
            pnp_results: List of detection results with PnP data
            timestamp: Timestamp string for filenames
        """
        try:
            from pickafresa_vision.vision_tools.data_persistence import DataSaver
            
            save_config = self.config.get("capture", {}).get("save_captures", {})
            captures_dir = REPO_ROOT / save_config.get("directory", "pickafresa_vision/captures")
            
            # Initialize DataSaver
            data_saver = DataSaver(output_dir=captures_dir)
            
            # Prepare data for saving
            # Convert PnP results dict to objects if needed (simplified - just save raw data)
            # For now, we'll save the JSON data directly
            
            # Save raw image
            if save_config.get("save_rgb", True):
                data_saver.save_raw_image(color_frame, timestamp)
            
            # Save annotated image with bboxes and overlays
            bbox_filepath = None
            if save_config.get("save_rgb", True):
                # Extract bboxes from pnp_results
                bboxes_cxcywh = [r.get("bbox_cxcywh", [0, 0, 0, 0]) for r in pnp_results]
                
                # Create list of result objects for annotation (simplified)
                # The save_annotated_image expects result objects, we'll pass the dicts
                _, bbox_filepath = data_saver.save_annotated_image(
                    color_frame,
                    pnp_results,  # Pass the results dicts directly
                    bboxes_cxcywh,
                    timestamp=timestamp,
                    show_all_detections=True,
                    depth_frame=depth_frame
                )
            
            # Save JSON metadata
            if save_config.get("save_json", True):
                # Build JSON structure similar to data_persistence format
                intrinsics = self.camera.get_intrinsics() if self.camera else {}
                
                metadata = {
                    "timestamp": timestamp,
                    "timestamp_iso": datetime.now().isoformat(),
                    "resolution": {
                        "width": color_frame.shape[1],
                        "height": color_frame.shape[0]
                    },
                    "camera_intrinsics": intrinsics.to_dict() if hasattr(intrinsics, 'to_dict') else str(intrinsics),
                    "model_path": str(self.model_path) if self.model_path else None,
                    "detections": pnp_results,
                    "summary": {
                        "total_detections": len(pnp_results),
                        "successful_poses": sum(1 for r in pnp_results if r.get("success", False)),
                        "failed_poses": sum(1 for r in pnp_results if not r.get("success", False)),
                    }
                }
                
                json_filename = f"{timestamp}_data.json"
                json_filepath = captures_dir / json_filename
                
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved capture data: {timestamp}")
                
                # Upload to Supabase (if enabled)
                supabase_config = self.config.get("capture", {}).get("supabase", {})
                logger.debug(f"Supabase config: enabled={supabase_config.get('enabled', False)}, uploader_ready={self.supabase_uploader is not None}")
                
                if supabase_config.get("enabled", False) and self.supabase_uploader:
                    upload_mode = supabase_config.get("upload_mode", "async")
                    upload_rgb = supabase_config.get("upload_rgb", True)
                    upload_json = supabase_config.get("upload_json", True)
                    
                    logger.info(f"Starting Supabase upload (mode: {upload_mode})...")
                    
                    if upload_rgb and upload_json:
                        # Get file paths - use bbox (annotated) image instead of raw
                        image_path = bbox_filepath if bbox_filepath else captures_dir / f"{timestamp}_bbox.png"
                        json_path = json_filepath
                        
                        logger.info(f"Upload paths: image={image_path}, json={json_path}")
                        
                        if upload_mode == "async":
                            # Non-blocking upload
                            def upload_callback(success, message, results):
                                if success:
                                    logger.info(f"✓ Uploaded to Supabase: {timestamp}")
                                else:
                                    logger.warning(f"✗ Supabase upload failed: {message}")
                            
                            self.supabase_uploader.upload_capture_async(
                                image_path,
                                json_path,
                                callback=upload_callback
                            )
                        else:
                            # Blocking upload
                            results = self.supabase_uploader.upload_capture(image_path, json_path)
                            if results["success"]:
                                logger.info(f"✓ Uploaded to Supabase: {timestamp}")
                            else:
                                logger.warning(f"✗ Supabase upload failed: {results.get('error')}")
                else:
                    if not supabase_config.get("enabled", False):
                        logger.debug("Supabase upload disabled in config")
                    elif not self.supabase_uploader:
                        logger.debug("Supabase uploader not initialized")
        
        except Exception as e:
            logger.error(f"Error saving/uploading capture: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
                
                # Compute PnP for detections (for depth display)
                pnp_results = self._compute_pnp(color_frame, depth_frame, detections) if detections else []
                
                # Draw detections with PnP results
                preview_frame = self._draw_preview(color_frame, depth_frame, pnp_results)
                
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
                elif key == ord('d'):
                    self.show_depth_overlay = not self.show_depth_overlay
                    logger.info(f"Depth overlay: {'ON' if self.show_depth_overlay else 'OFF'}")
                elif key == ord('h'):
                    self.show_hsv_overlay = not self.show_hsv_overlay
                    logger.info(f"HSV filter overlay: {'ON' if self.show_hsv_overlay else 'OFF'}")
                elif key == ord('['):
                    # Decrease depth overlay alpha
                    self.depth_overlay_alpha = max(0.0, self.depth_overlay_alpha - 0.1)
                    logger.info(f"Depth overlay alpha: {self.depth_overlay_alpha:.1f}")
                elif key == ord(']'):
                    # Increase depth overlay alpha
                    self.depth_overlay_alpha = min(1.0, self.depth_overlay_alpha + 0.1)
                    logger.info(f"Depth overlay alpha: {self.depth_overlay_alpha:.1f}")
                elif key == ord(','):
                    # Decrease HSV overlay alpha
                    self.hsv_overlay_alpha = max(0.0, self.hsv_overlay_alpha - 0.1)
                    logger.info(f"HSV overlay alpha: {self.hsv_overlay_alpha:.1f}")
                elif key == ord('.'):
                    # Increase HSV overlay alpha
                    self.hsv_overlay_alpha = min(1.0, self.hsv_overlay_alpha + 0.1)
                    logger.info(f"HSV overlay alpha: {self.hsv_overlay_alpha:.1f}")
                elif key == ord('c'):
                    # Cycle through colormaps
                    colormaps = [
                        (cv2.COLORMAP_JET, "JET"),
                        (cv2.COLORMAP_HOT, "HOT"),
                        (cv2.COLORMAP_RAINBOW, "RAINBOW"),
                        (cv2.COLORMAP_OCEAN, "OCEAN"),
                        (cv2.COLORMAP_VIRIDIS, "VIRIDIS"),
                        (cv2.COLORMAP_TURBO, "TURBO")
                    ]
                    current_idx = next((i for i, (cm, _) in enumerate(colormaps) if cm == self.depth_colormap), 0)
                    next_idx = (current_idx + 1) % len(colormaps)
                    self.depth_colormap, colormap_name = colormaps[next_idx]
                    logger.info(f"Depth colormap: {colormap_name}")
                elif key == ord('?'):
                    self._print_keyboard_help()
            
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
    
    def _create_depth_overlay(self, depth_frame, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create colorized depth overlay from depth frame.
        
        Args:
            depth_frame: RealSense depth frame
            target_shape: (height, width) of RGB frame for alignment
        
        Returns:
            BGR colorized depth image matching target_shape
        """
        try:
            # Get depth data as numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Normalize depth to 0-255 range
            depth_cfg = self.config.get("preview", {}).get("depth", {})
            min_depth_m = depth_cfg.get("min_depth_m", 0.1)
            max_depth_m = depth_cfg.get("max_depth_m", 3.0)
            
            # Convert to meters and clip
            depth_m = depth_image.astype(np.float32) * depth_frame.get_units()
            depth_m = np.clip(depth_m, min_depth_m, max_depth_m)
            
            # Normalize to 0-255
            depth_normalized = ((depth_m - min_depth_m) / (max_depth_m - min_depth_m) * 255).astype(np.uint8)
            
            # Apply colormap
            depth_colorized = cv2.applyColorMap(depth_normalized, self.depth_colormap)
            
            # Resize to match RGB frame if needed
            if depth_colorized.shape[:2] != target_shape:
                depth_colorized = cv2.resize(depth_colorized, (target_shape[1], target_shape[0]))
            
            return depth_colorized
        
        except Exception as e:
            logger.error(f"Error creating depth overlay: {e}")
            # Return black image on error
            return np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
    
    def _create_hsv_overlay(self, color_image: np.ndarray, pnp_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create HSV filter visualization overlay showing red strawberry surface detection.
        
        Args:
            color_image: RGB image
            pnp_results: List of PnP results with bounding boxes
        
        Returns:
            BGR image with HSV filter masks visualized (green = kept red pixels)
        """
        try:
            # Create blank overlay
            h, w = color_image.shape[:2]
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Get PnP config for HSV parameters
            pnp_config_path = self.config.get("pnp", {}).get("config_path", "")
            full_pnp_config_path = REPO_ROOT / pnp_config_path if pnp_config_path else None
            
            if not full_pnp_config_path or not full_pnp_config_path.exists():
                return overlay
            
            # Load HSV parameters from PnP config
            with open(full_pnp_config_path, 'r') as f:
                pnp_config = yaml.safe_load(f)
            
            color_cfg = pnp_config.get("depth_sampling", {}).get("color_filter", {})
            if not color_cfg.get("enabled", False):
                # Draw text indicating filter is disabled
                cv2.putText(overlay, "HSV Filter: DISABLED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return overlay
            
            mode = color_cfg.get("mode", "adaptive")
            
            # Get HSV parameters based on mode
            if mode == "preset":
                preset_cfg = color_cfg.get("preset", {})
                sat_min = preset_cfg.get("saturation_min", 50)
                val_min = preset_cfg.get("value_min", 50)
                hue_min_1 = preset_cfg.get("hue_min_1", 0)
                hue_max_1 = preset_cfg.get("hue_max_1", 10)
                hue_min_2 = preset_cfg.get("hue_min_2", 160)
                hue_max_2 = preset_cfg.get("hue_max_2", 179)
            else:  # adaptive
                adaptive_cfg = color_cfg.get("adaptive", {})
                sat_min = adaptive_cfg.get("saturation_min", 50)
                val_min = adaptive_cfg.get("value_min", 50)
                # Use preset defaults for visualization
                hue_min_1 = 0
                hue_max_1 = 10
                hue_min_2 = 160
                hue_max_2 = 179
            
            # Draw filter visualization for each detection's bbox
            for result in pnp_results:
                cx, cy, ww, hh = result["bbox_cxcywh"]
                x1 = int(max(0, cx - ww/2))
                y1 = int(max(0, cy - hh/2))
                x2 = int(min(w, cx + ww/2))
                y2 = int(min(h, cy + hh/2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract ROI
                roi = color_image[y1:y2, x1:x2]
                
                # Convert to HSV
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                
                # Create red mask (two ranges due to HSV wrap-around)
                lower1 = np.array([hue_min_1, sat_min, val_min], dtype=np.uint8)
                upper1 = np.array([hue_max_1, 255, 255], dtype=np.uint8)
                lower2 = np.array([hue_min_2, sat_min, val_min], dtype=np.uint8)
                upper2 = np.array([hue_max_2, 255, 255], dtype=np.uint8)
                
                mask1 = cv2.inRange(hsv_roi, lower1, upper1)
                mask2 = cv2.inRange(hsv_roi, lower2, upper2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Colorize mask (green for kept red pixels)
                mask_colorized = np.zeros_like(roi)
                mask_colorized[red_mask > 0] = [0, 255, 0]  # Green in RGB (will convert to BGR)
                
                # Place in overlay
                overlay[y1:y2, x1:x2] = cv2.cvtColor(mask_colorized, cv2.COLOR_RGB2BGR)
            
            return overlay
        
        except Exception as e:
            logger.error(f"Error creating HSV overlay: {e}")
            # Return black image on error
            h, w = color_image.shape[:2] if len(color_image.shape) >= 2 else (480, 640)
            return np.zeros((h, w, 3), dtype=np.uint8)
    
    def _print_keyboard_help(self):
        """Print keyboard shortcuts to console."""
        print()
        print("=" * 70)
        print("KEYBOARD SHORTCUTS")
        print("=" * 70)
        print("  q     - Quit service")
        print("  p     - Pause/Resume preview")
        print("  d     - Toggle depth overlay")
        print("  h     - Toggle HSV filter overlay")
        print("  [     - Decrease depth overlay alpha")
        print("  ]     - Increase depth overlay alpha")
        print("  ,     - Decrease HSV overlay alpha")
        print("  .     - Increase HSV overlay alpha")
        print("  c     - Cycle depth colormap")
        print("  ?     - Show this help")
        print("=" * 70)
        print()
    
    def _draw_preview(
        self,
        color_frame,
        depth_frame,
        pnp_results: List[Dict[str, Any]]
    ):
        """
        Draw preview frame with comprehensive overlays.
        
        Shows:
        - Bounding boxes with class, confidence, and depth (raw + final)
        - Frame statistics HUD (FPS, frames captured, model name, detection count)
        - Optional depth colormap overlay (toggle with 'd')
        - Optional HSV filter mask overlay (toggle with 'h')
        """
        # Clone frame for drawing
        preview = color_frame.copy()
        
        # Convert RGB to BGR for OpenCV display
        # RealSense returns RGB, but cv2.imshow expects BGR
        preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        h, w = preview.shape[:2]
        
        # Apply depth overlay if enabled
        if self.show_depth_overlay and depth_frame is not None:
            depth_colorized = self._create_depth_overlay(depth_frame, (h, w))
            preview = cv2.addWeighted(preview, 1.0 - self.depth_overlay_alpha, 
                                     depth_colorized, self.depth_overlay_alpha, 0)
        
        # Apply HSV filter overlay if enabled
        if self.show_hsv_overlay and pnp_results:
            hsv_overlay = self._create_hsv_overlay(color_frame, pnp_results)
            preview = cv2.addWeighted(preview, 1.0 - self.hsv_overlay_alpha,
                                     hsv_overlay, self.hsv_overlay_alpha, 0)
        
        # Draw detections with PnP results
        for result in pnp_results:
            cx, cy, ww, hh = result["bbox_cxcywh"]
            x1 = int(cx - ww/2)
            y1 = int(cy - hh/2)
            x2 = int(cx + ww/2)
            y2 = int(cy + hh/2)
            
            # Draw bbox (BGR format: Green = (0, 255, 0))
            color = (0, 255, 0)  # Green in BGR
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            
            # Build label with class, confidence, and depth
            label = f"{result['class_name']} {result['confidence']:.2f}"
            
            # Add depth information if PnP succeeded
            if result.get("success") and result.get("median_depth") is not None:
                raw_depth_mm = result["median_depth"] * 1000  # Convert m to mm
                
                # Check if offset was applied (from position_cam Z vs median_depth)
                if result.get("position_cam") is not None:
                    final_depth_mm = result["position_cam"][2] * 1000  # Z coordinate in mm
                    
                    # Show both raw and final if different (offset applied)
                    if abs(final_depth_mm - raw_depth_mm) > 1.0:  # > 1mm difference
                        label += f" | {raw_depth_mm:.0f}mm -> {final_depth_mm:.0f}mm"
                    else:
                        label += f" | {raw_depth_mm:.0f}mm"
                else:
                    label += f" | {raw_depth_mm:.0f}mm"
            elif result.get("median_depth") is not None:
                # PnP failed but we have depth measurement
                raw_depth_mm = result["median_depth"] * 1000
                label += f" | {raw_depth_mm:.0f}mm"
            
            # Draw label with background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background (black)
            label_y = max(y1 - 5, label_size[1] + 5)  # Ensure label is visible
            cv2.rectangle(preview, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 4, label_y + 2), (0, 0, 0), -1)
            
            # Draw label text (white)
            cv2.putText(preview, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw comprehensive HUD (top-left corner)
        self._draw_hud(preview, pnp_results)
        
        return preview
    
    def _draw_hud(self, frame: np.ndarray, pnp_results: List[Dict[str, Any]]) -> None:
        """
        Draw comprehensive heads-up display with frame statistics.
        
        Args:
            frame: BGR frame to draw on (modified in-place)
            pnp_results: List of PnP results for detection count
        """
        h, w = frame.shape[:2]
        
        # Calculate FPS
        fps = 0.0
        if self.perf_stats["total_time"] and len(self.perf_stats["total_time"]) >= 5:
            # Use last 30 frames for stable FPS calculation
            recent_times = self.perf_stats["total_time"][-30:]
            fps = 1.0 / np.mean(recent_times) if np.mean(recent_times) > 0 else 0.0
        
        # Prepare HUD text lines
        model_name = self.model_path.name if self.model_path else "No Model"
        
        hud_lines = [
            f"Model: {model_name}",
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Detections: {len(pnp_results)}",
        ]
        
        # Add overlay status if enabled
        if self.show_depth_overlay:
            hud_lines.append(f"Depth Overlay: {self.depth_overlay_alpha:.1f}")
        if self.show_hsv_overlay:
            hud_lines.append(f"HSV Overlay: {self.hsv_overlay_alpha:.1f}")
        
        # Calculate HUD background size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        padding = 8
        
        max_text_width = 0
        for line in hud_lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_text_width = max(max_text_width, text_size[0])
        
        hud_width = max_text_width + padding * 2
        hud_height = len(hud_lines) * line_height + padding * 2
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + hud_width, 5 + hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw HUD text
        y_offset = 5 + padding + 15
        for line in hud_lines:
            cv2.putText(frame, line, (5 + padding, y_offset),
                       font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_offset += line_height


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
