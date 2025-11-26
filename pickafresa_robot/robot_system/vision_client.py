"""
Vision Service Client for Robot PnP System

Handles IPC communication with the vision_service for fruit detection and pose estimation.
Provides a clean interface for requesting captures and parsing detection results.

Communication Protocol:
- Socket-based IPC (default: localhost:5555)
- JSON request/response format
- Commands: "status", "capture", "shutdown"

by: Aldrick T, 2025
for Team YEA
"""

import socket
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class VisionServiceError(Exception):
    """Raised when vision service is not available or fails."""
    pass


class FruitDetection:
    """
    Container for a single fruit detection with pose.
    
    Attributes:
        bbox_cxcywh: Bounding box in [cx, cy, width, height] format
        confidence: Detection confidence (0.0-1.0)
        class_name: Class label ("ripe", "unripe", "flower")
        class_id: Numeric class ID
        success: Whether PnP pose estimation succeeded
        T_cam_fruit: 4x4 transformation matrix (fruit pose in camera frame, meters)
        position_cam: [x, y, z] position in camera frame (meters)
        T_base_fruit: 4x4 transformation in robot base frame (computed later)
        position_base: [x, y, z] position in base frame (computed later)
    """
    
    def __init__(self, detection_dict: Dict[str, Any]):
        """
        Initialize from detection dictionary.
        
        Args:
            detection_dict: Dictionary from vision service response
        """
        self.bbox_cxcywh = detection_dict.get("bbox_cxcywh", [0, 0, 0, 0])
        self.confidence = detection_dict.get("confidence", 0.0)
        self.class_name = detection_dict.get("class_name", "unknown")
        self.class_id = detection_dict.get("class_id", -1)
        self.success = detection_dict.get("success", False)
        
        # Pose information
        T_cam_fruit_list = detection_dict.get("T_cam_fruit")
        if T_cam_fruit_list is not None:
            self.T_cam_fruit = np.array(T_cam_fruit_list, dtype=np.float64)
            # Ensure it's a 4x4 matrix
            if self.T_cam_fruit.shape != (4, 4):
                self.T_cam_fruit = self.T_cam_fruit.reshape(4, 4)
        else:
            self.T_cam_fruit = None
        
        pos_list = detection_dict.get("position_cam")
        self.position_cam = np.array(pos_list) if pos_list else None
        
        # Additional metadata
        self.error_reason = detection_dict.get("error_reason")
        self.depth_samples = detection_dict.get("depth_samples")
        self.median_depth = detection_dict.get("median_depth")
        self.sampling_strategy = detection_dict.get("sampling_strategy")
        
        # Robot base frame pose (computed later by transform_utils)
        self.T_base_fruit: Optional[np.ndarray] = None
        self.position_base: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        status = "[OK]" if self.success else "[FAIL]"
        if self.position_cam is not None:
            pos_str = f"[{self.position_cam[0]:.3f}, {self.position_cam[1]:.3f}, {self.position_cam[2]:.3f}]m"
        else:
            pos_str = "N/A"
        
        return (f"FruitDetection({status} {self.class_name}, "
                f"conf={self.confidence:.2f}, pos_cam={pos_str})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "bbox_cxcywh": self.bbox_cxcywh,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "success": self.success,
            "error_reason": self.error_reason,
            "depth_samples": self.depth_samples,
            "median_depth": self.median_depth,
            "sampling_strategy": self.sampling_strategy
        }
        
        if self.T_cam_fruit is not None:
            result["T_cam_fruit"] = self.T_cam_fruit.tolist()
        
        if self.position_cam is not None:
            result["position_cam"] = self.position_cam.tolist()
        
        if self.T_base_fruit is not None:
            result["T_base_fruit"] = self.T_base_fruit.tolist()
        
        if self.position_base is not None:
            result["position_base"] = self.position_base.tolist()
        
        return result


class VisionServiceClient:
    """Client for communicating with the vision service via IPC."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 30.0, logger=None):
        """
        Initialize vision service client.
        
        Args:
            host: Vision service host address
            port: Vision service port
            timeout: Socket timeout in seconds
            logger: Optional logger instance
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logger = logger
        self.socket: Optional[socket.socket] = None
    
    def _log(self, level: str, message: str):
        """Internal logging helper."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def connect(self) -> bool:
        """
        Connect to vision service.
        
        Returns:
            True if connected successfully
        
        Raises:
            VisionServiceError: If connection fails
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self._log("info", f"[OK] Connected to vision service at {self.host}:{self.port}")
            return True
        
        except socket.timeout:
            raise VisionServiceError(f"Connection timeout to {self.host}:{self.port}")
        except ConnectionRefusedError:
            raise VisionServiceError(
                f"Connection refused by {self.host}:{self.port} - Is vision service running?\n"
                f"  Start with: ./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py"
            )
        except Exception as e:
            raise VisionServiceError(f"Failed to connect to vision service: {e}")
    
    def disconnect(self):
        """Disconnect from vision service."""
        if self.socket:
            try:
                self.socket.close()
                self._log("info", "Disconnected from vision service")
            except:
                pass
            self.socket = None
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check vision service status.
        
        Returns:
            Status dictionary with keys: alive, ready, error
        
        Raises:
            VisionServiceError: If request fails
        """
        request = {"command": "status"}
        return self._send_request(request)
    
    def request_capture(
        self,
        multi_frame: bool = False,
        num_frames: int = 10,
        class_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[FruitDetection]:
        """
        Request a capture from vision service.
        
        Args:
            multi_frame: Enable multi-frame averaging
            num_frames: Number of frames to average
            class_filter: Optional list of class names to keep (e.g., ["ripe"])
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of FruitDetection objects
        
        Raises:
            VisionServiceError: If request fails
        """
        request = {
            "command": "capture",
            "multi_frame": multi_frame,
            "num_frames": num_frames
        }
        
        response = self._send_request(request)
        
        if not response.get("success", False):
            error_msg = response.get("error", "Unknown error")
            raise VisionServiceError(f"Capture failed: {error_msg}")
        
        # Parse detections
        detections_raw = response.get("detections", [])
        detections = [FruitDetection(d) for d in detections_raw]
        
        # Filter by success and confidence
        detections = [
            d for d in detections
            if d.success and d.confidence >= min_confidence
        ]
        
        # Filter by class if specified
        if class_filter:
            detections = [d for d in detections if d.class_name in class_filter]
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: -d.confidence)
        
        self._log("info", f"[OK] Received {len(detections)} valid detections from vision service")
        
        return detections
    
    def request_shutdown(self):
        """Request vision service to shutdown gracefully."""
        try:
            request = {"command": "shutdown"}
            self._send_request(request)
            self._log("info", "Sent shutdown request to vision service")
        except:
            pass  # Service may close connection before responding
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to vision service and receive response.
        
        Args:
            request: Request dictionary
        
        Returns:
            Response dictionary
        
        Raises:
            VisionServiceError: If communication fails
        """
        if not self.socket:
            raise VisionServiceError("Not connected to vision service")
        
        try:
            # Send request (JSON + newline)
            request_json = json.dumps(request)
            self.socket.sendall(request_json.encode('utf-8') + b'\n')
            
            # Receive response (read until newline)
            response_data = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise VisionServiceError("Connection closed by vision service")
                response_data += chunk
                if b'\n' in response_data:
                    break
            
            # Parse response
            response_json = response_data.decode('utf-8').strip()
            response = json.loads(response_json)
            
            return response
        
        except socket.timeout:
            raise VisionServiceError("Request timeout - vision service not responding")
        except json.JSONDecodeError as e:
            raise VisionServiceError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise VisionServiceError(f"Communication error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage and tests
if __name__ == "__main__":
    print("Vision Service Client - Unit Tests")
    print("=" * 70)
    
    # Test 1: Connection (requires vision_service to be running)
    print("\n[Test 1] Connect to vision service")
    print("NOTE: This requires vision_service to be running on localhost:5555")
    print("  Start with: ./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py")
    
    try:
        client = VisionServiceClient(host="127.0.0.1", port=5555, timeout=5.0)
        client.connect()
        
        # Test 2: Check status
        print("\n[Test 2] Check service status")
        status = client.check_status()
        print(f"Status: {status}")
        
        # Test 3: Request capture
        print("\n[Test 3] Request capture")
        response = input("Capture frame? [y/N]: ").strip().lower()
        if response == 'y':
            detections = client.request_capture(
                multi_frame=False,
                min_confidence=0.5,
                class_filter=["ripe"]
            )
            
            print(f"\nReceived {len(detections)} detections:")
            for i, det in enumerate(detections, 1):
                print(f"  [{i}] {det}")
        
        # Disconnect
        client.disconnect()
        print("\n[OK] All tests completed!")
    
    except VisionServiceError as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure vision_service is running:")
        print("  ./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py")
    
    print("\n" + "=" * 70)
