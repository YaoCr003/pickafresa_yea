'''
Camera Capture API for RealSense and Generic Cameras
Unified interface for capturing images from various camera types
Supports both RealSense D400 series and standard USB/webcams via OpenCV

Usage:
    from vision_tools import camera_capture
    
    ## Automatic camera detection
    cam = camera_capture.auto_detect_camera(resolution=(640, 480))
    frame = cam.read()
    cam.release()
    
    ## Or use the high-level capture function
    camera_capture.capture_images(
        num_images=10,
        output_dir='./captures',
        prefix='img',
        camera_type='realsense'
    )
    
Features:
    - Automatic camera detection (tries RealSense first, falls back to OpenCV)
    - Manual camera type selection (realsense/opencv)
    - Interactive capture mode with preview window
    - Press ENTER to start capturing, 's' to pause/resume, ESC to exit
    - Configurable resolution, format, and capture interval
    - Optional metadata logging to JSON
    - Image naming with user-defined prefix, datetime, and frame index
    
@aldrick-t, 2025
'''

import cv2
import datetime
import os
import threading

try:
	import pyrealsense2 as rs
	HAS_REALSENSE = True
except ImportError:
	HAS_REALSENSE = False

class CameraCaptureError(Exception):
	"""Exception raised for camera capture errors"""
	pass

def get_compact_datetime():
	"""
	Generate a compact datetime string for filenames.
	
	Returns:
		str: Datetime string in format YYYYMMDD_HHMMSS
	"""
	return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

class BaseCamera:
	"""
	Base class for camera implementations.
	
	Attributes:
		resolution (tuple): Camera resolution as (width, height)
		camera_type (str): Type identifier for the camera
	"""
	def __init__(self, resolution=(640, 480)):
		self.resolution = resolution
		self.camera_type = 'generic'
	
	def open(self):
		"""Open and initialize the camera connection"""
		raise NotImplementedError
	
	def read(self):
		"""
		Read a single frame from the camera.
		
		Returns:
			numpy.ndarray: BGR image frame
		"""
		raise NotImplementedError
	
	def release(self):
		"""Release camera resources and close connection"""
		raise NotImplementedError
	
	def get_resolution(self):
		"""
		Get the camera resolution.
		
		Returns:
			tuple: Resolution as (width, height)
		"""
		return self.resolution
	
	def get_type(self):
		"""
		Get the camera type identifier.
		
		Returns:
			str: Camera type ('opencv', 'realsense', etc.)
		"""
		return self.camera_type

class OpenCVCamera(BaseCamera):
	"""
	OpenCV-based camera implementation for standard USB/webcams.
	
	Args:
		index (int): Camera device index (default: 0)
		resolution (tuple): Camera resolution as (width, height)
	"""
	def __init__(self, index=0, resolution=(640, 480)):
		super().__init__(resolution)
		self.index = index
		self.cap = None
		self.camera_type = 'opencv'
	
	def open(self):
		"""Open the OpenCV camera"""
		self.cap = cv2.VideoCapture(self.index)
		if not self.cap.isOpened():
			raise CameraCaptureError(f"Cannot open camera index {self.index}")
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
	
	def read(self):
		"""Read a frame from the OpenCV camera"""
		ret, frame = self.cap.read()
		if not ret:
			raise CameraCaptureError("Failed to read frame from OpenCV camera")
		return frame
	
	def release(self):
		"""Release the OpenCV camera"""
		if self.cap:
			self.cap.release()

class RealSenseCamera(BaseCamera):
	"""
	RealSense D400 series camera implementation.
	
	Args:
		resolution (tuple): Camera resolution as (width, height)
	
	Raises:
		CameraCaptureError: If pyrealsense2 is not installed
	"""
	def __init__(self, resolution=(640, 480)):
		super().__init__(resolution)
		self.pipeline = None
		self.profile = None
		self.camera_type = 'realsense'
	
	def open(self):
		"""Open and configure the RealSense camera pipeline"""
		if not HAS_REALSENSE:
			raise CameraCaptureError("pyrealsense2 not installed")
		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, 30)
		self.profile = self.pipeline.start(config)
	
	def read(self):
		"""Read a frame from the RealSense camera"""
		import numpy as np
		frames = self.pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
			raise CameraCaptureError("No color frame from RealSense camera")
		frame = np.asanyarray(color_frame.get_data())
		return frame
	
	def release(self):
		"""Stop the RealSense pipeline"""
		if self.pipeline:
			self.pipeline.stop()

def auto_detect_camera(resolution=(640, 480)):
	"""
	Automatically detect and open an available camera.
	Tries RealSense first, then falls back to OpenCV cameras.
	
	Args:
		resolution (tuple): Desired camera resolution as (width, height)
	
	Returns:
		BaseCamera: Opened camera object (RealSenseCamera or OpenCVCamera)
	
	Raises:
		CameraCaptureError: If no camera is detected
	"""
	# Try RealSense first
	if HAS_REALSENSE:
		try:
			cam = RealSenseCamera(resolution)
			cam.open()
			return cam
		except Exception:
			pass
	# Fallback to OpenCV
	for idx in range(3):
		try:
			cam = OpenCVCamera(idx, resolution)
			cam.open()
			return cam
		except Exception:
			continue
	raise CameraCaptureError("No camera detected. Please specify camera index or type.")

def get_camera(camera_type=None, index=0, resolution=(640, 480)):
	"""
	Get a camera instance based on type specification.
	
	Args:
		camera_type (str, optional): Camera type ('realsense', 'opencv', or None for auto)
		index (int): Camera index for OpenCV cameras (default: 0)
		resolution (tuple): Camera resolution as (width, height)
	
	Returns:
		BaseCamera: Opened camera object
	
	Raises:
		CameraCaptureError: If camera cannot be opened
	"""
	if camera_type == 'realsense':
		cam = RealSenseCamera(resolution)
		cam.open()
		return cam
	elif camera_type == 'opencv':
		cam = OpenCVCamera(index, resolution)
		cam.open()
		return cam
	else:
		return auto_detect_camera(resolution)

def overlay_metadata(frame, frame_idx, timestamp, resolution, camera_type):
	"""
	Overlay metadata text on a frame.
	
	Args:
		frame (numpy.ndarray): Image frame to overlay text on
		frame_idx (int): Current frame index
		timestamp (str): Timestamp string
		resolution (tuple): Resolution as (width, height)
		camera_type (str): Camera type identifier
	
	Returns:
		numpy.ndarray: Frame with overlaid metadata text
	"""
	text = f"Frame: {frame_idx}  Time: {timestamp}  Res: {resolution[0]}x{resolution[1]}  Cam: {camera_type}"
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
	return frame

def capture_images(
	num_images=10,
	output_dir="./captures",
	prefix="img",
	image_format="png",
	interval=1.0,
	resolution=(640, 480),
	camera_type=None,
	camera_index=0,
	show_preview=True,
	log_metadata=False,
	metadata_path=None
):
	"""
	Capture a series of images from a camera with interactive controls.
	
	This function opens a camera preview window and allows the user to:
	- Press ENTER to start capturing images
	- Press 's' to pause/resume capturing
	- Press ESC to exit early
	
	Args:
		num_images (int): Number of images to capture (default: 10)
		output_dir (str): Directory to save images (default: "./captures")
		prefix (str): Filename prefix for images (default: "img")
		image_format (str): Image format ('png', 'jpg', 'jpeg') (default: "png")
		interval (float): Time interval between captures in seconds (default: 1.0)
		resolution (tuple): Camera resolution as (width, height) (default: (640, 480))
		camera_type (str, optional): Camera type ('realsense', 'opencv', or None for auto)
		camera_index (int): Camera index for OpenCV cameras (default: 0)
		show_preview (bool): Whether to show preview window (default: True)
		log_metadata (bool): Whether to log metadata to JSON (default: False)
		metadata_path (str, optional): Path to save metadata JSON file
	
	Raises:
		CameraCaptureError: If camera cannot be opened or read fails
	
	Returns:
		None
	
	Example:
		>>> capture_images(
		...     num_images=20,
		...     output_dir='./my_captures',
		...     prefix='test',
		...     camera_type='realsense',
		...     resolution=(1280, 720)
		... )
	"""
	import time
	import numpy as np
	cam = get_camera(camera_type, camera_index, resolution)
	batch_dir = os.path.join(output_dir, f"{prefix}_{get_compact_datetime()}")
	os.makedirs(batch_dir, exist_ok=True)
	metadata = []
	try:
		i = 0
		capturing = False
		paused = False
		print("\nPreview window opened. Press ENTER to start capturing images. Press 's' to pause/resume during capture. Press ESC to exit early.")
		while i < num_images:
			frame = cam.read()
			timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
			fname = f"{prefix}_{timestamp}_{i:03d}.{image_format}"
			fpath = os.path.join(batch_dir, fname)
			if show_preview:
				disp = frame.copy()
				overlay_metadata(disp, i, timestamp, resolution, cam.get_type())
				status = "Paused" if paused else ("Ready" if not capturing else "Capturing")
				cv2.putText(disp, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
				cv2.putText(disp, "ENTER: Start | s: Pause/Resume | ESC: Exit", (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
				cv2.imshow("Camera Capture", disp)
				key = cv2.waitKey(1)
				if not capturing and (key == 13 or key == 10):  # ENTER
					capturing = True
					paused = False
					print("Started capturing images...")
				elif key == ord('s'):
					if capturing:
						paused = not paused
						print("Paused." if paused else "Resumed.")
				elif key == 27:
					print("Exiting early.")
					break
			if capturing and not paused:
				cv2.imwrite(fpath, frame)
				metadata.append({
					"filename": fname,
					"frame_idx": i,
					"timestamp": timestamp,
					"resolution": resolution,
					"camera_type": cam.get_type()
				})
				i += 1
				time.sleep(interval)
	finally:
		cam.release()
		if show_preview:
			cv2.destroyAllWindows()
		if log_metadata and metadata_path:
			import json
			with open(metadata_path, 'w') as f:
				json.dump(metadata, f, indent=2)

__all__ = [
	"CameraCaptureError",
	"BaseCamera",
	"OpenCVCamera",
	"RealSenseCamera",
	"get_camera",
	"auto_detect_camera",
	"capture_images",
	"overlay_metadata"
]
