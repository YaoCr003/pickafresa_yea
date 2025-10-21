"""
CLI tool for camera capture using the camera_capture API.
"""

import sys
import os

# Fix import for camera_capture regardless of run context
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
try:
    from vision_tools import camera_capture
except ImportError:
    import camera_capture

def prompt_int(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    return int(val) if val else default

def prompt_float(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    return float(val) if val else default

def prompt_str(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def prompt_choice(prompt, choices, default):
    choices_str = '/'.join(choices)
    val = input(f"{prompt} ({choices_str}) [{default}]: ").strip().lower()
    return val if val in choices else default

def main():
    print("\n--- Camera Capture CLI Tool ---\n")
    num_images = prompt_int("Number of images to capture", 10)
    output_dir = prompt_str("Directory to save images", "./pickafresa_vision/images/")
    prefix = prompt_str("Image filename prefix", "img")
    image_format = prompt_choice("Image format", ['png', 'jpg', 'jpeg'], 'png')
    interval = prompt_float("Interval between captures (seconds)", 1.0)
    resolution_str = prompt_str("Resolution as WIDTHxHEIGHT", "640x480")
    try:
        width, height = map(int, resolution_str.lower().split('x'))
    except Exception:
        print('Invalid resolution format. Using default 640x480.')
        width, height = 640, 480
    camera_type = prompt_choice("Camera type", ['auto', 'realsense', 'opencv'], 'auto')
    camera_type = None if camera_type == 'auto' else camera_type
    camera_index = prompt_int("Camera index for OpenCV cameras", 0)
    show_preview = prompt_choice("Show preview window?", ['y', 'n'], 'y') == 'y'
    log_metadata = prompt_choice("Log metadata to JSON?", ['y', 'n'], 'n') == 'y'
    metadata_path = None
    if log_metadata:
        metadata_path = prompt_str("Path to save metadata JSON", os.path.join(output_dir, f'{prefix}_metadata.json'))

    try:
        camera_capture.capture_images(
            num_images=num_images,
            output_dir=output_dir,
            prefix=prefix,
            image_format=image_format,
            interval=interval,
            resolution=(width, height),
            camera_type=camera_type,
            camera_index=camera_index,
            show_preview=show_preview,
            log_metadata=log_metadata,
            metadata_path=metadata_path
        )
    except camera_capture.CameraCaptureError as e:
        print(f'Camera error: {e}')
        sys.exit(2)
    except Exception as e:
        print(f'Unexpected error: {e}')
        sys.exit(3)

if __name__ == '__main__':
    main()
