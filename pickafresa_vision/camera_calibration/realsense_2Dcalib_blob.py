'''
RealSense D400 Series Color Camera Calibration Tool
Using a circular grid or checkerboard pattern for calibration
Outputs camera intrinsics to a YAML file
Using standard OpenCV calibration tools

Usage:
    python realsense_2Dcalib_blob.py
    
Features:
    - Supports both circular grid and checkerboard patterns
    - Processes existing calibration images or captures new ones
    - Outputs complete calibration data including intrinsics, distortion coefficients, and metadata
    - Supports JPG and PNG image formats
    - CLI-based with interactive prompts
'''

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import glob
from datetime import datetime
from pathlib import Path


class CameraCalibration:
    """Camera calibration tool for RealSense D400 series cameras"""
    
    def __init__(self):
        self.pattern_type = None
        self.pattern_size = None
        self.square_size = None  # For checkerboard
        self.circle_diameter = None  # For circular grid
        self.circle_spacing = None  # For circular grid
        self.image_dir = None
        self.images = []
        self.image_points = []
        self.object_points = []
        self.image_size = None
        
    def print_header(self):
        """Print tool header"""
        print("\n" + "="*70)
        print("  RealSense D400 Camera Calibration Tool")
        print("="*70 + "\n")
    
    def select_pattern_type(self):
        """Prompt user to select calibration pattern type"""
        print("Select calibration pattern type:")
        print("1. Checkerboard")
        print("2. Circular Grid (Asymmetric)")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                self.pattern_type = "checkerboard"
                print("[OK] Selected: Checkerboard pattern\n")
                break
            elif choice == "2":
                self.pattern_type = "circular"
                print("[OK] Selected: Circular grid pattern\n")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def get_pattern_parameters(self):
        """Get pattern parameters from user"""
        if self.pattern_type == "checkerboard":
            print("Checkerboard Pattern Configuration")
            print("-" * 40)
            print("Enter the number of INTERNAL corners (not squares)")
            print("Example: A standard 8x8 checkerboard has 7x7 internal corners")
            
            while True:
                try:
                    cols = int(input("Number of internal corners per row: ").strip())
                    rows = int(input("Number of internal corners per column: ").strip())
                    if cols > 0 and rows > 0:
                        self.pattern_size = (cols, rows)
                        break
                    else:
                        print("Values must be positive integers.")
                except ValueError:
                    print("Invalid input. Please enter integers.")
            
            while True:
                try:
                    size = float(input("Size of each square (in mm): ").strip())
                    if size > 0:
                        self.square_size = size
                        break
                    else:
                        print("Size must be positive.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            print(f"\n[OK] Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} checkerboard")
            print(f"[OK] Square size: {self.square_size} mm\n")
            
        else:  # circular
            print("Circular Grid Pattern Configuration")
            print("-" * 40)
            print("Enter the grid dimensions (number of circles)")
            
            while True:
                try:
                    cols = int(input("Number of circles per row: ").strip())
                    rows = int(input("Number of circles per column: ").strip())
                    if cols > 0 and rows > 0:
                        self.pattern_size = (cols, rows)
                        break
                    else:
                        print("Values must be positive integers.")
                except ValueError:
                    print("Invalid input. Please enter integers.")
            
            while True:
                try:
                    diameter = float(input("Circle diameter (in mm): ").strip())
                    if diameter > 0:
                        self.circle_diameter = diameter
                        break
                    else:
                        print("Diameter must be positive.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            while True:
                try:
                    spacing = float(input("Center-to-center spacing between circles (in mm): ").strip())
                    if spacing > diameter:
                        self.circle_spacing = spacing
                        break
                    else:
                        print(f"Spacing must be greater than diameter ({diameter} mm).")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            print(f"\n[OK] Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} circular grid")
            print(f"[OK] Circle diameter: {self.circle_diameter} mm")
            print(f"[OK] Circle spacing: {self.circle_spacing} mm\n")
    
    def select_image_source(self):
        """Select source of calibration images"""
        default_dir = Path(__file__).parent.parent / "images" / "calib"
        
        # Check if default directory exists and has images
        if default_dir.exists():
            image_files = list(default_dir.glob("*.jpg")) + list(default_dir.glob("*.png"))
            if image_files:
                print(f"Found {len(image_files)} images in default directory:")
                print(f"  {default_dir}")
                use_default = input("\nUse these images? (y/n): ").strip().lower()
                if use_default == 'y':
                    self.image_dir = str(default_dir)
                    return
        
        print("\nCalibration Image Source:")
        print("1. Use images from a directory")
        print("2. Capture new images from RealSense camera")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                self.select_image_directory()
                break
            elif choice == "2":
                self.capture_images()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def select_image_directory(self):
        """Prompt user for image directory path"""
        while True:
            path = input("\nEnter path to directory containing calibration images: ").strip()
            path = os.path.expanduser(path)
            
            if os.path.isdir(path):
                image_files = glob.glob(os.path.join(path, "*.jpg")) + \
                             glob.glob(os.path.join(path, "*.png")) + \
                             glob.glob(os.path.join(path, "*.JPG")) + \
                             glob.glob(os.path.join(path, "*.PNG"))
                
                if image_files:
                    self.image_dir = path
                    print(f"[OK] Found {len(image_files)} images in directory\n")
                    break
                else:
                    print("No JPG or PNG images found in directory. Please try another path.")
            else:
                print("Directory does not exist. Please try again.")
    
    def capture_images(self):
        """Capture calibration images from RealSense camera (now using camera_capture API)"""
        from vision_tools import camera_capture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calib_dir = Path(__file__).parent.parent / "images" / "calib" / timestamp
        calib_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = str(calib_dir)
        print(f"\nImages will be saved to: {calib_dir}")
        print("\nStarting RealSense camera...")
        try:
            cam = camera_capture.get_camera(camera_type='realsense', resolution=(640, 640))
            print("[OK] Camera started successfully\n")
            print("Instructions:")
            print("  - Press SPACE to capture an image")
            print("  - Press 'q' to finish and proceed with calibration")
            print("  - Capture at least 10-15 images from different angles")
            print("  - Ensure the entire pattern is visible in each image\n")
            img_count = 0
            while True:
                frame = cam.read()
                display_img = frame.copy()
                cv2.putText(display_img, f"Images captured: {img_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_img, "SPACE: Capture | Q: Finish", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('RealSense Camera - Calibration Capture', display_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space bar
                    img_count += 1
                    filename = calib_dir / f"calib_{img_count:03d}.png"
                    cv2.imwrite(str(filename), frame)
                    print(f"[OK] Captured image {img_count}: {filename.name}")
                elif key == ord('q'):
                    if img_count >= 10:
                        print(f"\n[OK] Captured {img_count} images")
                        break
                    else:
                        print(f"\nWarning: Only {img_count} images captured.")
                        print("At least 10 images recommended for accurate calibration.")
                        confirm = input("Continue anyway? (y/n): ").strip().lower()
                        if confirm == 'y':
                            break
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            print("Make sure RealSense camera is connected.")
            raise
        finally:
            try:
                cam.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            print()
    
    def load_images(self):
        """Load calibration images from directory"""
        print("Loading images...")
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(self.image_dir, "*.png")) + \
                           glob.glob(os.path.join(self.image_dir, "*.JPG")) + \
                           glob.glob(os.path.join(self.image_dir, "*.PNG")))
        
        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                self.images.append((img_file, img))
        
        if self.images:
            self.image_size = (self.images[0][1].shape[1], self.images[0][1].shape[0])
            print(f"[OK] Loaded {len(self.images)} images")
            print(f"[OK] Image size: {self.image_size[0]}x{self.image_size[1]}\n")
        else:
            raise ValueError("No valid images found!")
    
    def detect_pattern(self):
        """Detect calibration pattern in images"""
        print("Detecting calibration pattern in images...")
        print("-" * 70)
        
        # Prepare object points
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        
        if self.pattern_type == "checkerboard":
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
        else:  # circular
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.circle_spacing
        
        successful_images = 0
        failed_images = []
        
        for img_file, img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.pattern_type == "checkerboard":
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
                
                if ret:
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    self.object_points.append(objp)
                    self.image_points.append(corners)
                    successful_images += 1
                    print(f"[OK] {os.path.basename(img_file)}")
                else:
                    failed_images.append(img_file)
                    print(f"[FAIL] {os.path.basename(img_file)} - Pattern not found")
            
            else:  # circular
                # Use SimpleBlobDetector for circular grids
                params = cv2.SimpleBlobDetector_Params()
                params.minArea = 10
                params.maxArea = 10000
                params.filterByCircularity = True
                params.minCircularity = 0.7
                params.filterByConvexity = True
                params.minConvexity = 0.8
                params.filterByInertia = True
                params.minInertiaRatio = 0.5
                
                detector = cv2.SimpleBlobDetector_create(params)
                ret, corners = cv2.findCirclesGrid(gray, self.pattern_size, None,
                                                   flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                                   blobDetector=detector)
                
                if ret:
                    self.object_points.append(objp)
                    self.image_points.append(corners)
                    successful_images += 1
                    print(f"[OK] {os.path.basename(img_file)}")
                else:
                    failed_images.append(img_file)
                    print(f"[FAIL] {os.path.basename(img_file)} - Pattern not found")
        
        print("-" * 70)
        print(f"\nPattern detection complete:")
        print(f"  Successfully detected: {successful_images}/{len(self.images)}")
        
        if failed_images:
            print(f"  Failed images: {len(failed_images)}")
        
        if successful_images < 10:
            print("\n[WARNING] Warning: Less than 10 successful images.")
            print("  Calibration quality may be poor.")
            confirm = input("Continue with calibration? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Calibration cancelled.")
                exit(0)
        
        print()
    
    def calibrate_camera(self):
        """Perform camera calibration"""
        print("Performing camera calibration...")
        print("This may take a moment...\n")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, 
            self.image_points, 
            self.image_size, 
            None, 
            None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed!")
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(self.object_points[i], rvecs[i], 
                                             tvecs[i], mtx, dist)
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.object_points)
        
        print("[OK] Calibration successful!")
        print(f"  Mean reprojection error: {mean_error:.4f} pixels")
        
        if mean_error > 1.0:
            print("  [WARNING] Warning: High reprojection error (>1.0 pixel)")
            print("  Consider recalibrating with better images.\n")
        elif mean_error > 0.5:
            print("  ℹ Note: Moderate reprojection error")
            print("  Calibration is acceptable but could be improved.\n")
        else:
            print("  [OK] Excellent calibration quality!\n")
        
        return mtx, dist, mean_error
    
    def save_calibration(self, mtx, dist, mean_error):
        """Save calibration results to YAML file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create abbreviated pattern name
        pattern_abbr = "cb" if self.pattern_type == "checkerboard" else "cg"
        num_images = len(self.object_points)
        
        # Generate filename with pattern and image count
        filename = f"calib_{pattern_abbr}_{num_images}x_{timestamp}.yaml"
        output_file = Path(__file__).parent.parent / "camera_calibration" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare calibration data
        calibration_data = {
            'calibration_date': datetime.now().isoformat(),
            'camera_model': 'RealSense D400 Series',
            'image_size': {
                'width': int(self.image_size[0]),
                'height': int(self.image_size[1])
            },
            'pattern_type': self.pattern_type,
            'pattern_size': {
                'cols': int(self.pattern_size[0]),
                'rows': int(self.pattern_size[1])
            },
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': mtx.flatten().tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': dist.flatten().tolist()
            },
            'camera_intrinsics': {
                'fx': float(mtx[0, 0]),
                'fy': float(mtx[1, 1]),
                'cx': float(mtx[0, 2]),
                'cy': float(mtx[1, 2])
            },
            'distortion_model': 'plumb_bob',
            'distortion_parameters': {
                'k1': float(dist[0, 0]),
                'k2': float(dist[0, 1]),
                'p1': float(dist[0, 2]),
                'p2': float(dist[0, 3]),
                'k3': float(dist[0, 4])
            },
            'calibration_quality': {
                'mean_reprojection_error': float(mean_error),
                'num_images_used': len(self.object_points),
                'total_images': len(self.images)
            }
        }
        
        # Add pattern-specific parameters
        if self.pattern_type == "checkerboard":
            calibration_data['pattern_parameters'] = {
                'square_size_mm': float(self.square_size)
            }
        else:
            calibration_data['pattern_parameters'] = {
                'circle_diameter_mm': float(self.circle_diameter),
                'circle_spacing_mm': float(self.circle_spacing)
            }
        
        # Save to YAML
        with open(output_file, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"[OK] Calibration data saved to:")
        print(f"  {output_file}\n")
        
        # Print summary
        print("="*70)
        print("  Calibration Summary")
        print("="*70)
        print(f"Camera Intrinsics:")
        print(f"  fx: {mtx[0, 0]:.2f}")
        print(f"  fy: {mtx[1, 1]:.2f}")
        print(f"  cx: {mtx[0, 2]:.2f}")
        print(f"  cy: {mtx[1, 2]:.2f}")
        print(f"\nDistortion Coefficients:")
        print(f"  k1: {dist[0, 0]:.6f}")
        print(f"  k2: {dist[0, 1]:.6f}")
        print(f"  p1: {dist[0, 2]:.6f}")
        print(f"  p2: {dist[0, 3]:.6f}")
        print(f"  k3: {dist[0, 4]:.6f}")
        print(f"\nCalibration Quality:")
        print(f"  Mean Error: {mean_error:.4f} pixels")
        print(f"  Images Used: {len(self.object_points)}/{len(self.images)}")
        print("="*70 + "\n")
    
    def run(self):
        """Run the calibration process"""
        try:
            self.print_header()
            self.select_pattern_type()
            self.get_pattern_parameters()
            self.select_image_source()
            self.load_images()
            self.detect_pattern()
            mtx, dist, mean_error = self.calibrate_camera()
            self.save_calibration(mtx, dist, mean_error)
            
            print("Calibration process completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nCalibration cancelled by user.")
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    calibration = CameraCalibration()
    calibration.run()


if __name__ == "__main__":
    main()

