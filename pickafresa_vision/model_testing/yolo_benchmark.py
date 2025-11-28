#!/usr/bin/env python3
"""
YOLO Benchmark Script
Benchmarks YOLO model performance including inference speed, FPS, and resource usage.

@aldrick-t, 2025
"""

import time
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Tuple
import json

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


class YOLOBenchmark:
    def __init__(self, model_path: str, input_size: int = 640, device: str = 'cpu'):
        """
        Initialize YOLO Benchmark
        
        Args:
            model_path: Path to YOLO model weights
            input_size: Input image size (default: 640)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.input_size = input_size
        self.device = device
        
        print(f"Loading model: {model_path}")
        print(f"Device: {device}")
        print(f"Input size: {input_size}")
        
        self.model = YOLO(model_path)
        self.results = {
            'model_path': model_path,
            'input_size': input_size,
            'device': device,
            'inference_times': [],
            'preprocessing_times': [],
            'postprocessing_times': [],
            'total_times': [],
            'detections': []
        }
    
    def generate_test_image(self, width: int = 640, height: int = 640) -> np.ndarray:
        """Generate a random test image"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    def load_test_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Load test images from file paths"""
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                print(f"Loaded: {img_path}")
            else:
                print(f"Warning: Could not load {img_path}")
        return images
    
    def benchmark_single_image(self, image: np.ndarray, warmup: bool = False) -> Dict:
        """
        Benchmark inference on a single image
        
        Args:
            image: Input image (numpy array)
            warmup: If True, don't record results (warmup run)
        
        Returns:
            Dictionary with timing and detection info
        """
        start_total = time.perf_counter()
        
        # Preprocessing
        start_pre = time.perf_counter()
        # YOLO handles preprocessing internally, but we can resize if needed
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            resized = cv2.resize(image, (self.input_size, self.input_size))
        else:
            resized = image
        end_pre = time.perf_counter()
        preprocessing_time = (end_pre - start_pre) * 1000  # ms
        
        # Inference
        start_inf = time.perf_counter()
        results = self.model(resized, device=self.device, verbose=False)
        end_inf = time.perf_counter()
        inference_time = (end_inf - start_inf) * 1000  # ms
        
        # Postprocessing (extracting results)
        start_post = time.perf_counter()
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                detections = len(boxes)
        end_post = time.perf_counter()
        postprocessing_time = (end_post - start_post) * 1000  # ms
        
        end_total = time.perf_counter()
        total_time = (end_total - start_total) * 1000  # ms
        
        result = {
            'preprocessing_time': preprocessing_time,
            'inference_time': inference_time,
            'postprocessing_time': postprocessing_time,
            'total_time': total_time,
            'detections': detections
        }
        
        if not warmup:
            self.results['preprocessing_times'].append(preprocessing_time)
            self.results['inference_times'].append(inference_time)
            self.results['postprocessing_times'].append(postprocessing_time)
            self.results['total_times'].append(total_time)
            self.results['detections'].append(detections)
        
        return result
    
    def run_benchmark(self, num_iterations: int = 100, warmup_iterations: int = 10,
                     test_images: List[np.ndarray] = None) -> Dict:
        """
        Run benchmark
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            test_images: List of test images (if None, generates random images)
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Starting Benchmark")
        print(f"{'='*60}")
        
        # Use provided images or generate random ones
        if test_images is None or len(test_images) == 0:
            print(f"Generating random test images...")
            test_images = [self.generate_test_image(self.input_size, self.input_size)]
        
        # Warmup
        print(f"\nWarmup: {warmup_iterations} iterations")
        warmup_img = test_images[0]
        for i in range(warmup_iterations):
            self.benchmark_single_image(warmup_img, warmup=True)
            if (i + 1) % 5 == 0:
                print(f"  Warmup progress: {i+1}/{warmup_iterations}")
        
        # Benchmark
        print(f"\nBenchmark: {num_iterations} iterations")
        for i in range(num_iterations):
            # Cycle through test images
            img = test_images[i % len(test_images)]
            self.benchmark_single_image(img, warmup=False)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        # Calculate statistics
        self._calculate_statistics()
        
        return self.results
    
    def _calculate_statistics(self):
        """Calculate benchmark statistics"""
        self.results['statistics'] = {
            'preprocessing': self._get_stats(self.results['preprocessing_times']),
            'inference': self._get_stats(self.results['inference_times']),
            'postprocessing': self._get_stats(self.results['postprocessing_times']),
            'total': self._get_stats(self.results['total_times']),
            'fps': {
                'mean': 1000.0 / np.mean(self.results['total_times']),
                'max': 1000.0 / np.min(self.results['total_times']),
                'min': 1000.0 / np.max(self.results['total_times'])
            }
        }
    
    def _get_stats(self, times: List[float]) -> Dict:
        """Calculate statistics for a list of times"""
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99))
        }
    
    def print_results(self):
        """Print benchmark results"""
        stats = self.results['statistics']
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Input Size: {self.input_size}")
        print(f"Iterations: {len(self.results['total_times'])}")
        
        print(f"\n{'-'*60}")
        print(f"Timing Statistics (ms)")
        print(f"{'-'*60}")
        
        categories = ['preprocessing', 'inference', 'postprocessing', 'total']
        headers = ['Stage', 'Mean', 'Std', 'Min', 'Max', 'Median', 'P95', 'P99']
        
        # Print header
        print(f"{headers[0]:<18} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8} {headers[5]:>8} {headers[6]:>8} {headers[7]:>8}")
        print(f"{'-'*60}")
        
        # Print stats for each category
        for cat in categories:
            s = stats[cat]
            print(f"{cat.capitalize():<18} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f} {s['median']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}")
        
        print(f"\n{'-'*60}")
        print(f"FPS Statistics")
        print(f"{'-'*60}")
        fps = stats['fps']
        print(f"Mean FPS:   {fps['mean']:.2f}")
        print(f"Max FPS:    {fps['max']:.2f}")
        print(f"Min FPS:    {fps['min']:.2f}")
        
        print(f"\n{'-'*60}")
        print(f"Detection Statistics")
        print(f"{'-'*60}")
        detections = self.results['detections']
        print(f"Mean detections per frame: {np.mean(detections):.2f}")
        print(f"Total detections: {sum(detections)}")
        print(f"{'='*60}\n")
    
    def save_results(self, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Benchmark')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model weights (e.g., yolo11n.pt)')
    parser.add_argument('--images', type=str, nargs='+',
                       help='Paths to test images (optional, generates random if not provided)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--input-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to run inference on (default: cpu)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results (default: benchmark_results.json)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load test images if provided
    test_images = None
    if args.images:
        benchmark = YOLOBenchmark(args.model, args.input_size, args.device)
        test_images = benchmark.load_test_images(args.images)
        if len(test_images) == 0:
            print("Warning: No valid images loaded, will use random images")
            test_images = None
    else:
        benchmark = YOLOBenchmark(args.model, args.input_size, args.device)
    
    # Run benchmark
    try:
        benchmark.run_benchmark(
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            test_images=test_images
        )
        
        # Print and save results
        benchmark.print_results()
        benchmark.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
