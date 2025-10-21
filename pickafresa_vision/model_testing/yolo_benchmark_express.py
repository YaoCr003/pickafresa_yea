from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO

model = YOLO('./datasets/best.pt')
# Benchmark on Apple Silicon (MPS)
benchmark(model=model data="data.yaml", imgsz=192, half=False, device='mps')

# Benchmark on CPU
#benchmark(model=model, data="data.yaml", imgsz=192, half=False, device='cpu')