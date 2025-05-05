from ultralytics import YOLO
from pytorch_bench import benchmark

yolo_model = YOLO('yolo11n.pt')
results = benchmark(yolo_model, 640)

print("Benchmark Results:")
print(results)
