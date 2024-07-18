import torch 
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_bench import benchmark

# Load model and example input
model = resnet50(weights=ResNet50_Weights.DEFAULT)
example_input = torch.randn(1, 3, 224, 224)

# Run benchmark 
results = benchmark(model, example_input)

# Access specific results
print(f"GPU FPS: {results['fps_gpu']:.2f}")
print(f"CPU FPS: {results['fps_cpu']:.2f}")
print(f"Model size: {results['model_size'] / 1024 / 1024:.2f} MB")

