import torch
from pytorch_bench import benchmark

# Load model and example input
model = resnet50(weights="DEFAULT")
example_input = torch.randn(1, 3, 224, 224)

# Run benchmark
results = benchmark(model, example_input)

print("Benchmark Results:")
print(results)