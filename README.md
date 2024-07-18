# PyTorch Model Benchmarking Tool

This tool provides a comprehensive set of utilities for benchmarking PyTorch models, including performance metrics, memory usage, and model statistics.

## Features

- Measure inference latency on both CPU and GPU
- Track GPU memory usage
- Calculate model size and number of parameters
- Compute MACs (Multiply-Accumulate operations)
- Calculate model sparsity
- Generate visualizations of parameter distributions and weight distributions
- Provide formatted output of benchmark results

## Installation

You can install the package using pip:

```bash
pip install pytorch-bench
```

## Example
```python 
import torch 
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_bench import benchmark

# Load model and example input
model = resnet50(weights=ResNet50_Weights.DEFAULT)
example_input = torch.randn(1, 3, 224, 224)

# Run benchmark 
results = benchmark(model, example_input)
```
You can run ```example.py``` to see the output in your terminal and play with the different functions. 


## Advanced Usage 
### Tracking gpu memory for a torch model
```python
from pytorch_bench import track_gpu_memory

with track_gpu_memory():
    # Your GPU operations here
    pass

max_memory = track_gpu_memory.max_memory
current_memory = track_gpu_memory.current_memory
print(f"Max GPU memory used: {max_memory:.2f} MB")
print(f"Current GPU memory used: {current_memory:.2f} MB")
```
### Getting info about GPU memory
```python
from pytorch_bench import detailed_memory_info

detailed_memory_info()
```
### Calculating model sparsity 
```python 
from pytorch_bench import get_model_sparsity, get_layer_sparsity

sparsity = get_model_sparsity(model)
print(f"Model sparsity: {sparsity:.2f}")

get_layer_sparsity(model)
```

### Visualizations

When plot=True is set in the benchmark function, two plots will be generated:

1) *num_parameters_distribution.png*: Bar chart showing the number of parameters in each layer.
2) *weight_distribution.png*: Histograms of weight distributions for each layer.

These plots can provide insights into the model's architecture and weight patterns.


### Notes

- Ensure you have a CUDA-capable GPU for GPU benchmarking.
- The tool uses CUDA events for precise GPU timing.
- Memory usage is tracked using PyNVML.
- MACs calculation requires the [torchprofile](https://github.com/zhijian-liu/torchprofile) package.


## Contributing

This project started as a personal tool to simplify the process of benchmarking models on EdgeAI resources. It's designed to be a lightweight, easy-to-use solution that can be quickly installed and utilized.

While this is primarily a personal project, I'm open to suggestions and improvements. If you have ideas or find any issues, feel free to:

1. Open an issue on the GitHub repository to report bugs or suggest enhancements.
2. Submit pull requests for minor fixes or improvements..

If you find this tool helpful, feel free to star the repository or share it with others who might benefit from it. Thanks for your interest!
