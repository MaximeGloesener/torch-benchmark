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
## API Reference

### Main Function

#### `benchmark(model, dummy_input, n_warmup=50, n_test=200, plot=False)`

Runs a comprehensive benchmark on the given model.

- **Parameters:**
  - `model`: PyTorch model to benchmark
  - `dummy_input`: A tensor matching the input shape expected by the model
  - `n_warmup`: Number of warm-up iterations (default: 50)
  - `n_test`: Number of test iterations (default: 200)
  - `plot`: If True, generates plots for parameter and weight distributions (default: False)
- **Returns:** A dictionary containing benchmark results

### Utility Functions

#### `measure_latency_cpu(model, dummy_input, n_warmup=50, n_test=200)`

Measures the inference time of the model on CPU.

- **Parameters:** Same as `benchmark`
- **Returns:** mean_syn (in ms), std_syn (in ms), fps

#### `measure_latency_gpu(model, dummy_input, n_warmup=50, n_test=200)`

Measures the inference time of the model on GPU.

- **Parameters:** Same as `benchmark`
- **Returns:** mean_syn (in ms), std_syn (in ms), fps

#### `get_model_macs(model, inputs) -> int`

Returns the number of multiply-accumulate operations for the given model and inputs.

- **Parameters:**
  - `model`: PyTorch model
  - `inputs`: Input tensor
- **Returns:** Number of MACs

#### `get_sparsity(tensor: torch.Tensor) -> float`

Calculates the sparsity of the given tensor.

- **Parameters:** `tensor`: PyTorch tensor
- **Returns:** Sparsity value (float)

#### `get_layer_sparsity(model: nn.Module)`

Prints the sparsity for each layer in the model.

- **Parameters:** `model`: PyTorch model

#### `get_model_sparsity(model: nn.Module) -> float`

Calculates the overall sparsity of the given model.

- **Parameters:** `model`: PyTorch model
- **Returns:** Model sparsity (float)

#### `get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int`

Calculates the total number of parameters in the model.

- **Parameters:**
  - `model`: PyTorch model
  - `count_nonzero_only`: If True, only counts non-zero parameters (default: False)
- **Returns:** Number of parameters

#### `get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int`

Calculates the model size in bits.

- **Parameters:**
  - `model`: PyTorch model
  - `data_width`: Number of bits per element (default: 32)
  - `count_nonzero_only`: If True, only counts non-zero parameters (default: False)
- **Returns:** Model size in bits

#### `plot_num_parameters_distribution(model)`

Plots the distribution of the number of parameters per layer.

- **Parameters:** `model`: PyTorch model

#### `plot_weight_distribution(model, bins=256, count_nonzero_only=False)`

Plots the distribution of the weights for each layer.

- **Parameters:**
  - `model`: PyTorch model
  - `bins`: Number of histogram bins (default: 256)
  - `count_nonzero_only`: If True, only plots non-zero weights (default: False)

### Context Managers

#### `track_gpu_memory()`

Context manager to track GPU memory usage during inference.

- **Usage:**
  ```python
  with track_gpu_memory():
      # Your GPU operations here
  max_memory = track_gpu_memory.max_memory
  current_memory = track_gpu_memory.current_memory
  ```
