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

Ensure you have PyTorch and the following dependencies installed:

```bash
pip install torch pynvml matplotlib numpy colorama torchprofile
```

## Example
```python 
import torch 
from torchvision.models import resnet50, ResNet50_Weights
from main import benchmark

# Load model and example input
model = resnet50(weights=ResNet50_Weights.DEFAULT)
example_input = torch.randn(1, 3, 224, 224)

# Run benchmark 
results = benchmark(model, example_input)
```
You can run ```example.py``` to see the output in your terminal and play with the different functions. 

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
- **Returns:** mean_syn, std_syn, fps

#### `measure_latency_gpu(model, dummy_input, n_warmup=50, n_test=200)`

Measures the inference time of the model on GPU.

- **Parameters:** Same as `benchmark`
- **Returns:** mean_syn, std_syn, fps

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
