import torch
import pynvml
from contextlib import contextmanager
import torch
from torchprofile import profile_macs
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
import colorama
from colorama import Fore, Style



@contextmanager
def track_gpu_memory():
    """
    Context manager to track GPU memory usage during inference
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        max_memory = torch.cuda.max_memory_allocated() / 1e6
        current_memory = torch.cuda.memory_allocated() / 1e6
        # print(f"Max GPU memory allocated: {max_memory:.2f} MB")
        # print(f"Current GPU memory allocated: {current_memory:.2f} MB")
        
    # Store the memory usage in the context manager object
    track_gpu_memory.max_memory = max_memory
    track_gpu_memory.current_memory = current_memory

def detailed_memory_info():
    """
    get detailed memory information of the GPU
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU memory: {info.total / 1e6:.2f} MB")
    print(f"Free GPU memory: {info.free / 1e6:.2f} MB")
    print(f"Used GPU memory: {info.used / 1e6:.2f} MB")
    pynvml.nvmlShutdown()



def get_model_macs(model, inputs) -> int:
    """
    returns the number of multiply-accumulate operations for the given model and inputs
    """
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_layer_sparsity(model: nn.Module) -> dict:
    print('sparsity per layer: ')
    for name, param in model.named_parameters():
        if param.dim() > 1:
            print(f'{name}: {get_sparsity(param):.3f}')


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def plot_num_parameters_distribution(model):
    """
    plot the distribution of the number of parameters per layer
    """
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig('num_parameters_distribution.png', dpi=300, bbox_inches='tight')


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width



def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    """
    plot the distribution of the weights
    """
    nlayers = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nlayers += 1
    fig, axes = plt.subplots((nlayers//2)+1, 2, figsize=(10, 10))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color='blue', alpha=0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def measure_latency_cpu(model, dummy_input, n_warmup=50, n_test=200):
    """
    measure the inference time of the model on cpu
    """
    batch_size = dummy_input.shape[0]
    model.eval()
    model = model.to('cpu')

    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    timings = np.zeros((n_test, 1))
    with torch.inference_mode():
        for i in range(n_test):
            t1 = time.perf_counter()
            _ = model(dummy_input)
            t2 = time.perf_counter()
            timings[i] = t2-t1
        
    # time.perf_counter() returns time in s -> converted to ms (*1000)
    mean_syn = np.sum(timings) / n_test * 1000
    std_syn = np.std(timings) * 1000
    fps = batch_size*1000/mean_syn
    # print(f'Inference time CPU (ms/image):{mean_syn/batch_size:.3f} ms +/- {std_syn/batch_size:.3f} ms')
    # print(f'FPS CPU: {fps:.2f}')
    return mean_syn, std_syn, fps  


@torch.no_grad()
def measure_latency_gpu(model, dummy_input, n_warmup=50, n_test=200):
    """
    measure the inference time of the model on gpu
    """
    assert torch.cuda.is_available(), 'CUDA is not available'
    batch_size = dummy_input.shape[0]
    model.eval()
    model.to('cuda')
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

    timings = np.zeros((n_test, 1))

    # gpu warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # mesure performance
    with torch.inference_mode():
        for rep in range(n_test):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # gpu sync to make sure that the timing is correct
            torch.cuda.synchronize()
            # return time in milliseconds
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / n_test
    std_syn = np.std(timings)
    fps = batch_size*1000/mean_syn
    # print(f'Inference time GPU (ms/image): {mean_syn/batch_size:.3f} ms +/- {std_syn/batch_size:.3f} ms')
    # print(f'FPS GPU: {fps:.2f}')
    return mean_syn, std_syn, fps 




colorama.init(autoreset=True)

def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

@torch.no_grad()
def benchmark(model, dummy_input, n_warmup=50, n_test=200, plot=False):
    batch_size = dummy_input.shape[0]
    dummy_input = dummy_input.to('cpu')
    mean_syn_cpu, std_syn_cpu, fps_cpu = measure_latency_cpu(model, dummy_input, n_warmup, n_test)
    dummy_input = dummy_input.to('cuda')
    with track_gpu_memory():
        mean_syn_gpu, std_syn_gpu, fps_gpu = measure_latency_gpu(model, dummy_input, n_warmup, n_test)
    max_memory_used = track_gpu_memory.max_memory
    current_memory_used = track_gpu_memory.current_memory

    num_parameters = get_num_parameters(model)
    model_size = get_model_size(model)
    num_macs = get_model_macs(model, dummy_input) / batch_size
    
    print(f"{Fore.YELLOW}{Style.BRIGHT}benchmark results")
    print('----------------------------')
    print(f"{Fore.GREEN}Number of parameters: {Fore.WHITE}{format_number(num_parameters)}")
    print(f"{Fore.GREEN}Model Size: {Fore.WHITE}{model_size/MiB:.3f} MiB")
    print(f"{Fore.GREEN}Number of MACs: {Fore.WHITE}{format_number(num_macs)}")

    print(f"\n{Fore.MAGENTA}Memory Usage:")
    print(f"  {Fore.BLUE}Max memory used: {Fore.WHITE}{format_number(max_memory_used)} MB")
    print(f"  {Fore.BLUE}Current memory used: {Fore.WHITE}{format_number(current_memory_used)} MB")

    print(f"\n{Fore.MAGENTA}Performance:")
    print(f"  {Fore.BLUE}CPU: {Fore.WHITE}{fps_cpu:.2f} FPS (±{std_syn_cpu:.2f} ms)")
    print(f"  {Fore.BLUE}GPU: {Fore.WHITE}{fps_gpu:.2f} FPS (±{std_syn_gpu:.2f} ms)")

    if plot:
        print(f"\n{Fore.MAGENTA}Generating plots...")
        plot_num_parameters_distribution(model)
        plot_weight_distribution(model)

    # return the benchmark results as a dictionary
    return {
        'num_parameters': num_parameters,
        'model_size': model_size,
        'num_macs': num_macs,
        'max_memory_used': max_memory_used,
        'current_memory_used': current_memory_used,
        'fps_cpu': fps_cpu,
        'fps_gpu': fps_gpu,
        'std_cpu': std_syn_cpu, 
        'std_gpu': std_syn_gpu,
        'std_syn_cpu': std_syn_cpu,
        'std_syn_gpu': std_syn_gpu
    }
    


