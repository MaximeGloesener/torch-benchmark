import torch
import pynvml
from contextlib import contextmanager
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
from codecarbon import EmissionsTracker
from torchprofile import profile_macs
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================
# Shared Utility Functions
# =========================

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

@contextmanager
def track_gpu_memory():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        max_memory = torch.cuda.max_memory_allocated() / 1e6
        current_memory = torch.cuda.memory_allocated() / 1e6
    track_gpu_memory.max_memory = max_memory
    track_gpu_memory.current_memory = current_memory

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    return get_num_parameters(model, count_nonzero_only) * data_width

def plot_num_parameters_distribution(model):
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

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
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

# =========================
# General Model Benchmark
# =========================

@torch.no_grad()
def measure_latency_cpu(model, dummy_input, n_warmup=50, n_test=300):
    batch_size = dummy_input.shape[0]
    model.eval()
    model = model.to('cpu')
    print("Warming up CPU...")
    for _ in tqdm(range(n_warmup), desc="CPU Warmup"):
        _ = model(dummy_input)
    timings = np.zeros((n_test, 1))
    print("Measuring CPU latency...")
    with torch.inference_mode():
        for i in tqdm(range(n_test), desc="CPU Test"):
            t1 = time.perf_counter()
            _ = model(dummy_input)
            t2 = time.perf_counter()
            timings[i] = t2-t1
    mean_syn = np.sum(timings) / n_test * 1000
    std_syn = np.std(timings) * 1000
    fps = batch_size*1000/mean_syn
    return mean_syn, std_syn, fps

@torch.no_grad()
def measure_latency_gpu(model, dummy_input, n_warmup=20, n_test=300):
    assert torch.cuda.is_available(), 'CUDA is not available'
    batch_size = dummy_input.shape[0]
    model.eval()
    model.to('cuda')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_test, 1))
    print("Warming up GPU...")
    for _ in tqdm(range(n_warmup), desc="GPU Warmup"):
        _ = model(dummy_input)
    print("Measuring GPU latency...")
    with torch.inference_mode():
        for rep in tqdm(range(n_test), desc="GPU Test"):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / n_test
    std_syn = np.std(timings)
    fps = batch_size*1000/mean_syn
    return mean_syn, std_syn, fps

def evaluate_emissions(model, dummy_input, warmup_rounds=20, test_rounds=100):
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    for _ in range(warmup_rounds):
        _ = model(dummy_input)
    with EmissionsTracker(log_level='critical') as tracker:
        for _ in range(test_rounds):
            _ = model(dummy_input)
    total_emissions = tracker.final_emissions
    total_energy_consumed = tracker.final_emissions_data.energy_consumed
    average_emissions_per_inference = total_emissions / test_rounds
    average_energy_per_inference = total_energy_consumed / test_rounds
    return total_emissions, total_energy_consumed, average_emissions_per_inference, average_energy_per_inference

@torch.no_grad()
def benchmark_general_model(model, dummy_input, n_warmup=30, n_test=300, plot=False, gpu_only=False):
    batch_size = dummy_input.shape[0]
    dummy_input = dummy_input.to('cpu')
    num_macs = get_model_macs(model, dummy_input) / batch_size
    if not gpu_only:
        mean_syn_cpu, std_syn_cpu, fps_cpu = measure_latency_cpu(model, dummy_input, n_warmup, n_test)
    else:
        mean_syn_cpu, std_syn_cpu, fps_cpu = None, None, None
    dummy_input = dummy_input.to('cuda')
    with track_gpu_memory():
        mean_syn_gpu, std_syn_gpu, fps_gpu = measure_latency_gpu(model, dummy_input, n_warmup, n_test)
    max_memory_used = track_gpu_memory.max_memory
    current_memory_used = track_gpu_memory.current_memory
    total_emissions, total_energy_consumed, average_emissions_per_inference, average_energy_per_inference = evaluate_emissions(
        model, dummy_input, warmup_rounds=n_warmup, test_rounds=n_test)
    num_parameters = get_num_parameters(model)
    model_size = get_model_size(model)
    print(f"benchmark results")
    print('----------------------------')
    print(f"Number of parameters: {num_parameters/1e6:.3f} M")
    print(f"Model Size: {model_size/MiB:.3f} MiB")
    print(f"Number of MACs: {num_macs/1e9:.3f} B")
    print(f"\nMemory Usage:")
    print(f"  Max memory used: {max_memory_used:.2f} MB")
    print(f"  Current memory used: {current_memory_used:.2f} MB")
    print(f"\nPerformance:")
    if not gpu_only:
        print(f"  CPU: {fps_cpu:.2f} FPS (±{std_syn_cpu:.2f} ms)")
    print(f"  GPU: {fps_gpu:.2f} FPS (±{std_syn_gpu:.2f} ms)")
    print(f"\nEmissions:")
    print(f"  Total emissions: {total_emissions*1e3:.4f} gCO2")
    print(f"  Total energy consumed: {total_energy_consumed*1e3:.4f} Wh")
    print(f"  Average emissions per inference: {average_emissions_per_inference*1e3:.8f} gCO2")
    print(f"  Average energy consumed per inference: {average_energy_per_inference*1e3:.8f} Wh")
    if plot:
        print(f"\nGenerating plots...")
        plot_num_parameters_distribution(model)
        plot_weight_distribution(model)
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
        'total_emissions': total_emissions,
        'total_energy_consumed': total_energy_consumed,
        'average_emissions_per_inference': average_emissions_per_inference,
        'average_energy_per_inference': average_energy_per_inference
    }

# =========================
# YOLO Model Benchmark
# =========================

def benchmark_yolo_model(model, imgsz, num_runs=500):
    input_data = np.random.rand(imgsz, imgsz, 3).astype(np.float32)
    # Warmup runs for gpu
    for _ in range(3):
        for _ in range(20):
            model(input_data, imgsz=imgsz, verbose=False)

    with track_gpu_memory():
        with EmissionsTracker(log_level='critical') as tracker:
            run_times = []
            for _ in tqdm(range(num_runs)):
                results = model(input_data, imgsz=imgsz, verbose=False)
                run_times.append(results[0].speed["inference"])

    run_times = np.array(run_times)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    fps = 1000 / mean_time
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f'FPS GPU: {1000/mean_time:.2f}')
    total_emissions = tracker.final_emissions
    total_energy = tracker.final_emissions_data.energy_consumed
    avg_emissions = total_emissions / num_runs
    avg_energy = total_energy / num_runs
    # Model info
    model.fuse()
    layers, params, gradients, flops = model.info()
    print("benchmark results base model")
    print('----------------------------')
    print('Number of layers: ', layers)
    print(f"Number of parameters: {params} M")
    print(f"Number of FLOPs: {flops} GFLOPS")
    print("\nMemory Usage:")
    print(f"  Max memory used: {track_gpu_memory.max_memory} MB")
    print(f"  Current memory used: {track_gpu_memory.current_memory} MB")
    print("\nPerformance:")
    print(f"  GPU: {fps:.2f} FPS (±{std_time:.2f} ms)")
    print("\nEnergy Consumption:")
    print(f"  Total emissions: {total_emissions*1e3:.8f} gCO2")
    print(f"  Total energy: {total_energy*1e3:.8f} Wh")
    print("\nAverage Energy Consumption Per Inference:")
    print(f"  Average emissions: {avg_emissions*1e3:.8f} gCO2")
    print(f"  Average energy: {avg_energy*1e3:.8f} Wh")
    return {
        "layers": layers,
        "params": params,
        "gradients": gradients,
        "flops": flops,
        "max_memory_used": track_gpu_memory.max_memory,
        "current_memory_used": track_gpu_memory.current_memory,
        "fps_gpu": fps,
        "total_emissions": total_emissions,
        "total_energy": total_energy,
        "avg_emissions": avg_emissions,
        "avg_energy": avg_energy,
    }

# =========================
# Model Type Detection
# =========================

def is_yolo_model(model):
    # Heuristic: YOLO models have .val, .fuse, .info methods
    return all(hasattr(model, attr) for attr in ("val", "fuse", "info"))

# =========================
# Unified Benchmark Function
# =========================

def benchmark(model, input_data, model_type=None, **kwargs):
    """
    Unified benchmarking function for both YOLO and general models.

    Args:
        model: The model to benchmark
        input_data: Either dummy_input (for general) or imgsz (for YOLO)
        model_type: 'yolo', 'general', or None (auto-detect)
        **kwargs: Additional arguments passed to the specific benchmark function

    Returns:
        dict: Benchmark results
    """
    if model_type is None:
        if is_yolo_model(model):
            model_type = 'yolo'
        else:
            model_type = 'general'

    if model_type == 'yolo':
        return benchmark_yolo_model(model, input_data, **kwargs)
    else:
        return benchmark_general_model(model, input_data, **kwargs)
