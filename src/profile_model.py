import torch
import deepxde as dde
from rich.console import Console
from rich.table import Table
import os
import sys

# Add the current directory to sys.path to allow importing from model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_pinn_model

def get_model_stats():
    # Build the small model
    model, _ = build_pinn_model(large=False)
    
    # Extract the PyTorch network inside DeepXDE FNN
    # Note: deepxde.nn.FNN creates a PyTorch module when using PyTorch backend
    net = model.net
    
    # Calculate parameters
    total_params = sum(p.numel() for p in net.parameters())
    
    # Calculate bytes for weights (assuming float32 which is 4 bytes)
    param_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    
    # Manual FLOPs calculation for [3, 32, 32, 5] FNN with Swish/SiLU activations
    layers = [3, 32, 32, 5]
    total_flops = 0
    for i in range(len(layers) - 1):
        in_features = layers[i]
        out_features = layers[i+1]
        
        # Multiply-Accumulate operations (MACs) per dense layer = in_features * out_features
        # 1 MAC = 2 FLOPs (one multiply, one add)
        flops_layer = 2 * in_features * out_features
        total_flops += flops_layer
        
        # Bias additions
        total_flops += out_features
        
        # Activation FLOPs
        # SiLU / Swish: x * sigmoid(x) = x / (1 + exp(-x))
        # Operations: neg(1), exp(1), add(1), div(1), mul(1) = ~5 ops per neuron
        if i < len(layers) - 2: # No activation on the final output layer
            total_flops += 5 * out_features

    return total_params, param_bytes, total_flops

def display_results(total_params, param_bytes, total_flops):
    console = Console()
    
    console.print(f"\n[bold green]IonPINN Small Model Profiling[/bold green]")
    console.print(f"Architecture: FNN [3 -> 32 -> 32 -> 5]")
    console.print(f"Total Parameters: [bold]{total_params}[/bold]")
    
    # MegaFLOPs (MFLOPs) = FLOPs / 1e6
    console.print(f"Total Compute: [bold]{total_flops}[/bold] FLOPs ({total_flops / 1e6:.6f} MegaFLOPs)")
    
    # Memory estimation
    kb = param_bytes / 1024
    console.print(f"\n[bold blue]Memory Estimation (Inference):[/bold blue]")
    console.print(f"Weights Memory (Float32): {kb:.2f} KB")
    
    # Estimate activation and buffer memory for a batch size of 1
    # 3 (input) + 32 (h1) + 32 (h2) + 5 (output) = 72 floats = 288 bytes = ~0.28 KB
    activation_kb = (72 * 4) / 1024
    console.print(f"Activation Memory (Float32): {activation_kb:.2f} KB")
    
    total_mem = kb + activation_kb
    console.print(f"Estimated Total SRAM Required: [bold]{total_mem:.2f} KB[/bold]")
    
    if total_mem < 200:
        console.print("[green]Status: Readily fits in on-chip SRAM of typical embedded microcontrollers.[/green]")
    else:
        console.print("[red]Status: Might require external PSRAM.[/red]")

    # Performance table
    table = Table(title="Expected Performance on Embedded Processors")
    table.add_column("Processor", style="cyan", no_wrap=True)
    table.add_column("Clock Speed", style="magenta")
    table.add_column("Est. FPU Utilization", style="yellow")
    table.add_column("Est. Latency / Inference", style="green")
    table.add_column("Max Inferences / Sec", justify="right")

    # Common embedded processors for control tasks
    processors = [
        {"name": "ESP32 (Espressif)", "clock_mhz": 240, "utilization": 0.5},
        {"name": "STM32F405 (ARM Cortex-M4F)", "clock_mhz": 168, "utilization": 0.4},
        {"name": "STM32H7 (ARM Cortex-M7F)", "clock_mhz": 480, "utilization": 0.6},
        {"name": "TI C2000 (TMS320F28379D)", "clock_mhz": 200, "utilization": 0.5},
        {"name": "RP2040 (Cortex-M0+, soft float)", "clock_mhz": 133, "utilization": 0.05}, # Software floating point is much slower
    ]

    for proc in processors:
        clock_hz = proc["clock_mhz"] * 1e6
        # Effective FLOPs per second = Clock Speed * Utilization (assuming roughly 1 FLOP per cycle when FPU is utilized)
        effective_flops_per_sec = clock_hz * proc["utilization"]
        
        latency_sec = total_flops / effective_flops_per_sec
        latency_us = latency_sec * 1e6
        
        inf_per_sec = 1 / latency_sec
        
        table.add_row(
            proc["name"],
            f"{proc['clock_mhz']} MHz",
            f"{proc['utilization']*100:.0f}%",
            f"{latency_us:.2f} us",
            f"{inf_per_sec:,.0f}"
        )

    console.print("\n")
    console.print(table)
    console.print("\n[dim]Note: Latency estimates assume ideal floating-point unit (FPU) usage and no memory bottleneck. Real-world performance may vary depending on firmware optimization (e.g. using CMSIS-NN or ESP-DSP) and quantization.[/dim]\n")

if __name__ == "__main__":
    try:
        import rich
    except ImportError:
        print("Please install 'rich' package to display the table: pip install rich")
        sys.exit(1)
        
    params, p_bytes, flops = get_model_stats()
    display_results(params, p_bytes, flops)
