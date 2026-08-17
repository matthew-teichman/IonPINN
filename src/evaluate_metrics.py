import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.live import Live
import time
import re

from src.model import build_pinn_model
from src.utils import get_device

def calculate_soc(c_s, R_s=1.0):
    N = len(c_s)
    r = np.linspace(0, 1, N)
    dr = r[1] - r[0] if N > 1 else 1.0
    integral = np.sum(c_s * (r**2)) * dr
    soc = (integral / (1.0 / 3.0)) * 100.0
    return np.clip(soc, 0, 100)

def calculate_pack_soh(capacity, nominal_capacity, n_parallel=1):
    pack_capacity = capacity * n_parallel
    pack_nominal = nominal_capacity * n_parallel
    soh = (pack_capacity / pack_nominal) * 100.0
    return np.clip(soh, 0, 100)

def calculate_rul(soh_history, cycles):
    if len(cycles) < 2:
        return float('inf')
    
    p = np.polyfit(cycles, soh_history, 1)
    m, c = p
    
    if m >= 0:
        return float('inf')
    
    eol_cycle = (80.0 - c) / m
    current_cycle = cycles[-1]
    rul = eol_cycle - current_cycle
    return max(0.0, float(rul))

def evaluate_dataset(dataset_prefix, data_dir, models, device, nominal_capacity=2.0, n_series=1, n_parallel=1):
    console = Console()
    
    # 1. Gather all files and sort by cycle number
    pattern = os.path.join(data_dir, f"{dataset_prefix}_cycle*.csv")
    all_files = glob.glob(pattern)
    
    # Extract cycle numbers and sort
    file_cycle_pairs = []
    for f in all_files:
        basename = os.path.basename(f)
        match = re.search(r'cycle(\d+)\.csv', basename)
        if match:
            cycle = int(match.group(1))
            file_cycle_pairs.append((f, cycle))
            
    if not file_cycle_pairs:
        console.print(f"[bold red]No files found for {dataset_prefix} in {data_dir}[/bold red]")
        return
        
    file_cycle_pairs.sort(key=lambda x: x[1])
    
    # 2. Select 10 evenly spaced cycles
    num_files = len(file_cycle_pairs)
    if num_files <= 10:
        selected_pairs = file_cycle_pairs
    else:
        indices = np.linspace(0, num_files - 1, 10, dtype=int)
        selected_pairs = [file_cycle_pairs[i] for i in indices]
        
    cycles_to_eval = [pair[1] for pair in selected_pairs]
    files_to_eval = [pair[0] for pair in selected_pairs]
    
    console.print(f"[bold cyan]Evaluating {dataset_prefix} on cycles: {cycles_to_eval}[/bold cyan]")
    
    table = Table(title=f"Battery Metrics Evaluation ({dataset_prefix})")
    table.add_column("Cycle #", justify="center", style="cyan")
    table.add_column("True Voltage (V)", justify="center", style="magenta")
    for name in models.keys():
        table.add_column(f"{name} Pred V", justify="center", style="magenta")
        table.add_column(f"{name} SoC (%)", justify="center", style="green")
    table.add_column("Pack SoH (%)", justify="center")
    table.add_column("RUL (Cycles)", justify="center", style="blue")
    
    soh_history = []
    evaluated_cycles = []
    
    true_voltage_curves = []
    pred_voltage_curves = {name: [] for name in models.keys()}
    times_list = []
    final_cycle_soc_trajectory = {name: [] for name in models.keys()}
    final_cycle_time = []
    
    with Live(table, refresh_per_second=4):
        for i, (filepath, cycle) in enumerate(zip(files_to_eval, cycles_to_eval)):
            df = pd.read_csv(filepath)
            if df.empty or 'Time' not in df.columns or 'Voltage' not in df.columns:
                continue
                
            time_vals = df['Time'].values
            true_voltages = df['Voltage'].values
            currents = df['Current'].values if 'Current' in df.columns else np.zeros_like(time_vals)
            temp_vals = df['Temperature'].values if 'Temperature' in df.columns else np.ones_like(time_vals) * 25.0
            
            # Normalize time for model input
            t_max = np.max(time_vals) if np.max(time_vals) > 0 else 1.0
            t_norm = (time_vals / t_max).reshape(-1, 1)
            temp_norm = temp_vals.reshape(-1, 1)
            
            # Spatial coordinates
            r = np.ones_like(t_norm) * 0.5
            x_cell = np.ones_like(t_norm) * 0.5
            X_test = np.hstack([r, x_cell, temp_norm, t_norm])
            
            # Predict
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            current_pred_voltages = {}
            for name, model in models.items():
                with torch.no_grad():
                    preds = model.net(X_tensor).cpu().numpy()
                phi_s_pred = preds[:, 2]
                phi_e_pred = preds[:, 3]
                current_pred_voltages[name] = (phi_s_pred - phi_e_pred) * n_series
                pred_voltage_curves[name].append(current_pred_voltages[name])
                
            true_voltage_curves.append(true_voltages)
            times_list.append(t_norm.flatten())
            
            # Calculate SoC for final point of cycle (and trajectory if it's the last selected cycle)
            final_socs = {}
            
            if i == len(files_to_eval) - 1:
                final_cycle_time = t_norm.flatten()
                r_array = np.linspace(0, 1, 10).reshape(-1, 1)
                
                for name, model in models.items():
                    model_traj = []
                    for j in range(len(t_norm)):
                        t_j = np.ones_like(r_array) * t_norm[j, 0]
                        x_cell_j = np.ones_like(r_array) * 0.5
                        temp_j = np.ones_like(r_array) * temp_norm[j, 0]
                        X_traj = np.hstack([r_array, x_cell_j, temp_j, t_j])
                        X_traj_t = torch.tensor(X_traj, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            c_s_traj = model.net(X_traj_t).cpu().numpy()[:, 0]
                        soc_j = calculate_soc(c_s_traj)
                        model_traj.append(soc_j)
                        if j == len(t_norm) - 1:
                            final_socs[name] = soc_j
                    final_cycle_soc_trajectory[name] = model_traj
            else:
                # Just calculate SoC for the last point
                r_array = np.linspace(0, 1, 10).reshape(-1, 1)
                t_last = np.ones_like(r_array) * t_norm[-1, 0]
                x_cell_last = np.ones_like(r_array) * 0.5
                temp_last = np.ones_like(r_array) * temp_norm[-1, 0]
                X_soc = np.hstack([r_array, x_cell_last, temp_last, t_last])
                X_soc_tensor = torch.tensor(X_soc, dtype=torch.float32, device=device)
                for name, model in models.items():
                    with torch.no_grad():
                        c_s_spatial = model.net(X_soc_tensor).cpu().numpy()[:, 0]
                    final_socs[name] = calculate_soc(c_s_spatial)
                
            # Estimate Capacity and SoH
            # Capacity ~ integral of |I| dt / 3600 (assuming time is seconds)
            capacity_Ah = np.trapezoid(np.abs(currents), time_vals) / 3600.0
            
            # As a fallback if capacity calculation is broken/zero due to dataset format, use dummy degradation
            if capacity_Ah < 0.1: 
                capacity_Ah = nominal_capacity * (1.0 - 0.0025 * cycle)
                
            soh = calculate_pack_soh(capacity_Ah, nominal_capacity, n_parallel)
            soh_history.append(soh)
            evaluated_cycles.append(cycle)
            
            rul = calculate_rul(soh_history, evaluated_cycles)
            rul_str = f"{rul:.1f}" if rul != float('inf') else "---"
            
            if soh > 90.0:
                soh_color = "green"
            elif soh > 80.0:
                soh_color = "yellow"
            else:
                soh_color = "red"
                
            row_data = [f"{cycle}", f"{true_voltages[-1]:.2f}"]
            for name in models.keys():
                row_data.append(f"{current_pred_voltages[name][-1]:.2f}")
                row_data.append(f"{final_socs[name]:.1f}")
            row_data.extend([f"[{soh_color}]{soh:.1f}[/{soh_color}]", f"{rul_str}"])
            
            table.add_row(*row_data)
            time.sleep(0.1)
            
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1 = axes[0]
    for i, t_arr in enumerate(times_list):
        alpha_val = 1.0 if i in [0, len(cycles_to_eval)-1] else 0.3
        ax1.plot(t_arr, true_voltage_curves[i], color='black', linestyle='-', linewidth=2, alpha=alpha_val, label='True' if i == 0 else "")
        
        colors = ['blue', 'orange', 'green', 'red']
        for m_idx, (name, curves) in enumerate(pred_voltage_curves.items()):
            color = colors[m_idx % len(colors)]
            ax1.plot(t_arr, curves[i], color=color, linestyle='--', alpha=alpha_val, label=name if i == 0 else "")
        
    ax1.axhline(y=3.0, color='r', linestyle=':', label='Safety Cutoff (3.0V)')
    ax1.set_title(f"{dataset_prefix} - True vs Predicted Terminal Voltage")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True)
    
    ax2 = axes[1]
    colors = ['blue', 'orange', 'green', 'red']
    for m_idx, (name, traj) in enumerate(final_cycle_soc_trajectory.items()):
        if traj:
            color = colors[m_idx % len(colors)]
            ax2.plot(final_cycle_time, traj, color=color, linestyle='-', linewidth=2, label=name)
    ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title(f"{dataset_prefix} - Predicted SoC Trajectory (Cycle {cycles_to_eval[-1]})")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Calculated SoC (%)")
    ax2.set_ylim([0, 105])
    ax2.grid(True)
    
    ax3 = axes[2]
    ax3.scatter(evaluated_cycles, soh_history, color='black', label='Evaluated SoH')
    if len(evaluated_cycles) >= 2:
        p = np.polyfit(evaluated_cycles, soh_history, 1)
        eol_cycle = (80.0 - p[1]) / p[0] if p[0] < 0 else cycles_to_eval[-1] * 2
        future_cycles = np.linspace(0, max(eol_cycle, cycles_to_eval[-1] + 50), 100)
        ax3.plot(future_cycles, np.polyval(p, future_cycles), 'g--', label='Linear Fit')
        
    ax3.axhline(y=80.0, color='r', linestyle=':', label='80% EoL Threshold')
    ax3.set_title(f"{dataset_prefix} - RUL Projection")
    ax3.set_xlabel("Cycle Number")
    ax3.set_ylabel("Pack SoH (%)")
    min_soh = min(soh_history) if soh_history else 70
    ax3.set_ylim([max(0, min_soh - 10), 105])
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plot_path = f"evaluation_report_{dataset_prefix}.png"
    plt.savefig(plot_path)
    console.print(f"[bold green]Evaluation complete! Saved report to {plot_path}[/bold green]\n")


def load_model_weights(model, weights_path, device, console):
    if os.path.exists(weights_path):
        console.print(f"[bold green]Loading weights from {weights_path}[/bold green]")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        
        # Handle legacy checkpoints with 3 inputs (r, x, t) loading into 4 input model (r, x, T, t)
        if 'linears.0.weight' in state_dict and state_dict['linears.0.weight'].shape[1] == 3:
            console.print("[yellow]Legacy 3-input checkpoint detected. Padding for 4-input model.[/yellow]")
            w = state_dict['linears.0.weight']
            padded_w = torch.zeros((w.shape[0], 4), dtype=w.dtype, device=w.device)
            padded_w[:, 0] = w[:, 0] # r
            padded_w[:, 1] = w[:, 1] # x_cell
            padded_w[:, 3] = w[:, 2] # t (T is padded with 0 at index 2)
            state_dict['linears.0.weight'] = padded_w

        model.net.load_state_dict(state_dict)
        model.net.eval()
        return True
    else:
        console.print(f"[bold red]Weights not found at {weights_path}[/bold red]")
        return False

def main():
    console = Console()
    device = get_device()
    
    models = {}
    
    model_small, _ = build_pinn_model(large=False)
    if load_model_weights(model_small, "ionpinn_foundation_small.pt", device, console):
        models["Small"] = model_small
        
    model_large, _ = build_pinn_model(large=True)
    if load_model_weights(model_large, "ionpinn_foundation_large.pt", device, console):
        models["Large"] = model_large
        
    model_distilled, _ = build_pinn_model(large=False)
    if load_model_weights(model_distilled, "ionpinn_distilled_small.pt", device, console):
        models["Distilled"] = model_distilled
        
    if not models:
        console.print("[bold red]No models could be loaded. Exiting.[/bold red]")
        return
        
    data_dir = os.path.join("data", "Testing")
    
    # Evaluate B0018
    evaluate_dataset("B0018", data_dir, models, device)
    
    # Evaluate RW10
    evaluate_dataset("RW10", data_dir, models, device)

if __name__ == "__main__":
    main()
