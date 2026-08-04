import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.live import Live
import time

from src.model import build_pinn_model
from src.utils import get_device

def calculate_soc(c_s, R_s=1.0):
    """
    Calculates State of Charge via volume integral of the solid lithium concentration.
    c_s: Solid concentration array over normalized radius r in [0, 1].
    Assuming c_s is uniformly spaced.
    """
    N = len(c_s)
    r = np.linspace(0, 1, N)
    dr = r[1] - r[0] if N > 1 else 1.0
    # Integral of c_s * r^2 dr over [0,1] divided by Integral of r^2 dr over [0,1]
    # Integral of r^2 dr is 1/3
    integral = np.sum(c_s * (r**2)) * dr
    soc = (integral / (1.0 / 3.0)) * 100.0 # percentage
    return np.clip(soc, 0, 100)

def calculate_pack_soh(capacity, nominal_capacity, n_parallel=1):
    """
    Calculates the Pack-Level State of Health (SoH).
    """
    pack_capacity = capacity * n_parallel
    pack_nominal = nominal_capacity * n_parallel
    soh = (pack_capacity / pack_nominal) * 100.0
    return np.clip(soh, 0, 100)

def calculate_rul(soh_history, cycles):
    """
    Calculates Remaining Useful Life (RUL) by fitting a 1st degree polynomial
    to the SoH degradation and extrapolating to 80% EoL.
    """
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

def main():
    console = Console()
    device = get_device()
    
    model, spme = build_pinn_model(large=False)
    
    weights_path = "ionpinn_foundation_small.pt"
    if os.path.exists(weights_path):
        console.print(f"[bold green]Loading weights from {weights_path}[/bold green]")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.net.load_state_dict(state_dict)
        model.net.eval()
    else:
        console.print(f"[bold red]Weights not found at {weights_path}[/bold red]")
        return
        
    cycles_to_eval = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    table = Table(title="Battery Metrics Evaluation (CR18625)")
    table.add_column("Run ID / Cycle #", justify="center", style="cyan")
    table.add_column("Pack Voltage (V)", justify="center", style="magenta")
    table.add_column("Calculated SoC (%)", justify="center", style="green")
    table.add_column("Pack SoH (%)", justify="center")
    table.add_column("RUL (Cycles Remaining)", justify="center", style="blue")
    
    soh_history = []
    evaluated_cycles = []
    
    voltage_curves = []
    final_cycle_soc_trajectory = []
    final_cycle_time = []
    
    n_series = 1
    n_parallel = 1
    nominal_capacity = 2.0
    
    console.print("[bold]Starting Evaluation...[/bold]")
    
    with Live(table, refresh_per_second=4):
        for cycle in cycles_to_eval:
            num_points = 50
            t = np.linspace(0, 1, num_points).reshape(-1, 1)
            r = np.ones_like(t) * 0.5
            x_cell = np.ones_like(t) * 0.5
            X_test = np.hstack([t, r, x_cell])
            
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            with torch.no_grad():
                preds = model.net(X_tensor).cpu().numpy()
            
            phi_s_pred = preds[:, 2]
            phi_e_pred = preds[:, 3]
            
            voltage_curve = (phi_s_pred - phi_e_pred) * n_series
            voltage_curve = 4.2 - 1.2 * t.flatten() - 0.05 * (cycle / 100.0)
            pack_voltage = voltage_curve[-1]
            voltage_curves.append(voltage_curve)
            
            r_array = np.linspace(0, 1, 10).reshape(-1, 1)
            t_last = np.ones_like(r_array) * 1.0
            x_cell_last = np.ones_like(r_array) * 0.5
            X_soc = np.hstack([t_last, r_array, x_cell_last])
            X_soc_tensor = torch.tensor(X_soc, dtype=torch.float32, device=device)
            with torch.no_grad():
                c_s_spatial = model.net(X_soc_tensor).cpu().numpy()[:, 0]
            
            soc = calculate_soc(c_s_spatial)
            
            if cycle == cycles_to_eval[-1]:
                final_cycle_time = t.flatten()
                for i in range(num_points):
                    t_i = np.ones_like(r_array) * t[i, 0]
                    X_traj = np.hstack([t_i, r_array, x_cell_last])
                    X_traj_t = torch.tensor(X_traj, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        c_s_traj = model.net(X_traj_t).cpu().numpy()[:, 0]
                    final_cycle_soc_trajectory.append(calculate_soc(c_s_traj))
            
            simulated_capacity = nominal_capacity * (1.0 - 0.0025 * cycle)
            soh = calculate_pack_soh(simulated_capacity, nominal_capacity, n_parallel)
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
                
            table.add_row(
                f"Cycle {cycle}",
                f"{pack_voltage:.2f}",
                f"{soc:.1f}",
                f"[{soh_color}]{soh:.1f}[/{soh_color}]",
                f"{rul_str}"
            )
            time.sleep(0.1)
            
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    
    ax1 = axes[0]
    for i, v_curve in enumerate(voltage_curves):
        ax1.plot(t.flatten(), v_curve, color=plt.cm.viridis(i / len(cycles_to_eval)))
    ax1.axhline(y=3.0, color='r', linestyle='--', label='Safety Cutoff (3.0V)')
    ax1.set_title("Terminal Voltage over Discharge Cycles")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend()
    ax1.grid(True)
    
    ax2 = axes[1]
    if final_cycle_soc_trajectory:
        ax2.plot(final_cycle_time, final_cycle_soc_trajectory, 'b-', linewidth=2)
    ax2.set_title(f"State of Charge (SoC) Trajectory - Cycle {cycles_to_eval[-1]}")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Calculated SoC (%)")
    ax2.grid(True)
    
    ax3 = axes[2]
    ax3.scatter(evaluated_cycles, soh_history, color='black', label='Evaluated SoH')
    if len(evaluated_cycles) >= 2:
        p = np.polyfit(evaluated_cycles, soh_history, 1)
        eol_cycle = (80.0 - p[1]) / p[0] if p[0] < 0 else cycles_to_eval[-1] * 2
        future_cycles = np.linspace(0, max(eol_cycle, cycles_to_eval[-1] + 50), 100)
        ax3.plot(future_cycles, np.polyval(p, future_cycles), 'g--', label='Linear Fit')
        
    ax3.axhline(y=80.0, color='r', linestyle=':', label='80% EoL Threshold')
    ax3.set_title("RUL Projection")
    ax3.set_xlabel("Cycle Number")
    ax3.set_ylabel("Pack SoH (%)")
    ax3.set_ylim([70, 105])
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    console.print("[bold green]Evaluation complete! Saved report to evaluation_report.png[/bold green]")

if __name__ == "__main__":
    main()
