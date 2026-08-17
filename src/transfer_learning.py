import os
import deepxde as dde
dde.config.set_default_float("float64")
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_device
from src.model import build_pinn_model
from src.data_loader import get_pinn_training_data
from src.train import suppress_builtin_print, RichLossCallback

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class DummyConsole:
        def rule(self, text): print(text)
        def print(self, text): print(text)
        def status(self, text, spinner="dots"):
            from contextlib import contextmanager
            @contextmanager
            def dummy(): yield
            print(text)
            return dummy()
    console = DummyConsole()

def transfer_learning():
    console.rule("[bold blue]Knowledge Distillation (Transfer Learning)[/bold blue]")
    
    device = get_device()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_train, Y_train = get_pinn_training_data(data_dir=data_dir)
    
    # 1. Load the Teacher Model (Large)
    console.print("[bold green]Step 1: Loading Pre-trained Teacher Model...[/bold green]")
    teacher_model, spme_large = build_pinn_model(X_train, Y_train, large=True)
    teacher_path = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation_large.pt")
    if not os.path.exists(teacher_path):
        console.print(f"[bold red]Error: Teacher weights not found at {teacher_path}. Please train the large model first.[/bold red]")
        return
        
    teacher_model.net.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher_model.net.eval()
    
    # 2. Generate Synthetic Dataset
    console.print("[bold green]Step 2: Generating Synthetic Distillation Dataset...[/bold green]")
    # Sample 20000 points from the domain [r, x_cell, T, t]
    X_kd = teacher_model.data.geom.random_points(20000)
    
    # Predict the states using the Teacher
    teacher_model.compile("adam", lr=1e-3)
    Y_teacher = teacher_model.predict(X_kd)
    
    # 3. Setup Distillation Boundary Conditions
    console.print("[bold green]Step 3: Configuring Student Model with KD Constraints...[/bold green]")
    extra_bcs = []
    # State components: 0=c_s, 1=c_e, 2=phi_s, 3=phi_e, 4=j_Li
    for i in range(5):
        # We use PointSetBC to force the student to match the teacher's output component i
        bc_kd = dde.icbc.PointSetBC(X_kd, Y_teacher[:, i:i+1], component=i)
        extra_bcs.append(bc_kd)
        
    # Build the Student Model (Small) passing the extra KD boundary conditions
    student_model, spme_small = build_pinn_model(X_train, Y_train, large=False, extra_bcs=extra_bcs)
    
    # 4. Train the Student
    console.print("[bold green]Step 4: Training Student Model (Distillation Phase)...[/bold green]")
    
    # Dynamically assign weights: 3 PDEs + BCs
    loss_weights = [1.0, 1.0, 1e-4] # PDEs
    for bc in student_model.data.bcs:
        if "PointSetBC" in str(type(bc)):
            loss_weights.append(100.0) # KD Data
        elif "OperatorBC" in str(type(bc)) or "PointSetOperatorBC" in str(type(bc)):
            loss_weights.append(100.0) # Voltage Data
        else:
            loss_weights.append(50.0) # BC_Left

    student_model.compile("adam", lr=1e-3, decay=("step", 2000, 0.4), loss_weights=loss_weights)
    
    cb = RichLossCallback(display_every=100)
    resampler = dde.callbacks.PDEPointResampler(period=2000)
    
    with console.status("[bold green]Distilling (Adam)...[/bold green]", spinner="dots"):
        with suppress_builtin_print():
            student_model.train(iterations=20000, display_every=100, callbacks=[cb, resampler])
            
    dde.optimizers.set_LBFGS_options(ftol=1e-12, gtol=1e-12, maxiter=50000, maxcor=50)
    student_model.compile("L-BFGS", loss_weights=loss_weights)
    
    with console.status("[bold green]Distilling (L-BFGS)...[/bold green]", spinner="dots"):
        with suppress_builtin_print():
            losshistory, _ = student_model.train(display_every=100, callbacks=[cb])
            
    save_path = os.path.join(os.path.dirname(__file__), "..", "ionpinn_distilled_small.pt")
    torch.save(student_model.net.state_dict(), save_path)
    console.print(f"[bold green]Success![/bold green] Distilled model saved to [bold cyan]{save_path}[/bold cyan]")
    
    # Plotting
    labels = ["PDE Solid", "PDE Elec", "Kinetics", "BC Left"]
    if X_train is not None:
        labels.append("Voltage Data")
    labels.extend(["KD c_s", "KD c_e", "KD phi_s", "KD phi_e", "KD j_Li"])
    
    loss_train = np.array(losshistory.loss_train)
    num_plots = len(labels)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=False)
    for i, label in enumerate(labels):
        if i < loss_train.shape[1]:
            axs[i].plot(losshistory.steps, loss_train[:, i], label="Train (Student)", color="purple")
        axs[i].axhline(y=1e-4, color="red", linestyle=":", linewidth=2, label="Target Goal ($10^{-4}$)")
        axs[i].set_ylabel("Loss")
        axs[i].set_yscale("log")
        axs[i].set_title(label)
        axs[i].legend(loc="upper left")
        axs[i].grid(True)
    axs[-1].set_xlabel("Epochs")
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "kd_loss_plot.png")
    plt.savefig(plot_path)
    console.print(f"[bold green]Success![/bold green] Saved distillation loss plot to [bold cyan]{plot_path}[/bold cyan]")
    plt.show()

if __name__ == "__main__":
    transfer_learning()
