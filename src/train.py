import os
import argparse
import deepxde as dde
dde.config.set_default_float("float64")
import torch
torch.set_default_dtype(torch.float64)
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_device
from src.model import build_pinn_model
from src.data_loader import get_pinn_training_data

import builtins
from contextlib import contextmanager
from deepxde.callbacks import Callback

try:
    from rich.logging import RichHandler
    from rich.console import Console
    FORMAT = "%(message)s"
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
    logger = logging.getLogger("rich")
    console = Console()
except ImportError:
    FORMAT = "%(message)s"
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]")
    logger = logging.getLogger("basic")
    class DummyConsole:
        def rule(self, text): print(text)
        def print(self, text): print(text)
        @contextmanager
        def status(self, text, spinner="dots"):
            print(text)
            yield
    console = DummyConsole()

@contextmanager
def suppress_builtin_print():
    """Temporarily replaces Python's built-in print to silence DeepXDE's hardcoded prints and reformat L-BFGS steps in Rich."""
    original_print = builtins.print
    def _smart_print(*args, **kwargs):
        if not args:
            return
        msg = str(args[0])
        # DeepXDE prints steps like: "11000     [6.23e-01, 9.03e-04, 2.02e-02, 6.60e-01] ..."
        # Check if the string starts with a number (the step)
        parts = msg.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].startswith("["):
            step = parts[0]
            try:
                # Parse the bracketed string for train losses
                loss_str_part = msg[msg.find("[")+1:msg.find("]")]
                losses = [float(x.strip()) for x in loss_str_part.split(",")]
                labels = ["PDE Solid", "PDE Elec", "Kinetics", "BC Left", "IC c_s", "Voltage Data"]
                # Only zip up to the number of available labels/losses
                rich_str = " | ".join([f"[cyan]{lbl}:[/cyan] {val:.2e}" for lbl, val in zip(labels, losses)])
                console.print(f"[bold yellow]Step {step}[/bold yellow] | {rich_str}")
            except Exception:
                pass # Silently drop if parsing fails
        # Otherwise, silently drop (suppress)
    builtins.print = _smart_print
    try:
        yield
    finally:
        builtins.print = original_print



class RichLossCallback(Callback):
    def __init__(self, display_every=1):
        super().__init__()
        self.display_every = display_every

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.display_every == 0:
            loss_train = self.model.train_state.loss_train
            labels = ["PDE Solid", "PDE Elec", "Kinetics", "BC Left", "IC c_s", "Voltage Data"]
            # Formatting them beautifully
            loss_str = " | ".join([f"[cyan]{lbl}:[/cyan] {val:.2e}" for lbl, val in zip(labels, loss_train)])
            console.print(f"[bold yellow]Step {step}[/bold yellow] | {loss_str}")

def train(epochs, learning_rate, plot_loss, skip_large):
    console.rule("[bold blue]IonPINN Foundation PINN Training[/bold blue]")
    # 1. Detect Device
    device = get_device()

    # 2. Setup Data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_train, Y_train = get_pinn_training_data(data_dir=data_dir)
    logger.info(f"Loaded empirical training data with {len(X_train)} points." if X_train is not None else "No empirical data loaded.")

    def train_model(model, spme, name, cb):
        # Weights: [PDE Solid, PDE Elec, Kinetics, BC Left, IC c_s, Voltage Data (if present)]
        num_losses = len(model.data.bcs) + 3 # 3 PDEs + BCs
        loss_weights = [1.0, 1.0, 1e-4, 50.0, 50.0]
        if num_losses > 5:
            loss_weights.append(100.0) # High weight for actual voltage data
        model.compile("adam", lr=learning_rate, decay=("step", 2000, 0.4), loss_weights=loss_weights)

        logger.info(f"Starting DeepXDE PINN optimization (Stage 1: Adam) for {name} model...")
        resampler = dde.callbacks.PDEPointResampler(period=2000)

        with console.status(f"[bold green]Training {name} model (Adam)...[/bold green]", spinner="dots"):
            with suppress_builtin_print():
                model.train(iterations=epochs, display_every=100, callbacks=[cb, resampler])

        logger.info(f"Starting DeepXDE PINN optimization (Stage 2: L-BFGS) for {name} model...")

        # Reduce maxiter and maxcor to prevent CUDA Out Of Memory / Unknown Error on 4GB GPUs
        dde.optimizers.set_LBFGS_options(ftol=1e-12, gtol=1e-12, maxiter=100000, maxcor=50)
        model.compile("L-BFGS", loss_weights=loss_weights)

        # Increase print frequency for L-BFGS because it is very slow per iteration
        cb_lbfgs = RichLossCallback(display_every=100)
        with console.status(f"[bold green]Training {name} model (L-BFGS)...[/bold green]", spinner="dots"):
            with suppress_builtin_print():
                losshistory, _ = model.train(display_every=100, callbacks=[cb_lbfgs])

        return losshistory

    # 3. Build & Train Small Model
    model_small, spme_small = build_pinn_model(X_train, Y_train, large=False)
    cb_small = RichLossCallback(display_every=100)
    history_small = train_model(model_small, spme_small, "Small", cb_small)

    # 4. Build & Train Large Model
    if not skip_large:
        model_large, spme_large = build_pinn_model(X_train, Y_train, large=True)
        cb_large = RichLossCallback(display_every=100)
        history_large = train_model(model_large, spme_large, "Large", cb_large)

    # 6. Save PyTorch Models
    save_path_small = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation_small.pt")
    torch.save(model_small.net.state_dict(), save_path_small)

    if not skip_large:
        save_path_large = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation_large.pt")
        torch.save(model_large.net.state_dict(), save_path_large)
        console.print(f"[bold green]Success![/bold green] Training complete. Foundation models saved to [bold cyan]{save_path_small}[/bold cyan] and [bold cyan]{save_path_large}[/bold cyan]")
    else:
        console.print(f"[bold green]Success![/bold green] Training complete. Foundation model saved to [bold cyan]{save_path_small}[/bold cyan]")

    # 7. Plot Loss if requested
    if plot_loss:
        labels = ["PDE Solid", "PDE Elec", "Kinetics", "BC Left", "IC c_s", "Voltage Data"]
        loss_train_small = np.array(history_small.loss_train)
        has_test_small = hasattr(history_small, "loss_test") and len(history_small.loss_test) > 0
        if has_test_small:
            loss_test_small = np.array(history_small.loss_test)

        if not skip_large:
            loss_train_large = np.array(history_large.loss_train)
            has_test_large = hasattr(history_large, "loss_test") and len(history_large.loss_test) > 0
            if has_test_large:
                loss_test_large = np.array(history_large.loss_test)

        num_plots = len(labels)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=False)

        for i, label in enumerate(labels):
            axs[i].plot(history_small.steps, loss_train_small[:, i], label="Train (Small)", color="blue")
            if has_test_small:
                axs[i].plot(history_small.steps, loss_test_small[:, i], label="Validation (Small)", color="lightblue", linestyle="--")

            if not skip_large:
                axs[i].plot(history_large.steps, loss_train_large[:, i], label="Train (Large)", color="green")
                if has_test_large:
                    axs[i].plot(history_large.steps, loss_test_large[:, i], label="Validation (Large)", color="lightgreen", linestyle="--")

            # Add horizontal dotted line for the target goal loss
            axs[i].axhline(y=1e-4, color="red", linestyle=":", linewidth=2, label="Target Goal ($10^{-4}$)")

            axs[i].set_ylabel("Loss")
            axs[i].set_yscale("log")
            axs[i].set_title(label)
            axs[i].legend(loc="upper left")
            axs[i].grid(True)

        axs[-1].set_xlabel("Epochs")
        plt.tight_layout()

        plot_path = os.path.join(os.path.dirname(__file__), "..", "loss_plot.png")
        plt.savefig(plot_path)
        console.print(f"[bold green]Success![/bold green] Saved detailed loss plot to [bold cyan]{plot_path}[/bold cyan]")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IonPINN Foundation Model")
    parser.add_argument("--epochs", type=int, default=40000, help="Number of training epochs (iterations)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--plot_loss", action="store_true", help="Plot the loss after training using matplotlib")
    parser.add_argument("--skip_large", action="store_true", help="Skip training the large model")

    args = parser.parse_args()

    train(epochs=args.epochs, learning_rate=args.learning_rate, plot_loss=args.plot_loss, skip_large=args.skip_large)
