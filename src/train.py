import os
import argparse
import deepxde as dde
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_device
from src.model import build_pinn_model
from src.data_loader import get_dataloaders

from rich.logging import RichHandler
from rich.console import Console

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
console = Console()

def train(epochs, batch_size, learning_rate, plot_loss):
    console.rule("[bold blue]IonPINN Foundation PINN Training[/bold blue]")
    # 1. Detect Device
    device = get_device()
    
    # 2. Setup Data (Optional depending on how DeepXDE handles points vs DataLoaders)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    aging_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    logger.info(f"Loaded aging dataset with {len(aging_loader.dataset)} cycles.")
    
    # 3. Build PINN Model
    model = build_pinn_model()
    
    # 4. Compile Model
    # Using L-BFGS or Adam as standard in PINNs
    model.compile("adam", lr=learning_rate)
    
    # 5. Train Model
    logger.info("Starting DeepXDE PINN optimization...")
    with console.status("[bold green]Training model...[/bold green]", spinner="dots"):
        losshistory, train_state = model.train(iterations=epochs, display_every=1)
    
    # 6. Save PyTorch Model
    save_path = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation.pt")
    # DeepXDE saves models in its own way, but we can extract the PyTorch module
    torch.save(model.net.state_dict(), save_path)
    console.print(f"[bold green]✔[/bold green] Training complete. Foundation model saved to [bold cyan]{save_path}[/bold cyan]")

    # 7. Plot Loss if requested
    if plot_loss:
        plt.figure(figsize=(10, 6))
        # DeepXDE losses are typically arrays (one for each PDE/BC component). Summing for total loss.
        train_loss = np.sum(np.array(losshistory.loss_train), axis=1)
        plt.plot(losshistory.steps, train_loss, label="Total Train Loss")
        
        if hasattr(losshistory, "loss_test") and len(losshistory.loss_test) > 0:
            test_loss = np.sum(np.array(losshistory.loss_test), axis=1)
            plt.plot(losshistory.steps, test_loss, label="Total Test Loss")
            
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(os.path.dirname(__file__), "..", "loss_plot.png")
        plt.savefig(plot_path)
        logger.info(f"Saved loss plot to {plot_path}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IonPINN Foundation Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (iterations)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--plot_loss", action="store_true", help="Plot the loss after training using matplotlib")
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, plot_loss=args.plot_loss)
