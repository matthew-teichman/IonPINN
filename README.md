# IonPINN: Foundation Battery Model 🔋

IonPINN is an open-source Physics-Informed Neural Network (PINN) designed for high-accuracy State of Charge (SoC), State of Health (SoH), and Remaining Useful Life (RUL) estimation in Lithium-ion batteries.

Instead of relying purely on empirical data or computationally heavy numerical solvers, IonPINN embeds a reduced-order electrochemical model (SPMe) into the neural network's loss function. This creates a lightweight, highly accurate model that obeys the laws of thermodynamics and electrochemistry.

## 🚀 The Foundation & Transfer Learning Approach
IonPINN is built to be a **Foundation Model**. The core network is pre-trained on the NASA PCoE Lithium-ion aging datasets (18650 LCO cells) to learn fundamental battery physics.

Because battery chemistry and pack topology vary wildly, IonPINN is not a rigid, one-size-fits-all model. It includes a streamlined **Transfer Learning API**. By feeding the model a short calibration cycle from *your specific battery pack*, IonPINN freezes its core physics layers and rapidly adapts to your hardware's unique capacity and internal resistance profiles.

## ✨ Features
*   **Physics-Informed Loss:** Uses an embedded Single Particle Model with Electrolyte (SPMe) to prevent physically impossible state predictions.
*   **Transient & Degradation Modeling:** Handles both short-term dynamic power draws (SoC) and long-term capacity fade (SoH/RUL).
*   **Transfer Learning API:** Easily adapt the model to new cell chemistries (NCA, LFP) and pack configurations ($N_s$/$N_p$) with minimal data.
*   **Edge-Ready:** Export the trained model to ONNX for lightweight inference on C++ embedded stacks or Battery Management Systems (BMS).

## 🛠 Installation & Quick Start

IonPINN uses [Poetry](https://python-poetry.org/) to manage its dependencies cross-platform.

### 1. Install Dependencies
Ensure you have Poetry installed. From the root directory, run:
```bash
poetry install
```
This will automatically set up a virtual environment and install all required packages (including PyTorch, DeepXDE, Pandas, and ONNX Runtime). It will detect if you have an NVIDIA GPU and install CUDA-compatible Torch binaries where applicable.

### 2. Run the Training Pipeline
To train the Foundation PINN on the Dataset 5 (aging) and Dataset 11 (transient) CSVs, run:
```bash
poetry run python -m src/train.py
```

You can customize the training process with several command-line arguments:
*   `--epochs`: Number of training iterations (default: 40000)
*   `--learning_rate`: Learning rate for the optimizer (default: 1e-3)
*   `--plot_loss`: Add this flag to plot the training and test loss using `matplotlib` when training is complete

For example, to train for 500 epochs with a custom learning rate and plot the results:
```bash
poetry run python -m src.train --epochs 20000 --learning_rate 0.005 --plot_loss
```
*Note: The script dynamically checks for a GPU and prints out whether it's using CUDA or falling back to the CPU.*

### 3. Export to ONNX
Once the model is trained (and weights saved as `.pt`), you can export it for embedded usage:
```bash
poetry run python -m src/export.py
```
This script will produce `ionpinn_foundation.onnx` in the root folder, equipped with dynamic batching properties ready for edge deployment.

### 4. Evaluate Metrics
You can evaluate the trained model's performance on aging datasets using the `evaluate_metrics.py` script. This script loads the saved weights, calculates Pack-Level SoC, SoH, and RUL across multiple cycles, and outputs a live `rich` terminal dashboard alongside a generated Matplotlib degradation report (`evaluation_report.png`).
```bash
poetry run python src/evaluate_metrics.py
```
