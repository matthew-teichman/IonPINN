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
This will automatically set up a virtual environment and install all required packages (including PyTorch, DeepXDE, Rich, Matplotlib, Pandas, and ONNX Runtime). It will detect if you have an NVIDIA GPU and install CUDA-compatible Torch binaries where applicable.

### 2. Run the Training Pipeline
To train the Large and Small Foundation PINNs on the aging and transient CSVs, run:
```bash
poetry run python src/train.py
```

You can customize the training process with several command-line arguments:
*   `--epochs`: Number of training iterations (default: 40000)
*   `--learning_rate`: Learning rate for the optimizer (default: 1e-3)
*   `--plot_loss`: Add this flag to plot the training and test loss using `matplotlib` when training is complete
*   `--skip_large`: Skip training the large foundation model and only train the small baseline.

For example, to train for 20000 epochs with a custom learning rate and plot the results:
```bash
poetry run python src/train.py --epochs 20000 --learning_rate 0.005 --plot_loss
```
*Note: The script dynamically checks for a GPU and prints out whether it's using CUDA or falling back to the CPU.*

### 3. Knowledge Distillation (Transfer Learning)
Once the Large foundation model is trained, you can distill its learned physics manifold into the lightweight Small model for embedded deployment:
```bash
poetry run python src/transfer_learning.py
```
This script evaluates the Teacher model over a dense grid of 20,000 synthetic points and uses DeepXDE's `PointSetBC` to distill the smooth physics manifold into the Student model, saving the result as `ionpinn_distilled_small.pt`.

### 4. Export to ONNX
Once the model is trained (and weights saved as `.pt`), you can export it for embedded usage:
```bash
poetry run python src/export.py
```
This script will produce ONNX files in the root folder, equipped with dynamic batching properties ready for edge deployment.

### 5. Evaluate Metrics
You can evaluate the models' performance on aging datasets using the `evaluate_metrics.py` script. This script loads the Large, Small, and Distilled models and calculates Pack-Level SoC, SoH, and RUL across multiple cycles, outputting a live `rich` terminal dashboard alongside generated Matplotlib degradation reports side-by-side (`evaluation_report_*.png`).
```bash
poetry run python src/evaluate_metrics.py
```

### 6. Profile Model
You can profile the model's memory footprint and FLOPs for edge embedded deployment using the `profile_model.py` script.
```bash
poetry run python src/profile_model.py
```

## 📁 Outputs & Artifacts

### PyTorch Models (`.pt`)
*   `ionpinn_foundation_large.pt`: The heavy, overparameterized foundation baseline. Highly accurate but too large for embedded hardware.
*   `ionpinn_foundation_small.pt`: The lightweight embedded foundation model trained natively on the physics and empirical data.
*   `ionpinn_distilled_small.pt`: The lightweight embedded model generated via Knowledge Distillation. It mimics the large foundation model's learned physics manifold, resulting in higher stability and convergence than the native small model.

### Plots & Reports (`.png`)
*   `loss_plot.png`: Generated by `src/train.py` (when using `--plot_loss`). Displays the Adam and L-BFGS training loss convergence over time, tracking the 3 underlying PDEs and boundary conditions.
*   `kd_loss_plot.png`: Generated by `src/transfer_learning.py`. Tracks the convergence of the Student model as it learns to regress to the synthetic targets produced by the Teacher model.
*   `evaluation_report_*.png`: Generated by `src/evaluate_metrics.py`. A multi-panel degradation report containing:
    1.  **Terminal Voltage:** Compares the true empirical voltage curve to the predictions of the Large, Small, and Distilled models over the length of the discharge cycle.
    2.  **SoC Trajectory:** Plots the real-time calculated State of Charge (SoC) for the evaluated cycle.
    3.  **Pack SoH & RUL:** A linear degradation scatterplot that tracks State of Health (SoH) fade over many cycles, projecting the Remaining Useful Life (RUL) until the 80% End-of-Life threshold is crossed.
