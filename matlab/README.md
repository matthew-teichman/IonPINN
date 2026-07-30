# MATLAB Data Preprocessing

This directory contains the MATLAB script (`preprocess_data.m`) used for Phase 1 of the IonPINN project to parse, clean, and export the NASA battery datasets into formats suitable for deep learning.

## What It Does

The `preprocess_data.m` script automates the ingestion of NASA's `.mat` files from both Dataset 5 and Dataset 11. It performs the following specific actions:

1. **Creates Target Directories:** It builds a `data/` structure at the root of the project featuring `Training`, `Validation`, `Testing`, and `Unused` folders.
2. **Deep Struct Parsing:** The script natively drills into the heavily nested MATLAB `.mat` files to dynamically extract the `Time`, `Voltage`, `Current`, and `Temperature` vectors for every cycle (or step).
3. **Zero-Phase Filtering:** Because raw battery sensor data often contains high-frequency noise that impedes neural network training, the script applies MATLAB's `filtfilt` command (via a 5-step moving average window). This acts as a zero-phase filter, meaning the noise is smoothed *without* artificially shifting or delaying the time-series transient peaks.
4. **Chronological CSV Export:** The script flattens the multidimensional struct arrays and exports them as clean, 2D `.csv` files. Each cycle is saved as an individual file named chronologically (e.g., `B0005_cycle001.csv`).
5. **Auto-Categorization:** Specific datasets (B0005, B0006, B0007, B0018, RW1, RW2, RW9, RW10) are organized into their respective Training/Validation/Testing split, and all remaining `.mat` files are dumped into the `Unused/` folder for later investigation.

## How to Run

Because `filtfilt` and struct parsing rely on the MATLAB environment, this script cannot be executed natively in Python.

1. Open MATLAB and navigate your Current Folder to `IonPINN/matlab/`.
2. In the MATLAB Command Window, simply run:
   ```matlab
   preprocess_data
   ```
3. Wait for the processing to finish. The generated datasets will populate the `IonPINN/data/` folder automatically.
