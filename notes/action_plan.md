# Action Plan

## Phase 1: Build the Single-Cell Base
- Model MATLAB Data Pre-Processing: Use MATLAB 2018b's native struct handling to drill into the deeply nested NASA .mat files. Apply the Signal Processing Toolbox (specifically filtfilt) to run zero-phase filtering on the noisy voltage and current sensors, removing the test cell noise without artificially shifting the transient timing. Export clean, flattened .csv files.

- Python PINN Training: Load the .csv files into your Ubuntu development stack. Use DeepXDE and PyTorch to train the SPMe-embedded neural network on Dataset 5 (for long-term capacity fade) and Dataset 11 (for transient dynamics).

- ONNX Export: Export the fully trained, cell-level Foundation PINN from PyTorch as an .onnx file.

## Phase 2: Build the Pack-Level Wrapper
- Simscape Electrical Modeling: Instead of manually coding an Equivalent Circuit Model (ECM) and thermal topology in Python, use Simulink and Simscape Electrical. Visually map out the target battery pack configuration (e.g., 10s4p), adding thermal masses and convective heat transfer blocks to simulate the pack's physical environment.

- ONNX Import: Install the Deep Learning Toolbox Converter for ONNX Model Format add-on. Use the importONNXNetwork command to bring your Python-trained foundation model directly into your MATLAB workspace.

- The Hybrid Simulation: Embed the imported PINN inside a MATLAB Function block connected to your Simscape pack model. As Simscape drives a dynamic load, the PINN acts as the central estimator, calculating the instantaneous cell-level SoC/SoH based on the simulated temperatures and voltages.

## Phase 3: The Open-Source Deployment Pipeline
- Transfer Learning API (Python): Maintain the open-source Python script that allows end-users to input a short calibration cycle from their custom hardware to rapidly update the network weights for their specific cell chemistry.

- C++ Edge Deployment (MATLAB Coder): This is where the embedded software architecture comes together. Once a user runs the Transfer Learning API and generates their customized .onnx model, they drop it into your provided Simulink pack wrapper. From there, use MATLAB Coder to automatically generate highly optimized, standalone C/C++ code from the entire Simulink model (the Simscape wrapper + the customized ONNX PINN). This gives users a direct, automated pipeline to flash a state-of-the-art physics-informed estimator directly to bare-metal microcontroller hardware.