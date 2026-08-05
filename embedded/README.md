# IonBMS: Embedded PINN Battery Management System

This library provides a bare-metal, dependency-free C++ implementation of a Physics-Informed Neural Network (PINN) for real-time battery state estimation. It is designed to run on microcontrollers like the ESP32 efficiently without relying on dynamic memory allocation (`malloc` or `new`).

## Workflow

To use this library, you first need to export the trained weights from your PyTorch model into a format the C++ code can understand.

1. **Export the Weights**
   Run the Python exporter script on your desktop:
   ```bash
   python export_weights.py
   ```
   This reads `ionpinn_foundation_small.pt` and generates an `IonWeights.hpp` file containing static arrays of the network's parameters.

2. **Add Files to Your Project**
   Place the following files in your embedded project's `src/` or `include/` directory:
   - `IonBMS.hpp`
   - `IonBMS.cpp`
   - `IonWeights.hpp`

## How to Use the IonBMS Object

### 1. Include the Headers
Include the main BMS header and the auto-generated weights header in your source file (e.g., `main.cpp`):
```cpp
#include "IonBMS.hpp"
#include "IonWeights.hpp"
```

### 2. Configure the Pack
Create a `PackConfig` struct to define your battery pack's physical parameters. This configuration handles scaling the neural network's single-cell predictions up to the multi-cell pack level.

```cpp
PackConfig config = {
    14,     // n_series (e.g., 14 cells in series)
    4,      // n_parallel (e.g., 4 cells in parallel)
    3.2f,   // cell_q_initial: initial capacity of a single cell (Ah)
    2.5f,   // cell_v_cutoff: minimum safe voltage per cell
    1.0f,   // scale_v: voltage scaling factor (if applied during training)
    1.0f,   // scale_i: current scaling factor (if applied during training)
    1.0f    // scale_t: time scaling factor (if applied during training)
};
```

### 3. Instantiate the Object
Create an instance of `IonBMS`, passing it the configuration struct:
```cpp
IonBMS bms(config);
```

### 4. Load the Model Weights
Before making predictions, you must load the generated PyTorch weights into the engine. This is typically done once during your device's startup/setup phase:
```cpp
void setup() {
    bms.load_model_weights(
        linears_0_weight, linears_0_bias,
        linears_1_weight, linears_1_bias,
        linears_2_weight, linears_2_bias
    );
}
```

### 5. Running Estimations

#### State of Charge (SoC)
SoC is calculated continuously by querying the PINN for the solid-phase lithium concentration over the particle radius and integrating it via the Trapezoidal rule.
```cpp
void loop() {
    // Pass the current operating time (in seconds)
    float current_time = millis() / 1000.0f; 
    
    // Returns a value between 0.0 (0%) and 1.0 (100%)
    float soc = bms.calculate_SoC(current_time);
}
```

#### State of Health (SoH)
SoH calculates the effective battery capacity by running a simulated time-marching discharge curve through the PINN to determine how much capacity can be delivered before hitting the cutoff voltage. 

*Note: Because it runs a simulation loop, it is computationally heavier than SoC. It is recommended to run this periodically (e.g., once a day, or while charging), rather than in a fast real-time loop.*
```cpp
float current_pack_current = 10.0f; // Amps being drawn from the pack
float max_simulation_time = 3600.0f; // Max seconds to simulate

// Returns capacity ratio (typically 0.0 to 1.0)
float soh = bms.calculate_SoH(current_pack_current, max_simulation_time);
```

#### Remaining Useful Life (RUL)
RUL performs a pure C++ least-squares linear regression on historical cycle data to project when the battery degradation trend will cross the 80% EoL (End of Life) threshold.
```cpp
// Maintain historical logs of SoH evaluated at different cycle intervals
float cycle_history[] = {10, 50, 100, 150, 200};
float soh_history[] = {1.0f, 0.98f, 0.95f, 0.91f, 0.88f};

// Returns the predicted remaining cycles until 80% capacity
float remaining_cycles = bms.calculate_RUL(cycle_history, soh_history, 5);
```

## Performance & System Notes
- **FPU Recommended:** Because the PINN relies heavily on floating-point arithmetic (matrix multiplications and exponential functions for the Swish activation), it is highly recommended to run this on a microcontroller with a hardware FPU (like the ESP32). 
- **Memory Allocation:** All neural network evaluation matrices and temporary states are safely allocated on the stack (via `std::array`). Ensure your RTOS task running the `IonBMS` routines has an adequate stack size allocated.
