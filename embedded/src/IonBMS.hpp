#ifndef ION_BMS_HPP
#define ION_BMS_HPP

#include <vector>
#include <array>
#include <cmath>

// Configuration for the Battery Pack
struct PackConfig {
    int n_series;
    int n_parallel;
    float cell_q_initial;
    float cell_v_cutoff;
    float scale_v;
    float scale_i;
    float scale_t;
};

// Lightweight Feed-Forward Neural Network Engine
// Architecture: [3, 32, 32, 5]
class FNN_Engine {
public:
    FNN_Engine();
    
    // Evaluate the network
    // Input array expected format: [time, spatial_coord, current]
    // Output array expected format: [c_s, phi_s, phi_e, c_e, j]
    void evaluate(const std::array<float, 3>& input, std::array<float, 5>& output) const;
    
    // Load weights dynamically to avoid hardcoding generated variable names
    void load_weights(const float* w1_in, const float* b1_in,
                      const float* w2_in, const float* b2_in,
                      const float* w3_in, const float* b3_in);

private:
    // Swish (SiLU) activation function: f(x) = x / (1.0 + exp(-x))
    inline float swish(float x) const {
        return x / (1.0f + std::exp(-x));
    }

    // Weights and biases
    // Layer 1: 3 -> 32
    std::array<std::array<float, 3>, 32> w1;
    std::array<float, 32> b1;
    
    // Layer 2: 32 -> 32
    std::array<std::array<float, 32>, 32> w2;
    std::array<float, 32> b2;
    
    // Layer 3: 32 -> 5
    std::array<std::array<float, 32>, 5> w3;
    std::array<float, 5> b3;
};

// Main Battery Management System Class
class IonBMS {
public:
    IonBMS(const PackConfig& config);

    // Initialize the FNN Engine with weights from the PyTorch exporter
    void load_model_weights(const float* w1_in, const float* b1_in,
                            const float* w2_in, const float* b2_in,
                            const float* w3_in, const float* b3_in);

    // Calculate State of Charge using Trapezoidal rule integrator
    float calculate_SoC(float t_current);

    // Calculate State of Health using a time-marching loop
    float calculate_SoH(float pack_current, float t_max);

    // Calculate Remaining Useful Life using least-squares linear regression
    float calculate_RUL(const float* cycle_history, const float* soh_history, int history_length);

private:
    FNN_Engine engine;
    PackConfig config;
};

#endif // ION_BMS_HPP
