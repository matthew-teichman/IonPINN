#include "IonBMS.hpp"

// -----------------------------------------------------------------------------
// FNN_Engine Implementation
// -----------------------------------------------------------------------------

FNN_Engine::FNN_Engine() {
    // Initialize weights and biases to zero
    for (auto& row : w1) row.fill(0.0f);
    b1.fill(0.0f);
    for (auto& row : w2) row.fill(0.0f);
    b2.fill(0.0f);
    for (auto& row : w3) row.fill(0.0f);
    b3.fill(0.0f);
}

void FNN_Engine::evaluate(const std::array<float, 3>& input, std::array<float, 5>& output) const {
    // Forward Pass Layer 1: 3 -> 32
    std::array<float, 32> h1;
    for (int i = 0; i < 32; ++i) {
        float sum = b1[i];
        for (int j = 0; j < 3; ++j) {
            sum += w1[i][j] * input[j];
        }
        h1[i] = swish(sum);
    }

    // Forward Pass Layer 2: 32 -> 32
    std::array<float, 32> h2;
    for (int i = 0; i < 32; ++i) {
        float sum = b2[i];
        for (int j = 0; j < 32; ++j) {
            sum += w2[i][j] * h1[j];
        }
        h2[i] = swish(sum);
    }

    // Forward Pass Layer 3: 32 -> 5 (Linear output)
    for (int i = 0; i < 5; ++i) {
        float sum = b3[i];
        for (int j = 0; j < 32; ++j) {
            sum += w3[i][j] * h2[j];
        }
        output[i] = sum; 
    }
}

void FNN_Engine::load_weights(const float* w1_in, const float* b1_in,
                              const float* w2_in, const float* b2_in,
                              const float* w3_in, const float* b3_in) {
    // Load Layer 1
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 3; ++j) {
            w1[i][j] = w1_in[i * 3 + j];
        }
        b1[i] = b1_in[i];
    }
    
    // Load Layer 2
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            w2[i][j] = w2_in[i * 32 + j];
        }
        b2[i] = b2_in[i];
    }
    
    // Load Layer 3
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            w3[i][j] = w3_in[i * 32 + j];
        }
        b3[i] = b3_in[i];
    }
}


// -----------------------------------------------------------------------------
// IonBMS Implementation
// -----------------------------------------------------------------------------

IonBMS::IonBMS(const PackConfig& cfg) : config(cfg) {
}

void IonBMS::load_model_weights(const float* w1_in, const float* b1_in,
                                const float* w2_in, const float* b2_in,
                                const float* w3_in, const float* b3_in) {
    engine.load_weights(w1_in, b1_in, w2_in, b2_in, w3_in, b3_in);
}

float IonBMS::calculate_SoC(float t_current) {
    constexpr int N_GRID = 50;
    float dr = 1.0f / (N_GRID - 1);
    float integral = 0.0f;
    
    // Purely numerical Trapezoidal rule integrator for spherical volume average: 3 * int_{0}^{1} c_s * r^2 dr
    for (int i = 0; i < N_GRID; ++i) {
        float r = i * dr;
        
        // Input to FNN: [time, spatial_coord, current]. Assume 0 current for rest SoC.
        std::array<float, 3> in = {t_current, r, 0.0f}; 
        std::array<float, 5> out;
        engine.evaluate(in, out);
        
        // Assume output index 0 is c_s
        float c_s = out[0]; 
        float val = c_s * r * r;
        
        float weight = (i == 0 || i == N_GRID - 1) ? 0.5f : 1.0f;
        integral += weight * val;
    }
    
    integral *= dr;
    return 3.0f * integral;
}

float IonBMS::calculate_SoH(float pack_current, float t_max) {
    // Scale pack current down to cell level using n_parallel
    float cell_current = pack_current / static_cast<float>(config.n_parallel);
    
    constexpr int N_STEPS = 1000;
    float dt = t_max / N_STEPS;
    float t = 0.0f;
    
    float capacity_delivered = 0.0f;
    
    // Simulated time-marching loop
    while (t <= t_max) {
        // Evaluate at x = 0.0 (negative electrode interface)
        std::array<float, 3> in_x0 = {t, 0.0f, cell_current};
        std::array<float, 5> out_x0;
        engine.evaluate(in_x0, out_x0);
        
        // Evaluate at x = 1.0 (positive electrode interface)
        std::array<float, 3> in_x1 = {t, 1.0f, cell_current};
        std::array<float, 5> out_x1;
        engine.evaluate(in_x1, out_x1);
        
        // Assume output index 1 is phi_s (solid phase potential)
        float phi_s_0 = out_x0[1]; 
        float phi_s_1 = out_x1[1]; 
        
        // Terminal voltage calculation: V = phi_s(x=1) - phi_s(x=0)
        float v_cell = phi_s_1 - phi_s_0;
        
        // Stop the loop when cell voltage drops below V_cutoff
        if (v_cell < config.cell_v_cutoff) {
            break;
        }
        
        capacity_delivered += cell_current * dt;
        t += dt;
    }
    
    // Calculate capacity ratio. Scale remains accurate for pack level.
    return capacity_delivered / config.cell_q_initial;
}

float IonBMS::calculate_RUL(const float* cycle_history, const float* soh_history, int history_length) {
    // Need at least 2 points to perform linear regression
    if (history_length < 2) return 0.0f;
    
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
    
    for (int i = 0; i < history_length; ++i) {
        sum_x += cycle_history[i];
        sum_y += soh_history[i];
        sum_xy += cycle_history[i] * soh_history[i];
        sum_xx += cycle_history[i] * cycle_history[i];
    }
    
    float n = static_cast<float>(history_length);
    float denominator = (n * sum_xx - sum_x * sum_x);
    
    // Prevent division by zero
    if (std::abs(denominator) < 1e-6f) return 0.0f;
    
    // Pure C++ least-squares linear regression (1st-degree polyfit)
    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    float intercept = (sum_y - slope * sum_x) / n;
    
    // If slope is flat or positive, degradation isn't happening properly
    if (std::abs(slope) < 1e-8f) return 9999.0f; 
    
    // Calculate intersection with 80% EoL threshold
    // 0.8 = slope * cycle_eol + intercept
    float cycle_eol = (0.8f - intercept) / slope;
    float current_cycle = cycle_history[history_length - 1];
    
    float rul = cycle_eol - current_cycle;
    return rul > 0.0f ? rul : 0.0f;
}
