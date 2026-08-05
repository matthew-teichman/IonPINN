#include <Arduino.h>
#include "IonBMS.hpp"

// Forward declaration of auto-generated weights
// The exporter script should have created IonWeights.hpp in the same directory.
// We include it here to pass to the BMS.
#include "IonWeights.hpp"

// Example pack configuration
PackConfig config = {
    14,     // n_series (e.g., 14S pack)
    4,      // n_parallel
    3.2f,   // cell_q_initial (Ah)
    2.5f,   // cell_v_cutoff
    1.0f,   // scale_v
    1.0f,   // scale_i
    1.0f    // scale_t
};

IonBMS bms(config);

void setup() {
    Serial.begin(115200);
    // Wait for serial monitor to connect
    delay(2000);

    Serial.println("Initializing IonBMS on ESP32...");

    // Load the neural network parameters from the exported header
    bms.load_model_weights(
        linears_0_weight, linears_0_bias,
        linears_1_weight, linears_1_bias,
        linears_2_weight, linears_2_bias
    );

    Serial.println("BMS Initialized. Weights loaded successfully.");
}

void loop() {
    // Simulated operation time (seconds)
    float current_time = millis() / 1000.0f;

    // Evaluate the State of Charge using the PINN
    // (Ensure you measure the real values in production rather than simulated time)
    float soc = bms.calculate_SoC(current_time);

    Serial.print("Time: ");
    Serial.print(current_time);
    Serial.print("s | Current SoC: ");
    Serial.print(soc * 100.0f);
    Serial.println("%");

    // You can also run calculate_SoH periodically (it's computationally heavier)

    delay(2000);
}
