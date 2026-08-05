import torch
import os

def export_weights(model_path, output_path):
    print(f"Loading PyTorch model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    try:
        # Load the PyTorch state dictionary
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Exporting weights to: {output_path}")
    with open(output_path, 'w') as f:
        f.write("#ifndef ION_WEIGHTS_HPP\n")
        f.write("#define ION_WEIGHTS_HPP\n\n")
        f.write("// Auto-generated weights from IonPINN PyTorch model\n")
        f.write("// DO NOT EDIT MANUALLY\n\n")

        # Iterate through all keys in the state_dict
        for key, tensor in state_dict.items():
            # Replace periods with underscores for valid C++ variable names
            var_name = key.replace('.', '_')

            # Flatten tensor into 1D array
            arr = tensor.detach().cpu().numpy().flatten()

            # Declare C++ static constant array
            f.write(f"static const float {var_name}[{len(arr)}] = {{\n    ")

            # Format each value using scientific notation to preserve float32/float64 precision
            formatted_vals = [f"{val:.8e}f" for val in arr]

            # Format the output to have 8 values per line for readability
            for i in range(0, len(formatted_vals), 8):
                f.write(", ".join(formatted_vals[i:i+8]))
                if i + 8 < len(formatted_vals):
                    f.write(",\n    ")

            f.write("\n};\n\n")

        f.write("#endif // ION_WEIGHTS_HPP\n")
        print("Export completed successfully.")

if __name__ == "__main__":
    # Standard usage as requested
    export_weights('../ionpinn_foundation_small.pt', 'IonWeights.hpp')
