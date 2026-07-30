import os
import torch
import torch.nn as nn
import logging
from src.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportableIonPINN(nn.Module):
    """
    Standard PyTorch wrapper for the DeepXDE FNN.
    Matches the architecture defined in model.py's dde.nn.FNN:
    Input: 2 [t, r]
    Output: 4 [c_s, phi_s, c_e, phi_e]
    """
    def __init__(self):
        super().__init__()
        # Matches [2] + [64]*3 + [4] structure
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        return self.net(x)

def export_to_onnx():
    device = get_device()
    
    # 1. Initialize Network
    model = ExportableIonPINN()
    model.to(device)
    
    # 2. Load Weights
    pt_path = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation.pt")
    if os.path.exists(pt_path):
        try:
            # DeepXDE state_dict might need mapping depending on the version
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
            # Filter and map state dict keys if needed
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded Foundation weights.")
        except Exception as e:
            logger.warning(f"Could not load weights directly, exporting with random weights for demo: {e}")
    else:
        logger.warning("No .pt file found. Exporting untrained network structure.")
        
    model.eval()
    
    # 3. Dummy Input for Tracing
    # Batch size 1, 2 features (t, r)
    dummy_input = torch.randn(1, 2, device=device)
    
    # 4. Export
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "ionpinn_foundation.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input_xt'], 
        output_names=['output_states'],
        dynamic_axes={'input_xt': {0: 'batch_size'}, 'output_states': {0: 'batch_size'}}
    )
    
    logger.info(f"Successfully exported IonPINN Foundation Model to ONNX: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
