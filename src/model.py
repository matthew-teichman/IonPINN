import torch
import torch.nn as nn
import deepxde as dde
import numpy as np

class SPMePhysics:
    """
    Defines the Single Particle Model with Electrolyte (SPMe) governing equations.
    This acts as the physics loss for the DeepXDE framework.
    """
    def __init__(self, D_s=1e-14, R_s=10e-6):
        self.D_s = D_s
        self.R_s = R_s
        # Parameters for Electrolyte and Kinetics (placeholders)
        self.eps_e = 0.3
        self.D_e = 1e-10
        self.t_plus = 0.36
        self.F = 96485.332
        self.alpha_a = 0.5
        self.alpha_c = 0.5
        self.R = 8.314
        self.T = 298.15
        self.i_0 = 1.0
        
        # Causal PINNs configuration
        self.causal_training = True
        self.epsilon = 100.0
        self.n_bins = 100

    def U_ocp(self, c_s):
        # Placeholder for Open Circuit Potential lookup or function
        return 4.2 - 1.0 * c_s

    def pde(self, x, y):
        """
        x: Spatiotemporal coordinates [t, r, x_cell] 
           (time, particle radius, and cell thickness)
        y: Predicted states [c_s, c_e, phi_s, phi_e, j_Li]
        """
        # 1. Extract predicted states from the neural network output
        c_s = y[:, 0:1]   # Solid concentration
        c_e = y[:, 1:2]   # Electrolyte concentration
        phi_s = y[:, 2:3] # Solid potential
        phi_e = y[:, 3:4] # Electrolyte potential
        j_Li = y[:, 4:5]  # Interfacial current density

        # Extract coordinates
        t = x[:, 0:1]
        r = x[:, 1:2]       # Microscale: inside the particle
        x_cell = x[:, 2:3]  # Macroscale: across the electrode

        # =======================================================
        # EQUATION 1: Solid-Phase Diffusion (Fick's Law)
        # =======================================================
        dc_s_dt = dde.grad.jacobian(y, x, i=0, j=0) # wrt time
        dc_s_dr = dde.grad.jacobian(y, x, i=0, j=1) # wrt radius
        d2c_s_dr2 = dde.grad.hessian(y, x, component=0, i=1, j=1)

        # Handle the r=0 singularity in spherical coordinates
        # PyTorch evaluates both branches of torch.where. 
        # If r=0, 2.0/r is inf, which can create NaNs in the backward pass!
        r_safe = torch.where(torch.abs(r) < 1e-7, torch.ones_like(r) * 1e-7, r)
        spherical_term = torch.where(
            torch.abs(r) < 1e-7, 
            3.0 * d2c_s_dr2, 
            d2c_s_dr2 + (2.0 / r_safe) * dc_s_dr
        )
        loss_solid = dc_s_dt - spherical_term

        # =======================================================
        # EQUATION 2: Electrolyte Dynamics
        # =======================================================
        dc_e_dt = dde.grad.jacobian(y, x, i=1, j=0) # wrt time
        d2c_e_dx2 = dde.grad.hessian(y, x, component=1, i=2, j=2) # wrt cell thickness

        # F Faraday's constant, t_plus Transference number
        loss_electrolyte = (self.eps_e * dc_e_dt) - (self.D_e * d2c_e_dx2) - ((1 - self.t_plus) / self.F) * j_Li

        # =======================================================
        # EQUATION 3: Electrochemical Kinetics (Butler-Volmer)
        # =======================================================
        # Calculate Overpotential (eta)
        # U_ocp would be a function or lookup table based on surface c_s
        eta = phi_s - phi_e - self.U_ocp(c_s) 
        
        # Clamp eta to physically realistic bounds to prevent gradient explosion.
        # In a real Li-ion battery, overpotential (eta) rarely exceeds +/- 0.2V. 
        # An overpotential of 2.0V causes the exponential to output huge values 
        # (e.g., 10^16), which produces gradients that overflow float32 during backprop!
        eta = torch.clamp(eta, min=-0.5, max=0.5)
        
        # Normalize eta by the thermal voltage (F / RT)
        eta_norm = eta * (self.F / (self.R * self.T))
        
        # Butler-Volmer equation
        term_a = torch.exp(self.alpha_a * eta_norm)
        term_c = torch.exp(-self.alpha_c * eta_norm)
        
        loss_kinetics = j_Li - self.i_0 * (term_a - term_c)

        # =======================================================
        # Continuous Causal Training (Causal PINNs)
        # =======================================================
        # Bypass causal weighting for the validation set (which has exactly 500 points)
        # to ensure the validation metrics reflect the true, unweighted physics error.
        is_test_set = x.shape[0] == 500
        
        if self.causal_training and not is_test_set:
            with torch.no_grad():
                t_flat = t.squeeze(1)
                # Determine bin index for each point (clamped to prevent out-of-bounds)
                bin_idx = torch.clamp((t_flat * self.n_bins).long(), 0, self.n_bins - 1)
                
                # Raw spatial residuals at current evaluation
                L_raw = torch.abs(loss_solid.squeeze(1)) + torch.abs(loss_electrolyte.squeeze(1))
                
                # Sum of residuals in each bin
                bin_loss = torch.zeros(self.n_bins, device=x.device, dtype=x.dtype)
                bin_loss.scatter_add_(0, bin_idx, L_raw)
                
                # Count points per bin to get spatial mean
                bin_count = torch.zeros(self.n_bins, device=x.device, dtype=x.dtype)
                bin_count.scatter_add_(0, bin_idx, torch.ones_like(t_flat))
                
                # Mean residual per bin
                bin_mean_loss = torch.where(bin_count > 0, bin_loss / bin_count, torch.zeros_like(bin_loss))
                
                # Cumulative sum of past errors (shifted by 1)
                cumsum_loss = torch.cumsum(bin_mean_loss, dim=0)
                shifted_cumsum = torch.cat([torch.tensor([0.0], device=x.device, dtype=x.dtype), cumsum_loss[:-1]])

                # FIX: Normalize the cumulative loss to a [0, 1] range to prevent exponential blowup
                max_cumsum = torch.max(shifted_cumsum) + 1e-8
                normalized_cumsum = shifted_cumsum / max_cumsum

                # FIX: Calculate causal weights and strictly clamp them to prevent 0.0 underflow
                W_bin = torch.exp(-self.epsilon * normalized_cumsum)
                W_bin = torch.clamp(W_bin, min=1e-8, max=1.0)
                
                # Map bin weights back to individual points
                W_point = W_bin[bin_idx].unsqueeze(1)
                
            # Apply causal weights to the temporal PDEs
            loss_solid = loss_solid * W_point
            loss_electrolyte = loss_electrolyte * W_point

        # =======================================================
        # Return all coupled physics constraints
        # =======================================================
        # DeepXDE will automatically enforce that all three of these equal 0
        return [loss_solid, loss_electrolyte, loss_kinetics]

class IonPINNNetwork(nn.Module):
    """
    GRU Hybrid architecture for IonPINN.
    Uses a small GRU to process time-series load data (Voltage, Current, Temperature).
    The final hidden state is fed into a tiny FNN (with SiLU activation) to output the final PDE predictions.
    """
    def __init__(self, gru_in_dim=3, gru_hidden=32, fnn_hidden=32, out_dim=5):
        super().__init__()
        # GRU for time-series data
        self.gru = nn.GRU(input_size=gru_in_dim, hidden_size=gru_hidden, batch_first=True)
        
        # Tiny FNN/MLP with SiLU (Swish) activation
        self.fnn = nn.Sequential(
            nn.Linear(gru_hidden, fnn_hidden),
            nn.SiLU(),
            nn.Linear(fnn_hidden, fnn_hidden),
            nn.SiLU(),
            nn.Linear(fnn_hidden, out_dim)
        )
        self._transform = None # Placeholder for DeepXDE output transformations
        
    def forward(self, x):
        inputs = x # Store original DeepXDE inputs for the output transform

        # x is expected to have sequence dimension (batch, seq_len, gru_in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        gru_out, _ = self.gru(x)
        
        # Extract the final hidden state
        final_state = gru_out[:, -1, :]
        
        y = self.fnn(final_state)

        # Apply the DeepXDE output transform if it has been defined
        if self._transform is not None:
            y = self._transform(inputs, y)

        return y

    def apply_output_transform(self, transform):
        """Allows DeepXDE to inject the hard initial condition transform."""
        self._transform = transform

def build_pinn_model(large=False):
    """
    Builds the DeepXDE Model that combines the empirical data and the SPMe physics.
    """
    # 1. Define Geometry and Time domain
    geom = dde.geometry.Hypercube([0, 0], [1, 1]) # Spatial domain normalized [r, x_cell]
    timedomain = dde.geometry.TimeDomain(0, 1) # Normalized time
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # 2. Define Physics
    spme = SPMePhysics()
    
    # Boundary and Initial Conditions (Placeholders for foundation model)
    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0)
        
    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[1], 1)
        
    bc_l = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_l, component=0)
    
    # We would add the experimental data from Dataset 5 & 11 as PointSetBC here
    data = dde.data.TimePDE(
        geomtime,
        spme.pde,
        [bc_l],
        num_domain=1000,
        num_boundary=100,
        num_initial=100,
        num_test=500  # Added to compute and display test metrics
    )
    
    # Define DeepXDE Network (FNN)
    if large:
        net = dde.nn.FNN([3] + [128, 128, 128, 128] + [5], "swish", "Glorot normal")
    else:
        # Shrunk from [64, 64, 64] down to [32, 32] and swapped Tanh for Swish (SiLU)
        net = IonPINNNetwork(gru_in_dim=3, gru_hidden=32, fnn_hidden=32, out_dim=5)
    
    def output_transform(inputs, outputs):
        t = inputs[:, 0:1]
        c_s = outputs[:, 0:1]
        other_states = outputs[:, 1:]
        # Hard enforce c_s = 0.5 at t=0
        c_s_new = 0.5 + (1 - torch.exp(-t)) * c_s
        return torch.cat([c_s_new, other_states], dim=1)
        
    net.apply_output_transform(output_transform)
    
    model = dde.Model(data, net)
    return model, spme

