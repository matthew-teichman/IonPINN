import torch
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
        self.i_0 = 1.0

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

        # Extract coordinates from DeepXDE's [r, x_cell, T, t] format
        r = x[:, 0:1]       # Microscale: inside the particle
        x_cell = x[:, 1:2]  # Macroscale: across the electrode
        T_kelvin = x[:, 2:3] + 273.15 # Temperature (convert Celsius to Kelvin)
        t = x[:, 3:4]       # Time

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
        
        # Temperature already extracted above
        
        # Normalize eta by the thermal voltage (F / RT)
        eta_norm = eta * (self.F / (self.R * T_kelvin))
        
        # Butler-Volmer equation
        term_a = torch.exp(self.alpha_a * eta_norm)
        term_c = torch.exp(-self.alpha_c * eta_norm)
        
        loss_kinetics = j_Li - self.i_0 * (term_a - term_c)

        # =======================================================
        # Return all coupled physics constraints
        # =======================================================
        # DeepXDE will automatically enforce that all three of these equal 0
        return [loss_solid, loss_electrolyte, loss_kinetics]

def build_pinn_model(X_train=None, Y_train=None, large=False):
    """
    Builds the DeepXDE Model that combines the empirical data and the SPMe physics.
    """
    # 1. Define Geometry and Time domain
    geom = dde.geometry.Hypercube([0, 0, 0], [1, 1, 100]) # Spatial domain [r, x_cell, T_celsius]
    timedomain = dde.geometry.TimeDomain(0, 1) # Normalized time
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # 2. Define Physics
    spme = SPMePhysics()
    
    # Boundary and Initial Conditions
    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0)
        
    bc_l = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_l, component=0)
    
    # Soft Initial Condition requiring c_s to start at 0.5
    ic_c_s = dde.icbc.IC(geomtime, lambda x: 0.5, lambda _, on_initial: on_initial, component=0)
    
    bcs = [bc_l, ic_c_s]
    
    # Add Experimental Data (Voltage)
    if X_train is not None and Y_train is not None:
        def voltage_op(inputs, outputs, X):
            # outputs are [c_s, c_e, phi_s, phi_e, j_Li]
            return outputs[:, 2:3] - outputs[:, 3:4]
            
        bc_voltage = dde.icbc.PointSetOperatorBC(X_train, Y_train, voltage_op)
        bcs.append(bc_voltage)
    
    data = dde.data.TimePDE(
        geomtime,
        spme.pde,
        bcs,
        num_domain=1000,
        num_boundary=100,
        num_initial=100,
        num_test=500
    )
    
    # Define DeepXDE Network (FNN)
    if large:
        net = dde.nn.FNN([4] + [128, 128, 128, 128] + [5], "swish", "Glorot normal")
    else:
        net = dde.nn.FNN([4] + [32, 32] + [5], "swish", "Glorot normal")
    
    model = dde.Model(data, net)
    return model, spme

