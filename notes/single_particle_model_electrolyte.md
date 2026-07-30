# IonPINN: Physics-Informed Neural Network for Battery Modeling

This repository contains the implementation of **IonPINN**, a Physics-Informed Neural Network (PINN) designed for battery state estimation and physics modeling.

## Overview

IonPINN utilizes the **Single Particle Model with Electrolyte (SPMe)** for its governing equations.

While the Doyle-Fuller-Newman (DFN) or Pseudo-2D (P2D) model is considered the "gold standard" of battery physics, it involves a massive system of coupled nonlinear Partial Differential Equations (PDEs). Embedding the full DFN model into a PINN's loss function severely hampers convergence, extending training time to days.

The SPMe provides the perfect mathematical compromise:
- It simplifies the geometry by assuming each electrode (anode and cathode) acts like a single spherical particle.
- Unlike the basic Single Particle Model (SPM), SPMe retains the equations for **electrolyte ion dynamics**, keeping the model accurate even during dynamic, high-current loads (e.g., target Dataset 11).

## Physics Loss Function ($\mathcal{L}_{physics}$)

IonPINN encodes three primary governing equations into DeepXDE to form the physics-informed loss function. The network is penalized if its predictions violate these physical laws.

### 1. Solid-Phase Diffusion (Fick's Second Law)
Describes how Lithium ions diffuse inside the solid spherical particles of the anode and cathode. This is the primary driver for tracking the State of Charge (SoC).

$$ \frac{\partial c_s}{\partial t} = \frac{D_s}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial c_s}{\partial r} \right) $$

**Variables:**
* $c_s$: Lithium concentration in the solid particle
* $D_s$: Solid-phase diffusion coefficient (degrades over time, correlating with State of Health / SoH)
* $r$: Radial position within the particle

### 2. Electrolyte Dynamics
Describes the concentration of Lithium ions moving through the liquid separator between the two electrodes, preventing the PINN from predicting impossible voltage recoveries during high power draws.

$$ \epsilon_e \frac{\partial c_e}{\partial t} = D_e \frac{\partial^2 c_e}{\partial x^2} + \frac{1-t_+}{F} j^{Li} $$

**Variables:**
* $c_e$: Lithium concentration in the electrolyte
* $\epsilon_e$: Electrolyte porosity
* $D_e$: Electrolyte diffusion coefficient
* $t_+$: Transference number
* $j^{Li}$: Interfacial current density

### 3. Electrochemical Kinetics (Butler-Volmer Equation)
Couples the internal chemistry to the electrical output. It dictates the reaction rate at the boundary where the solid particle meets the liquid electrolyte, determining the measured terminal voltage.

$$ j^{Li} = i_0 \left( \exp \left( \frac{\alpha_a F}{RT} \eta \right) - \exp \left( \frac{-\alpha_c F}{RT} \eta \right) \right) $$

**Variables:**
* $i_0$: Exchange current density
* $\alpha_a, \alpha_c$: Anodic and cathodic transfer coefficients
* $F$: Faraday's constant
* $R$: Universal gas constant
* $T$: Temperature (sourced from NASA dataset temperature sensors)
* $\eta$: Surface overpotential

## PINN Implementation Details

Instead of solving these PDEs using traditional, computationally heavy numerical methods (like finite element analysis), IonPINN leverages **DeepXDE**.

**How it works:**
1. The neural network predicts the voltage and concentration fields.
2. DeepXDE takes these predictions and computes their derivatives using automatic differentiation.
3. The predictions are plugged into the SPMe equations above.
4. The network computes the residual (how far the left side of the equation is from the right side).
5. If the physics are unbalanced, the network is heavily penalized and forced to adjust its weights until the laws of physics are satisfied.
README.md
Displaying README.md.