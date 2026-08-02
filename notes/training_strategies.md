# IonPINN SPMe Model Training Strategies & Architectural Notes

## Executive Summary
This document logs the core optimization strategies, architectural decisions, and physical constraints developed for training the Single Particle Model with Electrolyte (SPMe) Physics-Informed Neural Network (`IonPINN`).

The goal of these strategies is to overcome severe gradient stiffness, loss plateaus, and physical causality violations in coupled electrochemical partial differential equations (PDEs), driving PDE residual errors down to target tolerances ($\le 10^{-4}$) while maintaining a lightweight footprint suitable for real-time embedded execution.

---

## 1. Governing Equation & State Reformulations

### 1.1 Non-Dimensionalization
* **Problem:** Raw dimensional parameters span extreme scales ($D_s \approx 10^{-14} \, \text{m}^2/\text{s}$, Faraday constant $F \approx 96485 \, \text{C/mol}$), causing massive numerical stiffness in AutoDiff gradients.
* **Solution:** Rescaled all spatiotemporal inputs and dependent state outputs to $\mathcal{O}(1)$ scales:
  * **Time:** Rescaled by the characteristic diffusion timescale: $\tilde{t} = \frac{D_s}{R_s^2} t$.
  * **Concentration:** Normalized by maximum concentration $c_{s,\max}$ so that $\tilde{c}_s \in [0, 1]$.
  * **Overpotential:** Normalized overpotential by thermal voltage: $\tilde{\eta} = \frac{F}{RT} \eta$.
* **Impact:** Eliminates raw physical constants from the derivative multipliers, preventing exponential loss explosion in the Butler-Volmer kinetics equation.

### 1.2 Hard Initial Conditions (Output Transformations)
* **Problem:** Enforcing initial conditions ($c_s(r, x, 0) = 0.5$) via soft loss penalties caused the initial condition residual to flatline at $\sim 2.0$, forcing the network to waste capacity trying to learn $t=0$.
* **Solution:** Structurally enforced hard initial conditions on $c_s$ using an output transformation:
  $$y_{\text{final}}(x, t) = y_0(x) + (1 - e^{-t}) \cdot \text{NN}(x, t)$$
* **Targeting:** Applied **strictly to component 0 ($c_s$)**. Unconstrained algebraic states ($\phi_s, \phi_e, j_{\text{Li}}$) were left free to allow Butler-Volmer kinetics to self-balance at $t=0$ without creating mathematical contradictions.

---

## 2. Loss Landscape & Optimization Engineering

### 2.1 Static Loss Weighting vs. Dynamic Weighting
* **Problem:** The Butler-Volmer kinetics residual started at magnitudes of $10^8$, completely swamping gradients for solid diffusion and boundary conditions. Conversely, automated dynamic weighting (e.g., GradNorm / custom weight updaters) caused scaling collapse and optimizer paralysis.
* **Solution:** Implemented manual static loss weights:
  $$\text{loss\_weights} = [1.0,\, 1.0,\, 10^{-4},\, 50.0]$$
  * `PDE Solid` ($1.0$): Mass conservation in the active particle.
  * `PDE Elec` ($1.0$): Electrolyte dynamics.
  * `Kinetics` ($10^{-4}$): Dampens the exponential magnitude of the Butler-Volmer reaction equation.
  * `BC Left` ($50.0$): Strictly forces adherence to flux boundary conditions.

### 2.2 Two-Stage Hybrid Training Loop (Adam $\rightarrow$ L-BFGS)
* **Stage 1 (Adam Phase):** First-order optimization run for 10,000–20,000 epochs with step/inverse-time learning rate decay ($10^{-3} \to 10^{-5}$). Carves out the general loss landscape and guides parameters into the global minimum neighborhood without overshooting.
* **Stage 2 (L-BFGS Phase):** Quasi-Newton second-order optimization using line-search and Hessian approximations (`ftol=1e-12`, `gtol=1e-12`). Runs without explicit epoch caps until convergence tolerances are triggered, driving residual errors from $10^{-2}$ down to $10^{-4}$.

### 2.3 Full-Batch Training Requirement
* **Rule:** Mini-batching (`batch_size=32/64`) is **strictly prohibited**.
* **Rationale:** L-BFGS requires a deterministic loss landscape to evaluate function curvature. Mini-batching introduces stochastic noise between steps, breaking the line search and causing premature termination or "line search failed" errors. The full spatiotemporal domain must be evaluated simultaneously.

---

## 3. Advanced Sampling & Causality Controls

### 3.1 Residual-Based Adaptive Refinement (RAR / Point Resampling)
* **Implementation:** `dde.callbacks.PDEPointResampler(period=2000)`
* **Mechanism:** Dynamically redistributes spatial collocation points every 2,000 steps during the Adam phase to focus density on high-residual regions (e.g., steep concentration gradients at the particle boundary $r = R_s$).
* **Constraint:** Disabled prior to Stage 2 (L-BFGS) to maintain a static loss landscape for second-order gradient calculation.

### 3.2 Continuous Causal Training (Time-Marching)
* **Problem:** Evaluating all time steps $t \in [0, 1]$ simultaneously causes future state errors ($t=0.9$) to backpropagate and corrupt early gradients ($t=0.1$), locking `PDE Solid` loss at $10^{-1}$.
* **Solution:** Weighted temporal residuals sequentially:
  $$w_i = \exp\left(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}_k\right)$$
* **Behavior:** Mathematically forces the network to master early time steps ($t \to 0$) before allowing future states to influence gradient updates. Causal weighting is active during Adam and disabled right before L-BFGS handoff.

---

## 4. Architectural Benchmarking Strategy

To validate model efficiency for embedded microcontroller deployment, two architectures are trained and evaluated in parallel:

| Parameter / Feature | Embedded Model (Target) | Large Baseline Model |
| :--- | :--- | :--- |
| **Architecture** | `[32, 32]` FNN / GRU Hybrid | `[128, 128, 128, 128]` FNN |
| **Causal Training** | Enabled (Adam Phase) | Disabled (Standard Baseline) |
| **Hard IC Transform** | Applied ($c_s$ only) | Soft Loss Penalty / Standard |
| **Primary Metric** | Residual Accuracy ($\le 10^{-4}$) vs. Memory/MAC Footprint | Theoretical Loss Floor Benchmark |