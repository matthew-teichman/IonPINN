# IonPINN SPMe Model Training Strategies & Architectural Notes

## Executive Summary
This document logs the core optimization strategies, architectural decisions, and physical constraints developed for training the Single Particle Model with Electrolyte (SPMe) Physics-Informed Neural Network (`IonPINN`)[cite: 3].

The goal of these strategies is to overcome severe gradient stiffness, loss plateaus, and physical causality violations in coupled electrochemical partial differential equations (PDEs), driving PDE residual errors down to target tolerances ($\le 10^{-4}$) while maintaining a lightweight footprint suitable for real-time embedded execution[cite: 3].

---

## 1. Governing Equation & State Reformulations

### 1.1 Non-Dimensionalization
* **Problem:** Raw dimensional parameters span extreme scales ($D_s \approx 10^{-14} \, \text{m}^2/\text{s}$, Faraday constant $F \approx 96485 \, \text{C/mol}$), causing massive numerical stiffness in AutoDiff gradients[cite: 3].
* **Solution:** Rescaled all spatiotemporal inputs and dependent state outputs to $\mathcal{O}(1)$ scales[cite: 3]:
  * **Time:** Rescaled by the characteristic diffusion timescale: $\tilde{t} = \frac{D_s}{R_s^2} t$[cite: 3].
  * **Concentration:** Normalized by maximum concentration $c_{s,\max}$ so that $\tilde{c}_s \in [0, 1]$[cite: 3].
  * **Overpotential:** Normalized overpotential by thermal voltage: $\tilde{\eta} = \frac{F}{RT} \eta$[cite: 3].
* **Impact:** Eliminates raw physical constants from the derivative multipliers, preventing exponential loss explosion in the Butler-Volmer kinetics equation[cite: 3].

---

## 2. Loss Landscape & Optimization Engineering

### 2.1 Static Loss Weighting vs. Dynamic Weighting
* **Problem:** The Butler-Volmer kinetics residual started at magnitudes of $10^8$, completely swamping gradients for solid diffusion and boundary conditions[cite: 3]. Conversely, automated dynamic weighting (e.g., GradNorm / custom weight updaters) caused scaling collapse and optimizer paralysis[cite: 3].
* **Solution:** Implemented manual static loss weights[cite: 3]:
  $$\text{loss\_weights} = [1.0,\, 1.0,\, 10^{-4},\, 50.0]$$
  * `PDE Solid` ($1.0$): Mass conservation in the active particle[cite: 3].
  * `PDE Elec` ($1.0$): Electrolyte dynamics[cite: 3].
  * `Kinetics` ($10^{-4}$): Dampens the exponential magnitude of the Butler-Volmer reaction equation[cite: 3].
  * `BC Left` ($50.0$): Strictly forces adherence to flux boundary conditions[cite: 3].

### 2.2 Two-Stage Hybrid Training Loop (Adam $\rightarrow$ L-BFGS)
* **Stage 1 (Adam Phase):** First-order optimization run for 10,000–20,000 epochs with step/inverse-time learning rate decay ($10^{-3} \to 10^{-5}$)[cite: 3]. Carves out the general loss landscape and guides parameters into the global minimum neighborhood without overshooting[cite: 3].
* **Stage 2 (L-BFGS Phase):** Quasi-Newton second-order optimization using line-search and Hessian approximations (`ftol=1e-12`, `gtol=1e-12`)[cite: 3]. Runs without explicit epoch caps until convergence tolerances are triggered, driving residual errors from $10^{-2}$ down to $10^{-4}$[cite: 3].

### 2.3 Full-Batch Training Requirement
* **Rule:** Mini-batching (`batch_size=32/64`) is **strictly prohibited**[cite: 3].
* **Rationale:** L-BFGS requires a deterministic loss landscape to evaluate function curvature[cite: 3]. Mini-batching introduces stochastic noise between steps, breaking the line search and causing premature termination or "line search failed" errors[cite: 3]. The full spatiotemporal domain must be evaluated simultaneously[cite: 3].

---

## 3. Advanced Sampling & Causality Controls

### 3.1 Residual-Based Adaptive Refinement (RAR / Point Resampling)
* **Implementation:** `dde.callbacks.PDEPointResampler(period=2000)`[cite: 3].
* **Mechanism:** Dynamically redistributes spatial collocation points every 2,000 steps during the Adam phase to focus density on high-residual regions (e.g., steep concentration gradients at the particle boundary $r = R_s$)[cite: 3].
* **Constraint:** Disabled prior to Stage 2 (L-BFGS) to maintain a static loss landscape for second-order gradient calculation[cite: 3].


---

## 4. Architectural Benchmarking Strategy

To validate model efficiency for embedded microcontroller deployment, two architectures are trained and evaluated in parallel to conduct a strict ablation study on parameter capacity[cite: 3]. Both models utilize a custom `IonPINNNetwork` class (wrapped natively for DeepXDE compatibility via an `apply_output_transform` method).

| Parameter / Feature | Embedded Model (Target) | Large Baseline Model |
| :--- | :--- | :--- |
| **Architecture** | FNN [3, 32, 32, 5] | FNN [3, 128, 128, 128, 128, 5] |
| **Primary Metric** | Residual Accuracy ($\le 10^{-4}$) vs. Memory/MAC Footprint[cite: 3] | Theoretical Loss Floor & Parameter Capacity Test |

---

## 5. Capacity Enhancement Strategies (Small Model)

To bridge the performance gap between large parameter-heavy baselines and embedded-target networks, specific techniques are employed to increase parameter efficiency without increasing MACs significantly:

### 5.1 Fourier Feature Encoding (Input Transform)
* **Problem:** Small fully-connected networks suffer from spectral bias and struggle to learn sharp, high-frequency physical dynamics, leading to large residual errors.
* **Solution:** Input coordinates $x = [r, x_{cell}, T, t]$ are mapped into a higher-dimensional space using sinusoidal functions before entering the network: $\hat{x} = [x, \sin(\pi x), \cos(\pi x)]$.
* **Impact:** Drastically improves the expressiveness of the model, enabling the $[32, 32]$ architecture to capture non-linear PDE manifolds.

### 5.2 Hard-Constrained Boundary / Initial Conditions
* **Problem:** Enforcing initial conditions (e.g., $c_s = 0.5$ at $t=0$) via soft loss penalties (e.g., `IC c_s` with weight `50.0`) consumes valuable capacity in small networks and creates conflicting gradients.
* **Solution:** Apply an output transform to the network that mathematically forces the condition to hold true by definition. For example, $c_s(t) = 0.5 + t \cdot \text{NN}_0(x,t)$. 
* **Impact:** Eliminates the IC loss term entirely, removing gradient conflicts and freeing up the network's capacity to focus exclusively on resolving the PDEs.

### 5.3 Knowledge Distillation (Teacher-Student Transfer Learning)
* **Problem:** Even with architectural enhancements, discovering the complex loss manifold representing coupled electrochemical PDEs is exceedingly difficult for a small network from scratch.
* **Solution:** Train the overparameterized baseline model (Teacher) first. Then, sample a dense synthetic dataset of collocation points (e.g., 20,000 points) and evaluate them using the Teacher to produce continuous, high-fidelity state targets. The embedded model (Student) is then trained to minimize the MSE between its predictions and the Teacher's synthetic dataset (via `PointSetBC`), alongside the standard PDE residuals.
* **Impact:** The Teacher "smooths" the highly-coupled, noisy PDE physics into a direct data-fitting objective for the Student. The Student can easily regress to the smooth Teacher manifold, achieving convergence and accuracy that it could never attain by solving the raw PDEs alone.