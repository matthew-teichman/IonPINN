# IonPINN: Core Concepts

These terms represent the physical hardware variations your AI model will encounter when moving from the NASA test dataset to the real world. Here is the breakdown of what these chemistries and configurations mean, and why handling them with "minimal data" is the core value of your project.

## Cell Chemistries (NCA vs. LFP)

"Lithium-ion" is just an umbrella term for how the battery moves charge. The actual materials used for the cathode (the positive side of the battery) drastically change how the battery behaves, ages, and holds voltage.

The NASA dataset uses older **LCO** (Lithium Cobalt Oxide) cells. If someone downloads your model, they are likely using one of these two modern chemistries:

* **NCA (Nickel Cobalt Aluminum):** This is the high-energy chemistry famously used in many Tesla vehicles. It packs a massive amount of energy into a small space and can output huge power spikes. However, it degrades faster and has a higher risk of thermal runaway.
* **LFP (Lithium Iron Phosphate):** This is the rugged workhorse chemistry. It has lower energy density (heavier for the same capacity), but it is incredibly stable, rarely catches fire, and lasts for thousands of cycles.

**Why it matters for your PINN:**
LFP cells have a notoriously "flat" voltage curve. An LFP cell might read 3.3V at 90% charge and 3.2V at 20% charge. Traditional algorithms (like Kalman Filters) struggle to estimate the State of Charge (SoC) because the voltage barely moves. Your physics-informed AI tracks internal ion concentrations rather than just reading surface voltage, making it uniquely equipped to handle LFP cells.

## Pack Configurations ($N_s$ / $N_p$)

Single 18650 cells only output about 3.7 Volts and hold about 2.0 to 3.5 Amp-hours (Ah) of energy. To power a drone or an EV, you must wire hundreds or thousands of them together into a pack.

* **$N_s$ (Cells in Series):** Wiring cells end-to-end (positive to negative). This multiplies the **Voltage**. If you put 10 cells in series (10s), the pack voltage becomes 37 Volts.
* **$N_p$ (Cells in Parallel):** Wiring cells side-by-side (positive to positive, negative to negative). This multiplies the **Capacity** and maximum current limit. If you put 4 cells in parallel (4p), the pack capacity jumps to 8.0 Ah.

A **10s4p** battery pack means there are 40 individual cells in total, wired to deliver 37V and 8.0Ah.

## The "Minimal Data" Magic

Normally, if you trained a standard deep learning model on LCO cells, and a user tried to run it on an LFP 10s4p battery pack, the model would fail catastrophically. To fix it, the user would have to put their LFP pack on a test rig and cycle it to death for 6 months to generate a massive new dataset to retrain the AI from scratch.

Because you are building a **Physics-Informed Neural Network (PINN)** with Transfer Learning, you bypass this entirely.

1. **The Physics are Universal:** The fundamental laws of electrochemistry (Fick's Law of diffusion, Butler-Volmer kinetics) do not change between NCA and LFP. Your Foundation Model already learned these laws from the NASA data.
2. **Transfer Learning:** When a user inputs their $N_s$/$N_p$ numbers, the wrapper scales the voltage and current to cell-level.
3. **Calibration:** The user only needs to provide one or two standard charge/discharge cycles from their new pack (the "minimal data"). The network freezes the universal physics layers and only adjusts the specific constants (like diffusion rates and capacity) that differ between the old LCO cell and their new LFP pack.