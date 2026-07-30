# 📊 NASA Battery Datasets Guide

This directory holds the dataset files required for training, evaluating, and running transfer learning experiments with **IonPINN**. 

Due to file size limits and licensing distribution guidelines, the raw dataset files are **excluded from Git tracking** and are not hosted directly on GitHub. 

Follow the instructions below to download the datasets from the official NASA repository and set up your local workspace.

---

## 🔗 Dataset Sources & Citations

The datasets used in this project are published by the **NASA Ames Prognostics Center of Excellence (PCoE)**:

* **Official Repository Link:** [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

### 1. Battery Data Set (Dataset #5)
* **Description:** Run-to-failure degradation experiments conducted on commercial 18650 Lithium-ion cells under constant-current / constant-voltage charging and discharging cycles at various ambient temperatures ($4^\circ\text{C}$, $24^\circ\text{C}$, and $43^\circ\text{C}$).
* **Usage in IonPINN:** Core pre-training dataset for physical baseline models and capacity fade parameter estimation.
* **Reference:** Saha, B. and Goebel, K. (2007). *"Battery Data Set"*, NASA Ames Prognostics Center of Excellence (PCoE), NASA Ames Research Center, Moffett Field, CA.

### 2. Randomized Battery Usage Data Set (Dataset #11)
* **Description:** Continuous cycling datasets where Li-ion batteries were subjected to randomized current loading profiles ("random walk" operations) punctuated by periodic reference charge/discharge benchmark cycles.
* **Usage in IonPINN:** Evaluation of model generalizability under dynamic, non-standard power draws and variable duty cycles.
* **Reference:** Bole, B., Kulkarni, C., and Daigle, M. (2014). *"Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for Deterioration Observed Under Randomized Use"*, Annual Conference of the Prognostics and Health Management Society.

---

## 📥 Step-by-Step Setup Instructions

### Step 1: Download from NASA
1. Open the [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Locate and download the following dataset archives:
   - **5. Battery Data Set**
   - **11. Randomized Battery Usage Data Set**

### Step 2: Organize the Local `datasets/` Folder
Unpack or place the downloaded zip files into the `datasets` folder following this exact directory structure:

```
datasets/
├── README.md
├── 5.+Battery+Data+Set/
│   └── 5. Battery Data Set/
│       ├── 1. BatteryAgingARC-FY08Q4.zip
│       ├── 2. BatteryAgingARC_25_26_27_28_P1.zip
│       ├── 3. BatteryAgingARC_25-44.zip
│       ├── 4. BatteryAgingARC_45_46_47_48.zip
│       ├── 5. BatteryAgingARC_49_50_51_52.zip
│       └── 6. BatteryAgingARC_53_54_55_56.zip
└── 11.+Randomized+Battery+Usage+Data+Set/
    └── 11. Randomized Battery Usage Data Set/
        ├── 1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip
        ├── 2. Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip
        ├── 3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip
        ├── 4. RW_Skewed_High_40C_DataSet_2Post.zip
        ├── 5. RW_Skewed_High_Room_Temp_DataSet_2Post.zip
        ├── 6. RW_Skewed_Low_40C_DataSet_2Post.zip
        └── 7. RW_Skewed_Low_Room_Temp_DataSet_2Post.zip
```

---

## 🔒 Version Control & Distribution Notice

All raw dataset files (`.zip`, `.mat`, `.csv`, `.h5`) located within `datasets/` are ignored by `.gitignore` to prevent accidental commits to GitHub. Only this `README.md` documentation file is tracked in repository history.
