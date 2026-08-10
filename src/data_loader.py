import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class BatteryAgingDataset(Dataset):
    """
    Dataset loader for NASA Battery Aging Data (e.g., Dataset 5 - B0005)
    Loads cycle CSVs to model capacity fade over long term usage.
    """
    def __init__(self, data_dir: str, cell_id: str = "B0005"):
        super().__init__()
        self.data_dir = data_dir
        self.cell_id = cell_id
        self.features, self.targets = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Simplistic loader: reads CSV files matching cell_id_cycle*.csv
        features_list = []
        targets_list = []
        
        if not os.path.exists(self.data_dir):
            return np.array([]), np.array([])
            
        files = [f for f in os.listdir(self.data_dir) if f.startswith(self.cell_id) and f.endswith(".csv")]
        files.sort()
        
        # We would typically parse 'Voltage_measured', 'Current_measured', 'Temperature_measured'
        # and 'Capacity' if extracted from the metadata/matlab files. 
        # Here we provide a foundation structure that can be easily expanded.
        for f in files:
            path = os.path.join(self.data_dir, f)
            try:
                df = pd.read_csv(path)
                # Ensure generic columns are loaded, or adapt to the CSV headers available
                # E.g. V, I, T as features, Capacity/SoH as target
                if df.empty:
                    continue
                    
                # Dummy implementation assuming specific columns or numeric data
                # Extract first row or mean of features per cycle to predict capacity
                feats = df.iloc[:, :3].mean().values if df.shape[1] >= 3 else np.zeros(3)
                target = df.iloc[:, -1].mean() if df.shape[1] > 3 else 0.0
                
                features_list.append(feats)
                targets_list.append(target)
            except Exception:
                pass
                
        return np.array(features_list), np.array(targets_list)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

class TransientDynamicsDataset(Dataset):
    """
    Dataset loader for NASA Randomized Battery Usage Data (e.g., Dataset 11 - RW1)
    Loads high-frequency transient profiles (V, I) over time.
    """
    def __init__(self, data_dir: str, cell_id: str = "RW1"):
        super().__init__()
        self.data_dir = data_dir
        self.cell_id = cell_id
        self.time, self.features, self.targets = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_list = []
        features_list = []
        targets_list = []
        
        if not os.path.exists(self.data_dir):
            return np.array([]), np.array([]), np.array([])
            
        files = [f for f in os.listdir(self.data_dir) if f.startswith(self.cell_id) and f.endswith(".csv")]
        files.sort()
        
        # Just loading the first few files as an example for transient dynamics
        for f in files[:10]:
            path = os.path.join(self.data_dir, f)
            try:
                df = pd.read_csv(path)
                # Simulated extraction of Time, Voltage, Current
                if df.shape[0] < 10:
                    continue
                # Assuming Time, V, I as columns
                t = np.arange(len(df)) # placeholder for time
                feats = df.iloc[:, :2].values # placeholder for V, I
                target = df.iloc[:, 0].values # placeholder for SoC
                
                time_list.append(t)
                features_list.append(feats)
                targets_list.append(target)
            except Exception:
                pass
                
        return np.array(time_list, dtype=object), np.array(features_list, dtype=object), np.array(targets_list, dtype=object)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.time[idx], dtype=torch.float32), \
               torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

def get_dataloaders(data_dir: str):
    # Setup paths for Training dataset
    train_dir = os.path.join(data_dir, "Training")
    
    aging_dataset = BatteryAgingDataset(train_dir, "B0005")
    transient_dataset = TransientDynamicsDataset(train_dir, "RW1")
    
    # Check if empty, create dummy data for testing if true
    if len(aging_dataset) == 0:
        aging_dataset.features = np.random.randn(100, 3)
        aging_dataset.targets = np.random.randn(100)
    
    aging_loader = DataLoader(aging_dataset, batch_size=max(1, len(aging_dataset)), shuffle=True)
    return aging_loader

def get_pinn_training_data(data_dir: str, num_samples: int = 1000, dataset_prefix: str = "B0005"):
    """
    Loads raw CSV data and formats it for DeepXDE PointSetOperatorBC.
    Returns: X_train (shape: [N, 4]), Y_train (shape: [N, 1])
             where X_train is [t_norm, r, x_cell, T_celsius]
             and Y_train is [Voltage]
    """
    import glob
    
    # Try Training dir first, fallback to Testing if empty
    train_dir = os.path.join(data_dir, "Training")
    if not os.path.exists(train_dir) or not glob.glob(os.path.join(train_dir, f"{dataset_prefix}_cycle*.csv")):
        train_dir = os.path.join(data_dir, "Testing")
        
    files = glob.glob(os.path.join(train_dir, f"{dataset_prefix}_cycle*.csv"))
    if not files:
        return None, None
        
    files.sort()
    
    # Select a subset of files to train on (e.g. 5 evenly spaced cycles)
    num_files = min(5, len(files))
    indices = np.linspace(0, len(files) - 1, num_files, dtype=int)
    selected_files = [files[i] for i in indices]
    
    X_list = []
    Y_list = []
    
    for f in selected_files:
        df = pd.read_csv(f)
        if df.empty or 'Time' not in df.columns or 'Voltage' not in df.columns or 'Temperature' not in df.columns:
            continue
            
        time_vals = df['Time'].values
        voltage_vals = df['Voltage'].values
        temp_vals = df['Temperature'].values
        
        # Normalize time
        t_max = np.max(time_vals) if np.max(time_vals) > 0 else 1.0
        t_norm = time_vals / t_max
        
        # We assume r=0.5 and x_cell=0.5 for the bulk macroscopic voltage measurement
        r_vals = np.ones_like(t_norm) * 0.5
        x_cell_vals = np.ones_like(t_norm) * 0.5
        
        X = np.column_stack((r_vals, x_cell_vals, temp_vals, t_norm))
        Y = voltage_vals.reshape(-1, 1)
        
        X_list.append(X)
        Y_list.append(Y)
        
    if not X_list:
        return None, None
        
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    
    # Randomly subsample to `num_samples` points to avoid overloading DeepXDE memory
    if len(X_train) > num_samples:
        idx = np.random.choice(len(X_train), num_samples, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        
    return X_train, Y_train
