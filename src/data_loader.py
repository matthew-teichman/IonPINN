import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

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
            except Exception as e:
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
            except Exception as e:
                pass
                
        return np.array(time_list, dtype=object), np.array(features_list, dtype=object), np.array(targets_list, dtype=object)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.time[idx], dtype=torch.float32), \
               torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

def get_dataloaders(data_dir: str, batch_size: int = 32):
    # Setup paths for Training dataset
    train_dir = os.path.join(data_dir, "Training")
    
    aging_dataset = BatteryAgingDataset(train_dir, "B0005")
    transient_dataset = TransientDynamicsDataset(train_dir, "RW1")
    
    # Check if empty, create dummy data for testing if true
    if len(aging_dataset) == 0:
        aging_dataset.features = np.random.randn(100, 3)
        aging_dataset.targets = np.random.randn(100)
    
    aging_loader = DataLoader(aging_dataset, batch_size=batch_size, shuffle=True)
    return aging_loader
