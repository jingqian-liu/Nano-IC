import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from functools import partial
from gridData import Grid

class DXDataset(Dataset):
    def __init__(self, csv_path, base_dir):
        self.samples = []
        self.base_dir = base_dir
        
        # Pre-load metadata first
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((
                    os.path.join(base_dir, row['charge_dx']),
                    os.path.join(base_dir, row['occ_dx']),
                    float(row['K1_value']) * 2.5
                ))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_sample(charge_path, occ_path):
        """Load DX files in parallel"""
        def load_file(path):
            try:
                return Grid(path).grid.astype(np.float32)
            except:
                return None
        
        with ThreadPoolExecutor(2) as executor:
            charge_future = executor.submit(load_file, charge_path)
            occ_future = executor.submit(load_file, occ_path)
            return charge_future.result(), occ_future.result()

    def __getitem__(self, idx):
        charge_path, occ_path, k1 = self.samples[idx]
        
        # Load data with parallel I/O
        charge_data, occ_data = self.load_sample(charge_path, occ_path)
    

        if charge_data is None or occ_data is None:
            return self[(idx + 1) % len(self)]  # Skip bad samples
            
        input_tensor = torch.stack([
            torch.from_numpy(charge_data),
            torch.from_numpy(occ_data)
        ], dim=0).float()
        
            
        return input_tensor, torch.tensor([k1]).float()


def create_loaders(csv_file, base_dir, batch_size, num_workers=8, seed=42):
    # Create full dataset
    full_dataset = DXDataset(csv_file, base_dir)
    indices = np.arange(len(full_dataset))

    # First split: train vs temp (60% / 40%)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.4, random_state=seed
    )

    # Second split: val vs test (from 40%)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=seed
    )

    # Save indices and sample info
    def save_split_info(indices, split_name):
        with open(f'{split_name}_info.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'GroundTruth', 'Charge_DX_Path', 'Occ_DX_Path'])
            for idx in indices:
                charge_path, occ_path, k1 = full_dataset.samples[idx]
                writer.writerow([idx, f"{k1:.6f}", charge_path, occ_path])

    save_split_info(train_indices, 'train')
    save_split_info(val_indices, 'val')
    save_split_info(test_indices, 'test')

    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create data loaders
    loader_args = {
        'batch_size': batch_size,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'num_workers': num_workers
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    # Print summary
    print(f"\nDataset splits created:")
    print(f"Train samples: {len(train_dataset)} ({len(train_dataset)/len(full_dataset):.1%})")
    print(f"Val samples: {len(val_dataset)} ({len(val_dataset)/len(full_dataset):.1%})")
    print(f"Test samples: {len(test_dataset)} ({len(test_dataset)/len(full_dataset):.1%})")

    return train_loader, val_loader, test_loader

