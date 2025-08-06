import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
from pytorch3dunet.unet3d import model
from load_data import *
from CurrentCBAMNet import *
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


class PreloadedDatasetWithCurrent(torch.utils.data.Dataset):
    def __init__(self, npy_dir, csv_file, total_samples):
        self.npy_dir = npy_dir
        self.total_samples = total_samples

        df = pd.read_csv(csv_file)
        self.current_values = df["K1_value"].values[:total_samples]

        self.inputs = []
        self.currents = []
        for idx in range(total_samples):
            occ = np.load(os.path.join(npy_dir, f'occ_{idx}.npy'))
            input_tensor = np.stack([occ], axis=0)
            self.inputs.append(torch.from_numpy(input_tensor).float())
            self.currents.append(torch.tensor(self.current_values[idx], dtype=torch.float32))

        print(f"Loaded {self.total_samples} samples into RAM.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.currents[idx]

def train_model(model, train_loader, val_loader, output_prefix, optimizer, scheduler, criterion, epochs=100):
    scaler = GradScaler()
    best_val_loss = float('inf')

    log_path = output_prefix + "_loss.txt"
    with open(log_path, "w") as log_file:
        for epoch in range(1, epochs+1):
            # — Training —
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            scheduler.step()

            # — Validation —
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    val_loss += criterion(model(inputs), targets).item()
            val_loss /= len(val_loader)

            # Log
            log_file.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")
            print(f"Fold [{output_prefix}] Epoch {epoch} — train: {train_loss:.6f}, val: {val_loss:.6f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_prefix + "_best.pth")

        # End of epochs: save last
        torch.save(model.state_dict(), output_prefix + "_last.pth")

if __name__ == "__main__":
    # Hyperparameters
    epochs = 65
    lr = 1e-3
    batch_size = 32
    weight_decay = 1e-5
    n_splits = 5
    random_seed = 42

    # Data paths
    npy_dir = '../Jz_training/npy_preprocessed2/'
    csv_file = "40000_sample_2.csv"
    total_samples = 20000

    # Load dataset
    full_dataset = PreloadedDatasetWithCurrent(npy_dir, csv_file, total_samples)
    indices = np.arange(total_samples)

    # Prepare K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        # Create fold directory
        fold_dir = f"fold_{fold}_test"
        os.makedirs(fold_dir, exist_ok=True)

        # Save split indices
        np.savetxt(os.path.join(fold_dir, "train_idx.txt"), train_idx, fmt="%d")
        np.savetxt(os.path.join(fold_dir, "val_idx.txt"), val_idx, fmt="%d")

        # DataLoaders
        train_loader = DataLoader(Subset(full_dataset, train_idx),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(Subset(full_dataset, val_idx),
                                batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        # Model / Optimizer / Scheduler setup
        model = GlobalContextPredictor(1).cuda()


        # CALCULATE AND PRINT PARAMETER COUNT
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = count_parameters(model)
        print(f"\n=== Model Information ===")
        print(f"Model type: {model.__class__.__name__}")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Approximate size: {total_params / 1e6:.2f} million parameters")



        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        criterion = nn.L1Loss()

        # Train
        prefix = os.path.join(fold_dir, f"model_fold{fold}")
        train_model(model, train_loader, val_loader, prefix,
                    optimizer, scheduler, criterion, epochs)

    print("Cross‐validation complete.")

