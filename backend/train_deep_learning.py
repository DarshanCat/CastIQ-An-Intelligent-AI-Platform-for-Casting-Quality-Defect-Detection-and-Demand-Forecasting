"""
VSPL AI Platform — Advanced Physics-Based Deep Learning Training Suite
File: backend/train_deep_learning.py
"""
import sys
import io

# Prevent UnicodeEncodeError on Windows standard output
if sys.platform.startswith('win') and (sys.stdout is None or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

BASE = Path(__file__).parent
DATA_DIR = BASE / 'data'
MODEL_DIR = BASE / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(" 🧠 VSPL AI Platform — Large-Scale Physics Deep Learning Pipeline")
print("=" * 65)

large_csv_path = DATA_DIR / 'casting_data_large.csv'

# Ensure physics-derived dataset is generated
if not large_csv_path.exists():
    print("\n[!] Physics-derived dataset not found. Generating now...")
    try:
        from backend.data.generate_physics_data import N as dummy
    except ImportError:
        # If running as standard script, execute generate_physics_data directly
        import subprocess
        subprocess.run([sys.executable, str(DATA_DIR / 'generate_physics_data.py')], check=True)

# ── 1. LOAD LARGE-SCALE INDUSTRIAL DATASET (50,000 ROWS) ──
print(f"\n[1/4] Loading simulated centrifugal casting database...")
df = pd.read_csv(large_csv_path)
print(f"   ✅ Loaded large physics dataset ({df.shape[0]} rows) from: {large_csv_path.name}")

# ── 2. PREPROCESS DATA & PYTORCH DATALOADERS ──
print("\n[2/4] Preprocessing datasets & scaling features...")

# Physics-derived features list (as specified by user)
feats = [
    # Raw inputs
    'carbon_pct', 'silicon_pct', 'manganese_pct', 'phosphorus_pct',
    'magnesium_added_pct', 'treatment_time_min', 'rpm', 'mold_diameter_m',
    'pour_temp_c', 'mold_preheat_c', 'wall_thickness_mm', 'mold_enc',
    # Physics-derived (the secret sauce)
    'carbon_equivalent', 'superheat_c', 'G_factor',
    'Mg_effective_pct', 'nodularity_index', 'cooling_rate_cs'
]

X = df[feats].values
y_score = df['quality_score'].values
y_reject = df['reject'].values

# Train / Test split (80% train, 20% test)
X_train, X_test, y_s_train, y_s_test, y_r_train, y_r_test = train_test_split(
    X, y_score, y_reject, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and features mapping for Streamlit loading
joblib.dump(scaler, MODEL_DIR / 'dnn_scaler.pkl')
joblib.dump(feats, MODEL_DIR / 'dnn_features.pkl')

class CastingDataset(Dataset):
    def __init__(self, X_data, y_score, y_reject):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y_score = torch.tensor(y_score, dtype=torch.float32).unsqueeze(1)
        self.y_reject = torch.tensor(y_reject, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_score[idx], self.y_reject[idx]

train_dataset = CastingDataset(X_train_scaled, y_s_train, y_r_train)
test_dataset = CastingDataset(X_test_scaled, y_s_test, y_r_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# ── 3. DEFINE PYTORCH MULTI-TASK DENSE RESIDUAL NEURAL NETWORK (DNN) ──

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # skip connection!
        return self.elu(out)

class CastingMultitaskDNN(nn.Module):
    def __init__(self, input_dim):
        super(CastingMultitaskDNN, self).__init__()
        # Shared representations
        self.shared_input = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.15)
        )
        self.res1 = ResidualBlock(256)
        self.shared_mid = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.15)
        )
        self.res2 = ResidualBlock(128)
        self.shared_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        
        # Task 1: Quality Score Regressor
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        # Task 2: Reject Classification
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.shared_input(x)
        h = self.res1(h)
        h = self.shared_mid(h)
        h = self.res2(h)
        shared_out = self.shared_out(h)
        score = self.regressor(shared_out)
        prob = self.classifier(shared_out)
        return score, prob

# Instantiate model
model = CastingMultitaskDNN(input_dim=len(feats))

# Loss Functions & Optimizer
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ── 4. TRAINING DEEP LEARNING MODEL ──
print("\n[3/4] Training PyTorch Multi-Task Residual DNN model...")

epochs = 30
train_losses = []
val_scores_mae = []
val_reject_acc = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"   👉 Training on device: {device}")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for inputs, targets_score, targets_reject in train_loader:
        inputs = inputs.to(device)
        targets_score = targets_score.to(device)
        targets_reject = targets_reject.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_scores, pred_rejects = model(inputs)
        
        # Calculate losses
        loss_mse = criterion_mse(pred_scores, targets_score)
        loss_bce = criterion_bce(pred_rejects, targets_reject)
        
        # Total combined loss
        loss = (0.04 * loss_mse) + loss_bce
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Evaluate model
    model.eval()
    val_mae = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, targets_score, targets_reject in test_loader:
            inputs = inputs.to(device)
            targets_score = targets_score.to(device)
            targets_reject = targets_reject.to(device)
            
            pred_scores, pred_rejects = model(inputs)
            
            # MAE
            val_mae += torch.mean(torch.abs(pred_scores - targets_score)).item() * inputs.size(0)
            
            # Accuracy
            preds_binary = (pred_rejects >= 0.5).float()
            val_correct += (preds_binary == targets_reject).sum().item()
            
    epoch_mae = val_mae / len(test_loader.dataset)
    epoch_acc = (val_correct / len(test_loader.dataset)) * 100.0
    
    val_scores_mae.append(epoch_mae)
    val_reject_acc.append(epoch_acc)
    
    scheduler.step(epoch_loss)
    
    print(f"   Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Val MAE: {epoch_mae:.2f} pts | Val Acc: {epoch_acc:.1f}%")

# Save model weights
model_weights_path = MODEL_DIR / 'casting_dnn.pth'
torch.save(model.state_dict(), model_weights_path)
print(f"\n[4/4] Saving Deep Learning model assets...")
print(f"   ✅ Saved PyTorch DNN state dict to: {model_weights_path.name}")

# Generate training diagnostics plot
plt.figure(figsize=(12, 5))

# Plot Combined Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, 'r-o', label='Combined Loss')
plt.title('PyTorch Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True)
plt.legend()

# Plot Validation Metrics
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_reject_acc, 'g-s', label='Classification Acc (%)')
plt.plot(range(1, epochs + 1), val_scores_mae, 'b-d', label='Regressor MAE (pts)')
plt.title('Validation Accuracy & Error Diagnostics')
plt.xlabel('Epochs')
plt.ylabel('Score Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
diag_img_path = MODEL_DIR / 'dnn_training_diagnostics.png'
plt.savefig(diag_img_path)
print(f"   ✅ Saved training loss curves to: {diag_img_path.name}")

print("\n" + "=" * 65)
print("  🎉 DEEP LEARNING MODEL TRAINING SUCCESSFULLY COMPLETED!")
print("=" * 65)
