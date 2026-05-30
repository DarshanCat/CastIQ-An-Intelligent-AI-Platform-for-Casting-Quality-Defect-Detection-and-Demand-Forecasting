"""
VSPL AI Platform — Large Dataset Generation & Deep Learning Training
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
print(" 🧠 VSPL AI Platform — Large-Scale Deep Learning Pipeline")
print("=" * 65)

# ── 1. GENERATE LARGE-SCALE INDUSTRIAL DATASET (50,000 ROWS) ──
print("\n[1/4] Simulating 50,000 high-fidelity casting runs...")

N = 50000

temperature     = np.random.uniform(1340, 1530, N)
rpm             = np.random.uniform(650, 1650, N)
carbon_pct      = np.random.uniform(2.9, 4.1, N)
silicon_pct     = np.random.uniform(1.4, 3.1, N)
cooling_time    = np.random.uniform(20, 140, N)
mold_type       = np.random.choice(['Permanent', 'Sand', 'Die'], N, p=[0.5, 0.3, 0.2])
metal_flow_rate = np.random.uniform(0.4, 2.6, N)

# Complex non-linear physical metallurgy interaction equations
quality = 100.0
# Temperature thermal shocks
quality -= np.abs(temperature - 1425) * 0.28
# Centrifugal spin speed stability
quality -= np.abs(rpm - 1100) * 0.055
# Carbon Equivalent (CE) ratio matching (CE = C + Si/3)
ce = carbon_pct + (silicon_pct / 3.0)
quality -= np.abs(ce - 4.3) * 15.0
# Heat dissipation times per mold type
mold_factor = np.array([{'Permanent': 75.0, 'Sand': 90.0, 'Die': 65.0}[m] for m in mold_type])
quality -= np.abs(cooling_time - mold_factor) * 0.22
# Metal flow rates
quality -= np.abs(metal_flow_rate - 1.5) * 6.5
# Add metallurgical mold bonus and gaussian process noise
mold_bonus = np.array([{'Permanent': 4.0, 'Sand': -1.0, 'Die': 2.0}[m] for m in mold_type])
quality += mold_bonus + np.random.normal(0, 3.5, N)
quality = np.clip(quality, 0.0, 100.0)

# 1 = Reject (Quality < 65), 0 = Good
reject = (quality < 65.0).astype(int)

# Map mold types to integers for PyTorch model input
mold_map = {'Permanent': 0, 'Sand': 1, 'Die': 2}
mold_enc = np.array([mold_map[m] for m in mold_type])

df_large = pd.DataFrame({
    'pour_temp': temperature.round(1),
    'spin_rpm': rpm.round(0).astype(int),
    'carbon_pct': carbon_pct.round(3),
    'silicon_pct': silicon_pct.round(3),
    'cooling_sec': cooling_time.round(1),
    'flow_rate': metal_flow_rate.round(2),
    'mold_type': mold_type,
    'quality_score': quality.round(2),
    'reject': reject
})

large_csv_path = DATA_DIR / 'casting_data_large.csv'
df_large.to_csv(large_csv_path, index=False)
print(f"   ✅ Saved large dataset ({df_large.shape[0]} rows) to: {large_csv_path.name}")

# ── 2. PREPROCESS DATA & PYTORCH DATALOADERS ──
print("\n[2/4] Preprocessing datasets & scaling features...")

# Columns for inputs
feats = ['pour_temp', 'spin_rpm', 'carbon_pct', 'silicon_pct', 'cooling_sec', 'flow_rate', 'mold_enc']
X = np.column_stack([
    temperature, rpm, carbon_pct, silicon_pct, cooling_time, metal_flow_rate, mold_enc
])
y_score = quality
y_reject = reject

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

# ── 3. DEFINE PYTORCH MULTI-TASK DEEP NEURAL NETWORK (DNN) ──
class CastingMultitaskDNN(nn.Module):
    def __init__(self, input_dim):
        super(CastingMultitaskDNN, self).__init__()
        # Shared representations
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Task 1: Quality Score Regressor
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Task 2: Reject Classification
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_out = self.shared(x)
        score = self.regressor(shared_out)
        prob = self.classifier(shared_out)
        return score, prob

# Instantiate model
model = CastingMultitaskDNN(input_dim=len(feats))

# Loss Functions & Optimizer
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ── 4. TRAINING DEEP LEARNING MODEL ──
print("\n[3/4] Training PyTorch Multi-Task DNN model...")

epochs = 15
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
        
        # Total combined loss (scale MSE down so they contribute evenly)
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
