import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

# =========================================================================
# --- U-Net Architecture & Dataset (Updated for 2 Channels) ---
# =========================================================================
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(UNet1D, self).__init__()
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        self.upconv3 = nn.ConvTranspose1d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        self.upconv2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        self.upconv1 = nn.ConvTranspose1d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        self.out = nn.Conv1d(base_filters, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return self.out(dec1)

class VMD_ECG_Dataset(Dataset):
    def __init__(self, heartbeat_modes, breathing_modes, ecg_targets, segment_length):
        self.segment_length = (segment_length // 8) * 8
        self.input_data = np.stack([heartbeat_modes, breathing_modes], axis=0)
        self.target_data = ecg_targets.reshape(1, -1)
        self.n_segments = self.input_data.shape[1] // self.segment_length

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        start = idx * self.segment_length
        end = start + self.segment_length
        input_segment = self.input_data[:, start:end]
        target_segment = self.target_data[:, start:end]
        return torch.FloatTensor(input_segment), torch.FloatTensor(target_segment)

# =========================================================================
# --- Metrics Calculation Function ---
# =========================================================================
def calculate_metrics(ground_truth, prediction):
    gt = ground_truth.flatten()
    pred = prediction.flatten()
    if np.std(gt) < 1e-9 or np.std(pred) < 1e-9:
        return np.nan, np.nan
    correlation, _ = pearsonr(gt, pred)
    rmse = np.sqrt(np.mean((gt - pred)**2))
    return correlation, rmse

# =========================================================================
# --- Main Training Pipeline ---
# =========================================================================
def main():
    # --- Configuration ---
    VMD_PROCESSED_ROOT = Path('./vmd_output_2channel')
    SEGMENT_LENGTH = 2048
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load and Consolidate Data ---
    print("\n--- Step 1: Loading all 2-channel VMD files ---")
    all_files = sorted(list(VMD_PROCESSED_ROOT.glob('*.vmd.npz')))
    
    if not all_files:
        print(f"❌ Error: No processed files found in {VMD_PROCESSED_ROOT}.")
        sys.exit(1)

    all_hb_modes, all_br_modes, all_ecg_targets = [], [], []
    for file_path in all_files:
        try:
            data = np.load(file_path)
            all_hb_modes.append(data['heartbeat_mode'])
            all_br_modes.append(data['breathing_mode'])
            all_ecg_targets.append(data['ground_truth_ecg'])
            print(f"  - Loaded {file_path.name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load {file_path.name}. Skipping. Error: {e}")
    
    combined_hb_mode = np.concatenate(all_hb_modes)
    combined_br_mode = np.concatenate(all_br_modes)
    combined_ecg_target = np.concatenate(all_ecg_targets)
    
    combined_hb_mode = np.nan_to_num(combined_hb_mode)
    combined_br_mode = np.nan_to_num(combined_br_mode)
    combined_ecg_target = np.nan_to_num(combined_ecg_target)
    
    # --- Step 2: Create Train/Validation/Test Splits ---
    print("\n--- Step 2: Creating data splits ---")
    dataset = VMD_ECG_Dataset(combined_hb_mode, combined_br_mode, combined_ecg_target, SEGMENT_LENGTH)
    
    # --- NEW: Create a Train/Test Split (80/20) ---
    test_split = 0.20
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_val_size = dataset_size - test_size

    train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_val_size, test_size])

    # --- NEW: Split the training data further into Train/Validation ---
    val_split = 0.20 # 20% of the 80%
    train_val_size = len(train_val_dataset)
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset created:")
    print(f"  - Training segments:   {len(train_dataset)}")
    print(f"  - Validation segments: {len(val_dataset)}")
    print(f"  - Test segments:       {len(test_dataset)}")

    # --- Step 3: Initialize and Train U-Net Model ---
    print("\n--- Step 3: Training U-Net ---")
    model = UNet1D(in_channels=2, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.6f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model_2channel.pth')
            print("  -> Saved new best model.")
    
    # --- Step 4: Final Evaluation on Test Set ---
    print("\n--- Step 4: Evaluating final model on the hold-out test set ---")
    model.load_state_dict(torch.load('best_unet_model_2channel.pth'))
    model.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            all_predictions.append(outputs)
            all_targets.append(targets.numpy())

    # Concatenate all batches from the test set
    final_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
    final_targets = np.concatenate([t.reshape(-1) for t in all_targets])

    # Normalize for metrics
    gt_norm = (final_targets - np.mean(final_targets)) / np.std(final_targets)
    pred_norm = (final_predictions - np.mean(final_predictions)) / np.std(final_predictions)

    pcc, rmse = calculate_metrics(gt_norm, pred_norm)
    print(f"\n--- Final Test Set Performance ---")
    print(f"  - Pearson Correlation (PCC): {pcc:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")

    # --- Step 5: Visualize Final Reconstruction on Test Samples ---
    print("\n--- Step 5: Visualizing final reconstruction ---")
    
    # Use the test_loader for visualization
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()

    num_to_plot = min(5, BATCH_SIZE)
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(18, 4 * num_to_plot), sharex=True)
    fig.suptitle("Final ECG Reconstruction (on Test Set Samples)", fontsize=16, fontweight='bold')
    
    for i in range(num_to_plot):
        gt = targets[i, 0, :].numpy()
        pred = predictions[i, 0, :]
        gt_norm = (gt - np.mean(gt)) / np.std(gt)
        pred_norm = (pred - np.mean(pred)) / np.std(pred)

        axes[i].plot(gt_norm, label='Ground Truth ECG', color='green', linewidth=2)
        axes[i].plot(pred_norm, label='U-Net Reconstructed ECG', color='red', linestyle='--')
        axes[i].set_title(f"Test Sample {i+1}")
        axes[i].set_ylabel("Normalized Amp.")
        axes[i].legend()
    axes[-1].set_xlabel("Sample Number")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('unet_final_reconstruction_2channel.png', dpi=150)
    print("✅ Final plot saved to unet_final_reconstruction_2channel.png")

if __name__ == '__main__':
    main()

