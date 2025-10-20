import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from scipy.stats import pearsonr

# =========================================================================
# --- U-Net Architecture (Must match the trained 2-channel model) ---
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

# =========================================================================
# --- Metrics Calculation Function (Updated) ---
# =========================================================================
def calculate_metrics(ground_truth, prediction):
    """
    Calculates Pearson Correlation (PCC) and RMSE.
    Handles cases where input signals might be flat (zero variance).
    """
    gt = ground_truth.flatten()
    pred = prediction.flatten()
    
    # --- MODIFICATION: Check for flat signals to prevent NaN results ---
    if np.std(gt) < 1e-9 or np.std(pred) < 1e-9:
        print("  - ⚠️ Warning: One of the signals is flat. Metrics are invalid (nan).")
        return np.nan, np.nan
    
    correlation, _ = pearsonr(gt, pred)
    rmse = np.sqrt(np.mean((gt - pred)**2))
    
    return correlation, rmse

# =========================================================================
# --- Main Analysis Pipeline ---
# =========================================================================
def main():
    # --- Configuration ---
    MODEL_PATH = Path('./best_unet_model_2channel.pth')
    VMD_OUTPUT_ROOT = Path('./vmd_output_2channel')
    SEGMENT_LENGTH = 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load the trained U-Net model ---
    print("\n--- Step 1: Loading trained U-Net model ---")
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found at {MODEL_PATH}.")
        print("   Please run '3_run_unet_trainer.py' first.")
        sys.exit(1)
        
    model = UNet1D(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("✅ Model loaded successfully.")

    # --- Step 2: Find all processed files ---
    all_files = sorted(list(VMD_OUTPUT_ROOT.glob('*.vmd.npz')))
    if not all_files:
        print(f"❌ Error: No processed 2-channel VMD files found in {VMD_OUTPUT_ROOT}.")
        sys.exit(1)
    
    print(f"\n--- Step 2: Found {len(all_files)} subjects to analyze ---")

    # --- Step 3: Loop through each subject, predict, and calculate metrics ---
    all_metrics = []

    for file_path in all_files:
        print(f"\n--- Analyzing: {file_path.name} ---")
        
        # Load the pre-processed 2-channel data
        vmd_data = np.load(file_path)
        heartbeat_mode = vmd_data['heartbeat_mode']
        breathing_mode = vmd_data['breathing_mode']
        ground_truth_ecg = vmd_data['ground_truth_ecg']
        
        # Stack the modes to create the 2-channel input
        vmd_input_signal = np.stack([heartbeat_mode, breathing_mode], axis=0)
        vmd_input_signal = np.nan_to_num(vmd_input_signal)
        
        num_segments = vmd_input_signal.shape[1] // SEGMENT_LENGTH
        
        # Reshape for batch processing
        input_tensor = torch.FloatTensor(vmd_input_signal[:, :num_segments * SEGMENT_LENGTH]).reshape(2, num_segments, SEGMENT_LENGTH)
        input_tensor = input_tensor.permute(1, 0, 2)
        
        # Run inference segment by segment
        predictions = []
        with torch.no_grad():
            for i in range(num_segments):
                segment = input_tensor[i:i+1].to(device)
                prediction = model(segment).cpu().numpy()
                predictions.append(prediction.flatten())
        
        final_reconstruction = np.concatenate(predictions)
        
        # Calculate metrics for this subject
        gt_trimmed = ground_truth_ecg[:len(final_reconstruction)]
        
        # Normalize signals for a fair comparison
        gt_norm = (gt_trimmed - np.mean(gt_trimmed)) / np.std(gt_trimmed)
        recon_norm = (final_reconstruction - np.mean(final_reconstruction)) / np.std(final_reconstruction)
        
        pcc, rmse = calculate_metrics(gt_norm, recon_norm)
        
        all_metrics.append({'subject': file_path.stem.replace('.vmd', ''), 'pcc': pcc, 'rmse': rmse})
        print(f"  - PCC: {pcc:.4f} | RMSE: {rmse:.4f}")

    # --- Step 4: Display Final Summary ---
    print(f"\n{'='*50}")
    print("--- Overall Performance Summary ---")
    print(f"{'='*50}")
    
    # --- MODIFICATION: Use np.nanmean to ignore NaN values when calculating the average ---
    valid_metrics = [m for m in all_metrics if not np.isnan(m['pcc'])]
    avg_pcc = np.nanmean([m['pcc'] for m in all_metrics])
    avg_rmse = np.nanmean([m['rmse'] for m in all_metrics])
    
    print(f"\nAverage across {len(valid_metrics)} successful subjects:")
    print(f"  - Average Pearson Correlation (PCC): {avg_pcc:.4f}")
    print(f"  - Average Root Mean Squared Error (RMSE): {avg_rmse:.4f}")
    
    print("\nIndividual Subject Results:")
    print("-" * 50)
    print(f"{'Subject':<25} | {'PCC':<10} | {'RMSE':<10}")
    print("-" * 50)
    for m in all_metrics:
        # Print 'nan' gracefully for the failed subject
        pcc_str = f"{m['pcc']:.4f}" if not np.isnan(m['pcc']) else "nan"
        rmse_str = f"{m['rmse']:.4f}" if not np.isnan(m['rmse']) else "nan"
        print(f"{m['subject']:<25} | {pcc_str:<10} | {rmse_str:<10}")
    print("-" * 50)

if __name__ == '__main__':
    main()

