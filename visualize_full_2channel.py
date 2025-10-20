import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import decimate
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
# --- Metrics Calculation Function ---
# =========================================================================
def calculate_and_print_metrics(ground_truth, prediction):
    """
    Calculates and prints Pearson Correlation (PCC) and RMSE.
    """
    gt = ground_truth.flatten()
    pred = prediction.flatten()
    
    correlation, _ = pearsonr(gt, pred)
    rmse = np.sqrt(np.mean((gt - pred)**2))
    
    print("\n--- Final Reconstruction Accuracy Metrics ---")
    print(f"  - Pearson Correlation (PCC): {correlation:.4f}  (Closer to 1.0 is better)")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}  (Closer to 0.0 is better)")
    
    return f"PCC: {correlation:.4f} | RMSE: {rmse:.4f}"


# =========================================================================
# --- Main Visualization Pipeline ---
# =========================================================================
def main():
    # --- Configuration ---
    SUBJECT_ID = 'GDN0030'
    MODEL_PATH = Path('./best_unet_model_2channel.pth')
    DATA_ROOT = Path('./data')
    VMD_OUTPUT_ROOT = Path('./vmd_output_2channel')
    SEGMENT_LENGTH = 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load Raw and Processed Data ---
    raw_file_path = DATA_ROOT / SUBJECT_ID / f"{SUBJECT_ID}_1_Resting.mat"
    vmd_file_path = VMD_OUTPUT_ROOT / f"{SUBJECT_ID}_1_Resting.vmd.npz"

    if not raw_file_path.exists() or not vmd_file_path.exists():
        print(f"❌ Error: Make sure both raw data and processed 2-channel VMD files exist.")
        sys.exit(1)
    
    raw_data = sio.loadmat(raw_file_path)
    raw_radar_i = raw_data['radar_i'].flatten()
    raw_radar_q = raw_data['radar_q'].flatten()
    original_fs = raw_data['fs_radar'][0, 0]

    vmd_data = np.load(vmd_file_path)
    heartbeat_mode = vmd_data['heartbeat_mode']
    breathing_mode = vmd_data['breathing_mode']
    ground_truth_ecg = vmd_data['ground_truth_ecg']
    fs = vmd_data['fs'].item()
    print(f"✅ Loaded raw and processed 2-channel data for {SUBJECT_ID}")

    # --- Step 2: Load Model and Run Inference ---
    print("✅ Loading trained U-Net and reconstructing ECG...")
    model = UNet1D(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Stack the modes to create the 2-channel input
    vmd_input_signal = np.stack([heartbeat_mode, breathing_mode], axis=0)
    vmd_input_signal = np.nan_to_num(vmd_input_signal)
    
    num_segments = vmd_input_signal.shape[1] // SEGMENT_LENGTH
    
    # Reshape for batch processing
    input_tensor = torch.FloatTensor(vmd_input_signal[:, :num_segments * SEGMENT_LENGTH]).reshape(2, num_segments, SEGMENT_LENGTH)
    input_tensor = input_tensor.permute(1, 0, 2) # (num_segments, 2, segment_length)
    
    predictions = []
    with torch.no_grad():
        for i in range(num_segments):
            segment = input_tensor[i:i+1].to(device)
            prediction = model(segment).cpu().numpy()
            predictions.append(prediction.flatten())
    final_reconstruction = np.concatenate(predictions)
    
    # --- Step 3: Calculate Metrics ---
    gt_trimmed = ground_truth_ecg[:len(final_reconstruction)]
    
    gt_norm = (gt_trimmed - np.mean(gt_trimmed)) / np.std(gt_trimmed)
    recon_norm = (final_reconstruction - np.mean(final_reconstruction)) / np.std(final_reconstruction)
    
    metrics_string = calculate_and_print_metrics(gt_norm, recon_norm)

    # --- Step 4: Visualize Everything ---
    print("✅ Generating final comparison plot...")

    fig, axes = plt.subplots(3, 1, figsize=(18, 12)) 
    fig.suptitle(f"Full 2-Channel Pipeline Visualization for {SUBJECT_ID}\n({metrics_string})", fontsize=16, fontweight='bold')

    # Plot 1: Raw Radar Input
    raw_time_axis = np.arange(len(raw_radar_i)) / original_fs
    axes[0].plot(raw_time_axis, raw_radar_i, label='Raw Radar I-Channel', color='blue', alpha=0.7)
    axes[0].plot(raw_time_axis, raw_radar_q, label='Raw Radar Q-Channel', color='red', alpha=0.7)
    axes[0].set_title("1. Raw Radar Signal Input")
    axes[0].set_ylabel("Raw Amplitude")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].legend()
    axes[0].set_xlim(0, 30)

    # Plot 2: Final Reconstruction vs. Ground Truth
    processed_time_axis = np.arange(len(recon_norm)) / fs
    axes[1].plot(processed_time_axis, gt_norm, label='Ground Truth ECG', color='green', linewidth=2)
    axes[1].plot(processed_time_axis, recon_norm, label='Final Reconstructed ECG (U-Net Output)', color='red', linestyle='--')
    axes[1].set_title(f"2. Final Reconstructed ECG vs. Ground Truth ({metrics_string})")
    axes[1].set_ylabel("Normalized Amp.")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend()
    axes[1].set_xlim(0, processed_time_axis[-1])

    # Plot 3: Zoomed-In View
    zoom_start_sec = 100.0
    zoom_duration_sec = 8.0
    axes[2].plot(processed_time_axis, gt_norm, label='Ground Truth ECG', color='green', linewidth=2)
    axes[2].plot(processed_time_axis, recon_norm, label='Final Reconstructed ECG', color='red', linestyle='--')
    axes[2].set_title("3. Zoomed-In View for Morphology Comparison")
    axes[2].set_ylabel("Normalized Amp.")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_xlim(zoom_start_sec, zoom_start_sec + zoom_duration_sec)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"full_pipeline_visualization_2channel_{SUBJECT_ID}.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"✅ Final plot saved to: {plot_filename}")

if __name__ == '__main__':
    main()
