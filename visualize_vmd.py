import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.signal import find_peaks, butter, filtfilt

def find_ecg_r_peaks(ecg_signal, fs, prominence):
    """
    Finds R-peaks using a robust Pan-Tompkins-like algorithm.
    """
    # Use a bandpass filter to isolate the QRS complex frequencies
    lowcut = 5.0
    highcut = 20.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)

    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg**2
    window_size = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    integrated_ecg_normalized = (integrated_ecg - np.mean(integrated_ecg)) / np.std(integrated_ecg)
    min_distance = int(0.3 * fs)
    r_peaks, _ = find_peaks(integrated_ecg_normalized, prominence=prominence, distance=min_distance)
    return r_peaks

def visualize_vmd_peaks(npz_file_path, zoom_start_sec, zoom_duration_sec):
    """
    Loads a processed .vmd.npz file, finds peaks on both the VMD signal and
    the ground truth, and creates a zoomed-in comparison plot.
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        print(f"✅ Successfully loaded processed file: {npz_file_path.name}")
    except Exception as e:
        print(f"❌ Error loading the file: {e}")
        sys.exit(1)

    # --- Configuration for Peak Detection ---
    VMD_PEAK_PROMINENCE = 0.5
    ECG_PEAK_PROMINENCE = 2.0

    # --- Extract data arrays ---
    reconstructed_signal = data['reconstructed_signal']
    ground_truth_ecg = data.get('ground_truth_ecg')
    fs = data['fs'].item()

    if ground_truth_ecg is None:
        print(f"❌ Error: 'ground_truth_ecg' not found in {npz_file_path.name}.")
        return

    # --- Normalize signals for visualization ---
    vmd_normalized = (reconstructed_signal - np.mean(reconstructed_signal)) / np.std(reconstructed_signal)
    ecg_normalized = (ground_truth_ecg - np.mean(ground_truth_ecg)) / np.std(ground_truth_ecg)
    
    time_axis = np.arange(len(vmd_normalized)) / fs

    # --- Find Peaks in Both Signals ---
    print("Finding peaks in VMD signal and Ground Truth ECG...")
    min_beat_distance = int(0.4 * fs) # Assume minimum 40 BPM
    vmd_peaks, _ = find_peaks(vmd_normalized, prominence=VMD_PEAK_PROMINENCE, distance=min_beat_distance)
    ecg_peaks = find_ecg_r_peaks(ground_truth_ecg, fs, ECG_PEAK_PROMINENCE)
    print(f"Found {len(vmd_peaks)} peaks in VMD signal.")
    print(f"Found {len(ecg_peaks)} R-peaks in Ground Truth ECG.")

    # --- Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    
    zoom_end_sec = zoom_start_sec + zoom_duration_sec
    fig.suptitle(f"VMD Peak Detection vs. Ground Truth for {npz_file_path.stem}", fontsize=16, fontweight='bold')
    ax.set_title(f"Comparison (Zoomed View: {zoom_start_sec}s - {zoom_end_sec}s)")

    # Plot the signals
    ax.plot(time_axis, ecg_normalized, color='green', label='Ground Truth ECG (Normalized)', alpha=0.8, linewidth=2)
    ax.plot(time_axis, vmd_normalized, color='red', label='VMD Heartbeat Mode', alpha=0.8, linestyle='--')
    
    # Plot the detected peaks
    ax.plot(time_axis[ecg_peaks], ecg_normalized[ecg_peaks], 'x', color='black', markersize=10, mew=2, label=f'ECG R-Peaks ({len(ecg_peaks)})')
    ax.plot(time_axis[vmd_peaks], vmd_normalized[vmd_peaks], 'o', color='magenta', markersize=10, fillstyle='none', mew=2, label=f'VMD Peaks ({len(vmd_peaks)})')
    
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("Time (seconds)")
    ax.legend()
    ax.set_xlim(zoom_start_sec, zoom_end_sec)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = npz_file_path.with_suffix(f'.peak_visualization_zoomed_{zoom_start_sec}s.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Peak comparison plot saved to: {plot_filename}")

if __name__ == '__main__':
    # =========================================================================
    # --- ⚙️ CONFIGURATION ⚙️ ---
    # =========================================================================
    SUBJECT_ID = 'GDN0001' 
    ZOOM_START_SECONDS = 100.0
    ZOOM_DURATION_SECONDS = 4.0
    # =========================================================================

    VMD_OUTPUT_ROOT = Path('./vmd_output')
    file_to_visualize = VMD_OUTPUT_ROOT / f"{SUBJECT_ID}_1_Resting.vmd.npz"
    
    if not file_to_visualize.exists():
        print(f"❌ Error: Processed VMD file not found at: {file_to_visualize}")
        sys.exit(1)
        
    visualize_vmd_peaks(file_to_visualize, ZOOM_START_SECONDS, ZOOM_DURATION_SECONDS)
