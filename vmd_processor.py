import numpy as np
import scipy.io as sio
from pathlib import Path
import sys
from scipy.signal import decimate

# --- Attempt to import CuPy and the new GPU VMD function ---
try:
    import cupy as cp
    from vmd_gpu import VMD_gpu
    GPU_ENABLED = True
    print("✅ CuPy found. VMD will run on the GPU.")
except ImportError:
    print("⚠️ CuPy not found. VMD will run on the CPU.")
    print("   For GPU acceleration, install a compatible CUDA Toolkit and run: pip install cupy-cudaXX")
    from vmdpy import VMD
    GPU_ENABLED = False


def process_subject_with_vmd(mat_file_path, output_dir):
    """
    Loads a raw .mat file, downsamples, calculates the phase signal,
    adaptively applies VMD, robustly selects the heartbeat and breathing
    modes, and saves the result.
    """
    try:
        data = sio.loadmat(mat_file_path)
        radar_i = data['radar_i'].flatten()
        radar_q = data['radar_q'].flatten()
        ecg_gt = data['tfm_ecg1'].flatten()
        original_fs = data['fs_radar'][0, 0]
    except Exception as e:
        print(f"❌ Error loading {mat_file_path.name}: {e}")
        return

    # --- Step 1: Downsample the signals ---
    TARGET_FS = 100
    downsample_factor = int(original_fs / TARGET_FS)
    
    if downsample_factor > 1:
        radar_i = decimate(radar_i, downsample_factor)
        radar_q = decimate(radar_q, downsample_factor)
        ecg_gt = decimate(ecg_gt, downsample_factor)
    fs = TARGET_FS

    # --- Step 2: Reconstruct Phase Signal (Chest Displacement) ---
    complex_signal = radar_i + 1j * radar_q
    instantaneous_phase = np.unwrap(np.angle(complex_signal))
    
    # --- Step 3: Adaptive VMD ---
    alphas_to_try = [1000, 2000, 4000, 500] 
    best_modes = None
    final_alpha = None

    for alpha in alphas_to_try:
        K, tau, DC, init, tol = 5, 0, 0, 1, 1e-7

        if GPU_ENABLED:
            u, _, omega = VMD_gpu(instantaneous_phase, alpha, tau, K, DC, init, tol)
        else:
            u, _, omega_cpu = VMD(instantaneous_phase, alpha, tau, K, DC, init, tol)
            omega = omega_cpu[-1, :]

        center_freqs_hz = np.abs(omega * fs)
        
        # --- MODIFICATION: Widen frequency ranges for more robust selection ---
        heartbeat_range = (0.8, 3.0)
        breathing_range = (0.05, 0.7) # Widen the breathing range
        
        heartbeat_indices = np.where((center_freqs_hz >= heartbeat_range[0]) & (center_freqs_hz <= heartbeat_range[1]))[0]
        breathing_indices = np.where((center_freqs_hz >= breathing_range[0]) & (center_freqs_hz <= breathing_range[1]))[0]

        if len(heartbeat_indices) > 0:
            final_alpha = alpha
            
            # Find the strongest mode within the valid heartbeat band
            heartbeat_mode_idx = heartbeat_indices[np.argmax(np.sum(u[heartbeat_indices, :]**2, axis=1))]
            
            # --- MODIFICATION: Add robust fallback for breathing mode ---
            if len(breathing_indices) > 0:
                # If we find a clear breathing mode, use it
                breathing_mode_idx = breathing_indices[np.argmax(np.sum(u[breathing_indices, :]**2, axis=1))]
                print(f"✅ Found both modes with alpha = {alpha}.")
            else:
                # If not, use the strongest mode with a frequency below the heartbeat as a fallback
                print(f"⚠️ Could not find a distinct breathing mode. Using fallback.")
                fallback_breathing_indices = np.where(center_freqs_hz < heartbeat_range[0])[0]
                if len(fallback_breathing_indices) > 0:
                    breathing_mode_idx = fallback_breathing_indices[np.argmax(np.sum(u[fallback_breathing_indices, :]**2, axis=1))]
                else:
                    # If there's no mode below heartbeat, just reuse the first mode
                    breathing_mode_idx = 0
            
            best_modes = {
                "heartbeat": u[heartbeat_mode_idx, :],
                "breathing": u[breathing_mode_idx, :]
            }
            break 
    
    if best_modes is None:
        print("\n❌ Error: Could not find even a valid heartbeat mode. Skipping subject.")
        return

    # --- Step 4: Save the Results ---
    print(f"✅ Found optimal modes with alpha = {final_alpha}.")

    output_filename = output_dir / f"{mat_file_path.stem}.vmd.npz"
    np.savez_compressed(
        output_filename,
        heartbeat_mode=best_modes["heartbeat"],
        breathing_mode=best_modes["breathing"],
        ground_truth_ecg=ecg_gt,
        fs=fs
    )
    print(f"✅ Saved 2-channel VMD processed signal to: {output_filename.name}")


if __name__ == '__main__':
    DATA_ROOT = Path('./data')
    VMD_OUTPUT_ROOT = Path('./vmd_output_2channel')
    VMD_OUTPUT_ROOT.mkdir(exist_ok=True)

    all_files = sorted(list(DATA_ROOT.glob('GDN*/GDN*_1_Resting.mat')))
    
    if not all_files:
        print(f"❌ Error: No .mat files found in {DATA_ROOT}. Check the path and structure.")
        sys.exit(1)
        
    # Process all available subjects
    files_to_process = all_files
    
    print(f"Found {len(all_files)} subjects. Processing all of them.")

    for mat_file in files_to_process:
        print(f"\n{'='*60}")
        print(f"--- Processing: {mat_file.name} ---")
        process_subject_with_vmd(mat_file, VMD_OUTPUT_ROOT)

