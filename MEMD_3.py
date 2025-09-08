import os
import argparse
import time
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Dict, Tuple, List, Optional
from datetime import datetime

# Base dataset directory - CHANGE THIS TO YOUR PATH
BASE_DATASET_DIR = "E:/BIOMED RS/BIOMED DATASET/datasets_subject_01_to_10_scidata"

def get_patient_directory(base_dir: str, patient_number: int) -> str:
    """Get single patient directory"""
    patient_id = f"GDN{patient_number:04d}"
    patient_path = os.path.join(base_dir, patient_id)
    
    if not os.path.isdir(patient_path):
        raise ValueError(f"Patient directory not found: {patient_path}")
    
    print(f"Found patient directory: {patient_id}")
    return patient_path

@lru_cache(maxsize=128)
def _cached_rat(r: float, tol: float = 1e-6) -> Tuple[int, int]:
    from fractions import Fraction
    f = Fraction(r).limit_denominator(10000)
    p, q = f.numerator, f.denominator
    if abs(p/q - r) > tol:
        q = int(round(1/tol))
        p = int(round(r*q))
    return p, q

def as1d(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.ravel() if x.shape[0] == 1 else x[:, 0]
    return x

def first(dct, keys):
    for k in keys:
        if k in dct:
            return dct[k]
    return None

def fix_array_dimensions(arr, target_shape_info=None):
    """
    Fix common array dimension issues
    Convert (N,) to (N,1) or other compatible shapes
    """
    if arr.ndim == 1:
        # Convert (N,) to (N,1) - column vector
        return arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        # Already (N,1) - good
        return arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
        # Convert (1,N) to (N,1)
        return arr.T
    else:
        return arr

def align_signal_lengths(signals_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure all signals have exactly the same length
    """
    lengths = {name: len(sig) for name, sig in signals_dict.items()}
    min_length = min(lengths.values())
    max_length = max(lengths.values())
    
    print(f"    Signal lengths: min={min_length}, max={max_length}")
    
    if min_length != max_length:
        print(f"    Aligning all signals to length {min_length}")
        aligned_signals = {}
        for name, sig in signals_dict.items():
            aligned_signals[name] = sig[:min_length]
        return aligned_signals
    
    return signals_dict

def load_file_robust(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Load file with robust error handling"""
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {os.path.basename(path)}: {e}")

    fs = {
        "ecg": int(d.get("fs_ecg", 2000)),
        "icg": int(d.get("fs_icg", 1000)), 
        "radar": int(d.get("fs_radar", 2000)),
    }

    signals = {}
    signal_keys = {
        "ecg1": ["tfm_ecg1", "ecg1", "tfm_ecg_1"],
        "ecg2": ["tfm_ecg2", "ecg2", "tfm_ecg_2"], 
        "icg": ["tfm_icg", "icg"],
        "radar_i": ["radar_i", "ri", "radarI"],
        "radar_q": ["radar_q", "rq", "radarQ"]
    }
    
    for sig_name, possible_keys in signal_keys.items():
        data = first(d, possible_keys)
        if data is not None:
            processed_data = as1d(data)
            if processed_data is not None and len(processed_data) > 0:
                signals[sig_name] = processed_data

    if "ecg1" not in signals:
        raise ValueError(f"Missing ECG1 signal in: {os.path.basename(path)}")

    return signals, fs

def resample_signals_robust(signals: Dict[str, np.ndarray], 
                           fs_dict: Dict[str, int], 
                           target_fs: float) -> Tuple[Dict[str, np.ndarray], int]:
    """Robust resampling with dimension fixing"""
    fs_map = {
        "ecg1": fs_dict["ecg"], "ecg2": fs_dict["ecg"], 
        "icg": fs_dict["icg"],
        "radar_i": fs_dict["radar"], "radar_q": fs_dict["radar"]
    }
    
    resampled = {}
    for name, data in signals.items():
        original_fs = fs_map[name]
        if original_fs != target_fs:
            ratio = target_fs / original_fs
            
            if abs(ratio - 1.0) < 0.1:
                # Small change - use interpolation
                new_len = int(len(data) * ratio)
                resampled_data = np.interp(
                    np.linspace(0, len(data)-1, new_len),
                    np.arange(len(data)), data
                ).astype(np.float32)
            else:
                # Larger change - use scipy
                p, q = _cached_rat(ratio)
                resampled_data = sig.resample_poly(data, p, q).astype(np.float32)
        else:
            resampled_data = data.astype(np.float32)
        
        resampled[name] = resampled_data
    
    # Align lengths
    resampled = align_signal_lengths(resampled)
    min_len = min(len(data) for data in resampled.values())
    
    return resampled, min_len

def preprocess_signals_robust(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Robust preprocessing with dimension checking"""
    processed = {}
    for key, data in signals.items():
        try:
            # Ensure 1D array
            if data.ndim > 1:
                data = data.ravel()
            
            # Detrend and demean
            detrended = sig.detrend(data, type="linear")
            processed[key] = detrended - np.mean(detrended)
            
        except Exception as e:
            print(f"    Warning: Preprocessing failed for {key}: {e}")
            # Fallback: just demean
            processed[key] = data - np.mean(data)
    
    return processed

def run_memd_robust(signals_dict: Dict[str, np.ndarray], 
                   dirvec: int = 64) -> Tuple[np.ndarray, List[str]]:
    """
    Ultra-robust MEMD with dimension fixing and multiple fallback strategies
    """
    signal_order = ["ecg1", "ecg2", "icg", "radar_i", "radar_q"]
    available_signals = []
    channel_names = []
    
    for sig_name in signal_order:
        if sig_name in signals_dict:
            available_signals.append(signals_dict[sig_name])
            channel_names.append(sig_name)
    
    if len(available_signals) < 2:
        raise ValueError("Need at least 2 signals for MEMD analysis")
    
    # Ensure all signals have exactly the same length
    min_length = min(len(sig) for sig in available_signals)
    available_signals = [sig[:min_length] for sig in available_signals]
    
    print(f"    Final signal lengths: {[len(sig) for sig in available_signals]}")
    print(f"    Channel names: {channel_names}")
    
    try:
        from MEMD_all import memd
        print(f"    MEMD_all imported successfully")
        
        # Strategy 1: Standard approach with multiple input formats
        success = False
        imfs = None
        method_used = ""
        
        for strategy_num, (input_format, description) in enumerate([
            ("samples_x_channels", "Input as (samples, channels)"),
            ("channels_x_samples", "Input as (channels, samples)"),
            ("reshaped_2d", "Reshaped with explicit 2D"),
            ("minimal_channels", "Using only first 2 channels")
        ], 1):
            
            if success:
                break
                
            try:
                print(f"    Strategy {strategy_num}: {description}")
                
                if input_format == "samples_x_channels":
                    # Most common: (samples, channels)
                    X = np.column_stack(available_signals).astype(np.float64, order='C')
                    print(f"      Input shape: {X.shape}")
                    imfs = memd(X, dirvec)
                    
                elif input_format == "channels_x_samples":
                    # Alternative: (channels, samples)
                    X = np.vstack(available_signals).astype(np.float64, order='C')
                    print(f"      Input shape: {X.shape}")
                    imfs = memd(X, dirvec)
                    
                elif input_format == "reshaped_2d":
                    # Ensure explicit 2D arrays
                    reshaped_signals = []
                    for sig in available_signals:
                        if sig.ndim == 1:
                            reshaped_signals.append(sig.reshape(-1, 1))
                        else:
                            reshaped_signals.append(sig)
                    X = np.hstack(reshaped_signals).astype(np.float64, order='C')
                    print(f"      Reshaped input shape: {X.shape}")
                    imfs = memd(X, dirvec)
                    
                elif input_format == "minimal_channels":
                    # Use only first 2 channels to reduce complexity
                    minimal_signals = available_signals[:2]
                    minimal_names = channel_names[:2]
                    X = np.column_stack(minimal_signals).astype(np.float64, order='C')
                    print(f"      Minimal input shape: {X.shape} with channels: {minimal_names}")
                    imfs = memd(X, dirvec)
                    # Update for return
                    channel_names = minimal_names
                
                success = True
                method_used = f"Strategy {strategy_num}: {description}"
                print(f"    ✓ MEMD successful with {method_used}")
                
            except Exception as e:
                print(f"      Strategy {strategy_num} failed: {str(e)[:100]}")
                continue
        
        if not success:
            # Final fallback: Try with reduced dirvec
            print(f"    All strategies failed, trying reduced dirvec...")
            for reduced_dirvec in [32, 16, 8]:
                try:
                    X = np.column_stack(available_signals[:2]).astype(np.float64, order='C')
                    imfs = memd(X, reduced_dirvec)
                    success = True
                    method_used = f"Fallback with dirvec={reduced_dirvec}, 2 channels"
                    channel_names = channel_names[:2]
                    print(f"    ✓ Fallback successful: {method_used}")
                    break
                except Exception as e:
                    print(f"      Dirvec {reduced_dirvec} failed: {str(e)[:50]}")
                    continue
        
        if not success:
            raise RuntimeError("All MEMD strategies failed")
        
        # Handle output dimensions
        print(f"    Raw MEMD output shape: {imfs.shape}")
        
        if imfs.ndim == 2:
            # Single IMF case: (samples, channels) -> (channels, samples, 1)
            if imfs.shape[1] == len(channel_names):
                imfs = imfs.T[:, :, np.newaxis]
            else:
                imfs = imfs[:, :, np.newaxis].transpose(1, 0, 2)
            print(f"    Converted 2D to 3D: {imfs.shape}")
            
        elif imfs.ndim == 3:
            # 3D case: determine correct arrangement
            dim0, dim1, dim2 = imfs.shape
            n_channels = len(channel_names)
            
            # Try to identify: (n_imf, samples, channels) or (samples, channels, n_imf) etc.
            if dim2 == n_channels:
                # Likely (samples, n_imf, channels) -> (channels, samples, n_imf)
                imfs = imfs.transpose(2, 0, 1)
                print(f"    Converted to (channels, samples, n_imf): {imfs.shape}")
            elif dim1 == n_channels:
                # Likely (n_imf, channels, samples) -> (channels, samples, n_imf)  
                imfs = imfs.transpose(1, 2, 0)
                print(f"    Converted to (channels, samples, n_imf): {imfs.shape}")
            elif dim0 == n_channels:
                # Already (channels, samples, n_imf)
                print(f"    Already in correct format: {imfs.shape}")
            else:
                print(f"    Warning: Uncertain dimension arrangement, keeping as-is")
        
        # Final validation and cleanup
        expected_channels = len(channel_names)
        if imfs.shape[0] != expected_channels:
            print(f"    Warning: Channel mismatch. Expected {expected_channels}, got {imfs.shape[0]}")
            # Try to fix
            if imfs.shape[1] == expected_channels:
                imfs = imfs.transpose(1, 0, 2)
                print(f"    Fixed by transposing first two dims: {imfs.shape}")
        
        n_imf = imfs.shape[2]
        print(f"    ✓ Final MEMD result: {imfs.shape} with {n_imf} IMFs")
        print(f"    Method used: {method_used}")
        
        return imfs.astype(np.float32), channel_names
        
    except ImportError:
        raise RuntimeError("MEMD_all.py not found. Please ensure it's in the same directory.")
    except Exception as e:
        raise RuntimeError(f"MEMD processing failed: {e}")

def plot_memd_robust(tag: str, signals_dict: Dict[str, np.ndarray], 
                    channel_names: List[str], fs_common: float, 
                    IMFs: np.ndarray, outdir: str, show_imfs: int = 4) -> str:
    """Robust plotting with error handling"""
    try:
        first_signal = next(iter(signals_dict.values()))
        t = np.linspace(0, len(first_signal) / fs_common, len(first_signal))
        
        n_channels = len(channel_names)
        n_imf = IMFs.shape[2] if IMFs.ndim == 3 else 1
        show = min(show_imfs, n_imf)

        nrows = n_channels + show
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2*nrows), 
                               sharex=True, constrained_layout=True)
        
        if nrows == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        labels = {
            'ecg1': 'ECG Lead I', 'ecg2': 'ECG Lead II', 'icg': 'ICG',
            'radar_i': 'Radar I', 'radar_q': 'Radar Q'
        }

        # Plot original signals
        for i, ch_name in enumerate(channel_names):
            if ch_name in signals_dict:
                axes[i].plot(t, signals_dict[ch_name], 
                            color=colors[i % len(colors)], linewidth=0.8)
                axes[i].set_title(labels.get(ch_name, ch_name.upper()), fontsize=10)
                axes[i].grid(True, alpha=0.3)

        # Plot IMFs
        for k in range(show):
            ax = axes[n_channels + k]
            for i, ch_name in enumerate(channel_names):
                if i < IMFs.shape[0] and k < IMFs.shape[2]:
                    ax.plot(t, IMFs[i, :, k], color=colors[i % len(colors)], 
                           label=f"{labels.get(ch_name, ch_name)} IMF{k+1}", 
                           alpha=0.8, linewidth=0.8)
            
            ax.set_title(f"IMF {k+1}", fontsize=10)
            ax.legend(fontsize=8, ncol=2, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"MEMD Analysis: {tag}", fontsize=12)
        
        os.makedirs(outdir, exist_ok=True)
        out_png = os.path.join(outdir, f"{tag}_memd_multichannel.png")
        fig.savefig(out_png, dpi=120, bbox_inches='tight', 
                    facecolor='white')
        plt.close(fig)
        
        return out_png
        
    except Exception as e:
        print(f"    Plotting failed: {e}")
        return "plot_failed"

def process_single_patient(patient_dir: str, target_fs: float = 200.0, 
                          dirvec: int = 64, max_files: int = 0, 
                          output_root: str = "plots_MEMD") -> None:
    """
    Process all files for a single patient with robust error handling
    """
    patient_id = os.path.basename(patient_dir)
    out_dir = os.path.join(output_root, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nProcessing Patient: {patient_id}")
    print(f"Input directory: {patient_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Parameters: target_fs={target_fs} Hz, dirvec={dirvec}")
    
    # Get all .mat files
    files = [f for f in sorted(os.listdir(patient_dir)) if f.lower().endswith('.mat')]
    if max_files > 0:
        files = files[:max_files]
    
    if not files:
        print(f"No .mat files found in {patient_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    successful = 0
    total_time = 0
    
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(patient_dir, fname)
        tag = os.path.splitext(fname)[0]
        
        print(f"\n[{i}/{len(files)}] Processing: {fname}")
        start_time = time.time()
        
        try:
            # Load signals
            print(f"  Loading file...")
            sigs, fs = load_file_robust(fpath)
            print(f"  Loaded {len(sigs)} signals: {list(sigs.keys())}")
            
            # Show original signal info
            for name, data in sigs.items():
                fs_type = "ecg" if name.startswith("ecg") else "icg" if name == "icg" else "radar"
                print(f"    {name}: {len(data)} samples @ {fs[fs_type]} Hz")
            
            # Preprocess
            print(f"  Preprocessing...")
            sigs = preprocess_signals_robust(sigs)
            
            # Resample
            print(f"  Resampling to {target_fs} Hz...")
            sigs_resampled, L = resample_signals_robust(sigs, fs, target_fs)
            print(f"  Final length: {L} samples")
            
            # Run MEMD
            print(f"  Running MEMD...")
            IMFs, channel_names = run_memd_robust(sigs_resampled, dirvec=dirvec)
            
            # Plot
            print(f"  Generating plot...")
            png_path = plot_memd_robust(tag, sigs_resampled, channel_names, 
                                       target_fs, IMFs, out_dir, show_imfs=4)
            
            # Save MAT file
            print(f"  Saving results...")
            out_mat = os.path.join(out_dir, f"{tag}_memd_multichannel.mat")
            meta = {
                "IMFs": IMFs.astype(np.float32),
                "nIMF": np.int32(IMFs.shape[2]),
                "fs_common": np.float32(target_fs),
                "channel_names": np.array(channel_names, dtype=object),
                "n_channels": np.int32(len(channel_names)),
                "dirvec": np.int32(dirvec),
                "patient_id": patient_id,
                "n_samples": np.int32(L),
                "processing_timestamp": str(datetime.now())
            }
            sio.savemat(out_mat, meta, do_compression=True)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            successful += 1
            
            print(f"  ✓ Success! ({processing_time:.1f}s, {IMFs.shape[2]} IMFs)")
            print(f"    Saved: {os.path.basename(png_path)}")
            print(f"    Saved: {os.path.basename(out_mat)}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            total_time += processing_time
            print(f"  ✗ Failed after {processing_time:.1f}s: {e}")
            import traceback
            print(f"    Details: {traceback.format_exc().splitlines()[-1]}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Patient {patient_id} Processing Complete")
    print(f"{'='*50}")
    print(f"Successfully processed: {successful}/{len(files)} files")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    if successful > 0:
        print(f"Average time per file: {total_time/successful:.1f} seconds")
    print(f"Output directory: {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Single Patient MEMD Processing (Robust)")
    ap.add_argument("--patient", type=int, required=True,
                   help="Patient number (1-30 for GDN0001-GDN0030)")
    ap.add_argument("--out", type=str, default="plots_MEMD_single", 
                   help="Output directory root")
    ap.add_argument("--target_fs", type=float, default=200.0, 
                   help="Target sampling rate (Hz)")
    ap.add_argument("--dirvec", type=int, default=64, 
                   help="MEMD direction vectors")
    ap.add_argument("--max_files", type=int, default=0, 
                   help="Max files to process (0=all)")
    ap.add_argument("--base_dir", type=str, default=BASE_DATASET_DIR,
                   help="Base dataset directory")
    
    args = ap.parse_args()

    print(f"Single Patient MEMD Processing")
    print(f"Patient: GDN{args.patient:04d}")
    print(f"Base directory: {args.base_dir}")

    try:
        # Get patient directory
        patient_dir = get_patient_directory(args.base_dir, args.patient)
        
        # Process the patient
        process_single_patient(
            patient_dir=patient_dir,
            target_fs=args.target_fs,
            dirvec=args.dirvec,
            max_files=args.max_files,
            output_root=args.out
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())