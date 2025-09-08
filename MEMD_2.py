import os
import argparse
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional

# Match your subject layout; you'll change SUBJECT_DIR when switching patients
SUBJECT_DIR = os.path.join("E:/BIOMED RS/BIOMED DATASET/datasets_subject_01_to_10_scidata", "GDN0001")

# ---------- Optimized helpers ----------
@lru_cache(maxsize=128)
def _cached_rat(r: float, tol: float = 1e-6) -> Tuple[int, int]:
    """Cached rational approximation for resample_poly"""
    from fractions import Fraction
    f = Fraction(r).limit_denominator(10000)
    p, q = f.numerator, f.denominator
    if abs(p/q - r) > tol:
        q = int(round(1/tol))
        p = int(round(r*q))
    return p, q

def as1d(x):
    """Optimized array conversion"""
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)  # Use float32 for memory efficiency
    if x.ndim == 2:
        return x.ravel() if x.shape[0] == 1 else x[:, 0]
    return x

def first(dct, keys):
    """Find first available key in dictionary"""
    for k in keys:
        if k in dct:
            return dct[k]
    return None

def load_file_optimized(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Optimized loading with minimal memory allocation
    """
    # Load only required variables to save memory
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False, 
                       variable_names=None)  # Let scipy determine what to load
    except Exception:
        # Fallback to full load if selective loading fails
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    # Extract sampling frequencies once
    fs = {
        "ecg": int(d.get("fs_ecg", 2000)),
        "icg": int(d.get("fs_icg", 1000)), 
        "radar": int(d.get("fs_radar", 2000)),
    }

    # Pre-allocate signals dictionary
    signals = {}
    
    # Signal loading with error handling
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
            signals[sig_name] = as1d(data)

    if "ecg1" not in signals:
        raise ValueError(f"Missing ECG1 signal in: {os.path.basename(path)}")

    return signals, fs

def resample_signals_optimized(signals: Dict[str, np.ndarray], 
                             fs_dict: Dict[str, int], 
                             target_fs: float) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Optimized resampling with parallel processing for large signals
    """
    fs_map = {
        "ecg1": fs_dict["ecg"], "ecg2": fs_dict["ecg"], 
        "icg": fs_dict["icg"],
        "radar_i": fs_dict["radar"], "radar_q": fs_dict["radar"]
    }
    
    resampled = {}
    
    # Process resampling
    for name, data in signals.items():
        original_fs = fs_map[name]
        if original_fs != target_fs:
            ratio = target_fs / original_fs
            p, q = _cached_rat(ratio)
            
            # Use faster resampling for small ratios
            if abs(ratio - 1.0) < 0.1:  # Less than 10% change
                # Simple interpolation for small changes
                new_len = int(len(data) * ratio)
                resampled[name] = np.interp(
                    np.linspace(0, len(data)-1, new_len),
                    np.arange(len(data)), 
                    data
                ).astype(np.float32)
            else:
                # Use scipy for larger changes
                resampled[name] = sig.resample_poly(data, p, q).astype(np.float32)
        else:
            resampled[name] = data.astype(np.float32)
    
    # Align lengths efficiently
    min_len = min(len(data) for data in resampled.values())
    for key in resampled.keys():
        if len(resampled[key]) > min_len:
            resampled[key] = resampled[key][:min_len]
    
    return resampled, min_len

def preprocess_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Optimized preprocessing: detrend + demean
    """
    processed = {}
    for key, data in signals.items():
        # Combined detrend and demean in one pass
        detrended = sig.detrend(data, type="linear")
        processed[key] = detrended - np.mean(detrended)
    return processed

def run_memd_optimized(signals_dict: Dict[str, np.ndarray], 
                      dirvec: int = 64) -> Tuple[np.ndarray, List[str]]:
    """
    Optimized MEMD execution with better memory management
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
    
    # Stack efficiently with proper dtype
    X = np.vstack(available_signals).astype(np.float64)  # MEMD needs float64
    
    try:
        from MEMD_all import memd
        
        # Try optimized MEMD call
        print(f"  Running MEMD on {X.shape} (dirvec={dirvec})")
        
        # Prefer samples x channels format for most MEMD implementations
        try:
            imfs = memd(X.T, dirvec)
            print(f"  MEMD completed with output shape: {imfs.shape}")
        except Exception as e1:
            try:
                imfs = memd(X, dirvec)
                print(f"  MEMD completed with output shape: {imfs.shape}")
            except Exception as e2:
                raise RuntimeError(f"MEMD failed: {e1}, {e2}")
        
        # Ensure output is (channels, samples, n_imf)
        if imfs.ndim == 3:
            # Common MEMD output: (n_imf, channels, samples) -> (channels, samples, n_imf)
            if imfs.shape[0] < imfs.shape[1]:  # n_imf is likely first dimension
                imfs = np.transpose(imfs, (1, 2, 0))
        else:
            raise RuntimeError(f"Unexpected MEMD output shape: {imfs.shape}")
        
        # Convert back to float32 to save memory
        imfs = imfs.astype(np.float32)
        
        return imfs, channel_names
        
    except ImportError as e:
        raise RuntimeError(f"MEMD_all.py not found: {e}")

def plot_memd_fast(tag: str, signals_dict: Dict[str, np.ndarray], 
                   channel_names: List[str], fs_common: float, 
                   IMFs: np.ndarray, outdir: str, show_imfs: int = 4) -> str:
    """
    Optimized plotting with reduced memory usage
    """
    first_signal = next(iter(signals_dict.values()))
    t = np.linspace(0, len(first_signal) / fs_common, len(first_signal))
    
    n_channels = len(channel_names)
    n_imf = IMFs.shape[2]
    show = min(show_imfs, n_imf)

    # Use subplots more efficiently
    nrows = n_channels + show
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2*nrows), 
                           sharex=True, constrained_layout=True)
    
    if nrows == 1:
        axes = [axes]

    # Optimized colors and labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels = {
        'ecg1': 'ECG Lead I', 'ecg2': 'ECG Lead II', 'icg': 'ICG',
        'radar_i': 'Radar I', 'radar_q': 'Radar Q'
    }

    # Plot original signals with reduced line width for speed
    for i, ch_name in enumerate(channel_names):
        axes[i].plot(t, signals_dict[ch_name], 
                    color=colors[i % len(colors)], linewidth=0.8, rasterized=True)
        axes[i].set_title(labels.get(ch_name, ch_name.upper()), fontsize=10)
        axes[i].grid(True, alpha=0.3)

    # Plot IMFs
    for k in range(show):
        ax = axes[n_channels + k]
        for i, ch_name in enumerate(channel_names):
            ax.plot(t, IMFs[i, :, k], color=colors[i % len(colors)], 
                   label=f"{labels.get(ch_name, ch_name)} IMF{k+1}", 
                   alpha=0.8, linewidth=0.8, rasterized=True)
        
        ax.set_title(f"IMF {k+1}", fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"MEMD Analysis: {tag}", fontsize=12)
    
    # Save with optimization
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{tag}_memd_multichannel.png")
    fig.savefig(out_png, dpi=120, bbox_inches='tight', 
                facecolor='white', rasterized=True)
    plt.close(fig)
    
    return out_png

def process_single_file(args_tuple: Tuple) -> Tuple[str, bool, str]:
    """
    Process a single file - designed for parallel execution
    """
    fpath, tag, target_fs, dirvec, out_dir = args_tuple
    
    try:
        # Load and preprocess
        sigs, fs = load_file_optimized(fpath)
        sigs = preprocess_signals(sigs)
        
        # Resample
        sigs_resampled, L = resample_signals_optimized(sigs, fs, target_fs)
        
        # MEMD
        IMFs, channel_names = run_memd_optimized(sigs_resampled, dirvec=dirvec)
        
        # Plot
        png_path = plot_memd_fast(tag, sigs_resampled, channel_names, 
                                 target_fs, IMFs, out_dir, show_imfs=4)
        
        # Save results
        out_mat = os.path.join(out_dir, f"{tag}_memd_multichannel.mat")
        meta = {
            "IMFs": IMFs.astype(np.float32),  # Use float32 for storage
            "nIMF": np.int32(IMFs.shape[2]),
            "fs_common": np.float32(target_fs),
            "channel_names": np.array(channel_names, dtype=object),
            "n_channels": np.int32(len(channel_names)),
            "dirvec": np.int32(dirvec),
            "original_fs": {k: np.int32(v) for k, v in fs.items()},
            "n_samples": np.int32(L)
        }
        sio.savemat(out_mat, meta, do_compression=True)
        
        return tag, True, f"Processed successfully: {L} samples, {IMFs.shape[2]} IMFs"
        
    except Exception as e:
        return tag, False, str(e)

def main():
    ap = argparse.ArgumentParser(description="Optimized Multi-Channel MEMD")
    ap.add_argument("--out", type=str, default="plots_MEMD", help="Output directory")
    ap.add_argument("--target_fs", type=float, default=200.0, help="Target sampling rate")
    ap.add_argument("--dirvec", type=int, default=64, help="MEMD direction vectors")
    ap.add_argument("--max_files", type=int, default=0, help="Max files to process")
    ap.add_argument("--parallel", type=int, default=0, 
                   help="Parallel workers (0=auto, 1=sequential)")
    ap.add_argument("--chunk_size", type=int, default=5, 
                   help="Files per chunk for parallel processing")
    args = ap.parse_args()

    if not os.path.isdir(SUBJECT_DIR):
        raise SystemExit(f"Subject folder not found: {SUBJECT_DIR}")

    out_dir = os.path.join(args.out, os.path.basename(SUBJECT_DIR))
    os.makedirs(out_dir, exist_ok=True)

    # Get file list
    files = [f for f in sorted(os.listdir(SUBJECT_DIR)) if f.lower().endswith(".mat")]
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit("No .mat files found")

    print(f"Processing {len(files)} files with target_fs={args.target_fs} Hz")

    # Determine parallel workers
    if args.parallel == 0:
        n_workers = min(mp.cpu_count() - 1, len(files), 4)  # Conservative default
    elif args.parallel == 1:
        n_workers = 1  # Sequential
    else:
        n_workers = min(args.parallel, len(files))

    # Prepare arguments for processing
    file_args = []
    for fname in files:
        fpath = os.path.join(SUBJECT_DIR, fname)
        tag = os.path.splitext(fname)[0]
        file_args.append((fpath, tag, args.target_fs, args.dirvec, out_dir))

    successful = 0
    
    if n_workers == 1:
        # Sequential processing
        print("Running sequentially...")
        for i, args_tuple in enumerate(file_args, 1):
            tag, success, msg = process_single_file(args_tuple)
            print(f"[{i}/{len(files)}] {tag}: {msg}")
            if success:
                successful += 1
    else:
        # Parallel processing
        print(f"Running with {n_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit jobs in chunks
            futures = []
            for i in range(0, len(file_args), args.chunk_size):
                chunk = file_args[i:i + args.chunk_size]
                for args_tuple in chunk:
                    future = executor.submit(process_single_file, args_tuple)
                    futures.append(future)
            
            # Collect results
            for i, future in enumerate(as_completed(futures), 1):
                tag, success, msg = future.result()
                print(f"[{i}/{len(files)}] {tag}: {msg}")
                if success:
                    successful += 1

    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}/{len(files)} files")
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()