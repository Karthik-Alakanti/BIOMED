import os
import argparse
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

# Match your subject layout; you'll change SUBJECT_DIR when switching patients
SUBJECT_DIR = os.path.join("E:/BIOMED RS/BIOMED DATASET/datasets_subject_01_to_10_scidata", "GDN0001")

# ---------- helpers (same style as your utilities) ----------
def as1d(x):
    x = np.asarray(x)
    if x is None:
        return None
    if x.ndim == 2:
        if x.shape[0] == 1: return x.ravel()
        if x.shape[1] == 1: return x[:, 0]
        return x[:, 0]
    return x

def first(dct, keys):
    for k in keys:
        if k in dct:
            return dct[k]
    return None

def load_file(path):
    """
    Load all available physiological signals: ECG1, ECG2, ICG, Radar I, Radar Q
    """
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    fs = {
        "ecg": int(d.get("fs_ecg", 2000)),
        "icg": int(d.get("fs_icg", 1000)),
        "radar": int(d.get("fs_radar", 2000)),
    }

    signals = {}
    
    # ECG Lead 1
    ecg1 = first(d, ["tfm_ecg1", "ecg1", "tfm_ecg_1"])
    if ecg1 is not None:
        signals["ecg1"] = as1d(ecg1)

    # ECG Lead 2  
    ecg2 = first(d, ["tfm_ecg2", "ecg2", "tfm_ecg_2"])
    if ecg2 is not None:
        signals["ecg2"] = as1d(ecg2)

    # ICG Signal
    icg = first(d, ["tfm_icg", "icg"])
    if icg is not None:
        signals["icg"] = as1d(icg)

    # Radar I and Q components
    ri = first(d, ["radar_i", "ri", "radarI"])
    rq = first(d, ["radar_q", "rq", "radarQ"])
    
    if ri is not None:
        signals["radar_i"] = as1d(ri)
    if rq is not None:
        signals["radar_q"] = as1d(rq)

    # Check if we have at least ECG1 (minimum requirement)
    if "ecg1" not in signals:
        raise ValueError("Missing ECG1 signal in: " + os.path.basename(path))

    print(f"  Available signals: {list(signals.keys())}")
    return signals, fs

def resample_signals(signals, fs_dict, target_fs):
    """
    Resample all signals to target frequency and align lengths
    """
    resampled = {}
    
    # Map signals to their sampling rates
    fs_map = {
        "ecg1": fs_dict["ecg"],
        "ecg2": fs_dict["ecg"], 
        "icg": fs_dict["icg"],
        "radar_i": fs_dict["radar"],
        "radar_q": fs_dict["radar"]
    }
    
    for name, data in signals.items():
        original_fs = fs_map[name]
        if original_fs != target_fs:
            p, q = _rat(target_fs / original_fs)
            resampled[name] = sig.resample_poly(data, p, q)
        else:
            resampled[name] = data.copy()
    
    # Find minimum length and align all signals
    min_len = min(len(data) for data in resampled.values())
    for key in resampled.keys():
        resampled[key] = resampled[key][:min_len]
    
    return resampled, min_len

def _rat(r, tol=1e-6):
    # robust rational approximation for resample_poly
    from fractions import Fraction
    f = Fraction(r).limit_denominator(10000)
    p, q = f.numerator, f.denominator
    if abs(p/q - r) > tol:
        # fallback using numpy (rare)
        import math
        q = int(round(1/tol))
        p = int(round(r*q))
    return p, q

# ---------- MEMD wrapper ----------
def run_memd(signals_dict, dirvec=64):
    """
    Run MEMD on all available signals
    Returns IMFs with shape (n_channels, N, n_imf) and channel names
    """
    # Create ordered list of available signals
    signal_order = ["ecg1", "ecg2", "icg", "radar_i", "radar_q"]
    available_signals = []
    channel_names = []
    
    for sig_name in signal_order:
        if sig_name in signals_dict:
            available_signals.append(signals_dict[sig_name])
            channel_names.append(sig_name)
    
    if len(available_signals) < 2:
        raise ValueError("Need at least 2 signals for MEMD analysis")
    
    # Stack as channels x samples
    X = np.vstack(available_signals)
    print(f"  Input shape for MEMD: {X.shape} ({len(channel_names)} channels)")
    print(f"  Channel order: {channel_names}")
    
    try:
        # Import the standalone MEMD implementation
        from MEMD_all import memd
        print("  Using standalone MEMD_all.memd implementation")
        
        # Try with (samples, channels) - transpose our input
        try:
            print(f"  Trying input shape (samples, channels): {X.T.shape}")
            imfs = memd(X.T, dirvec)
        except Exception as e1:
            try:
                # Try with (channels, samples) - our current format
                print(f"  Trying input shape (channels, samples): {X.shape}")
                imfs = memd(X, dirvec)
            except Exception as e2:
                raise RuntimeError(f"MEMD failed with both input orientations:\n"
                                 f"  Transposed format error: {e1}\n"
                                 f"  Original format error: {e2}")
        
        # Ensure correct output shape: (channels, samples, n_imf)
        if imfs.ndim == 3:
            print(f"  MEMD output shape: {imfs.shape}")
            
            # The MEMD_all.py returns (n_imf, channels, samples)
            # We need (channels, samples, n_imf)
            imfs = np.transpose(imfs, (1, 2, 0))
            print(f"  Converted to (channels, samples, n_imf): {imfs.shape}")
        else:
            raise RuntimeError(f"Unexpected MEMD output dimensionality: {imfs.ndim}D")
        
        n_imf = imfs.shape[2]
        print(f"  MEMD completed: {n_imf} IMFs extracted for {len(channel_names)} channels")
        return imfs, channel_names
        
    except ImportError as e:
        raise RuntimeError(
            f"MEMD_all.py not found. Please download it from:\n"
            f"https://raw.githubusercontent.com/mariogrune/MEMD-Python-/master/MEMD_all.py\n"
            f"and save it in the same directory as this script.\n\n"
            f"Import error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"MEMD execution failed: {e}")

# ---------- plotting ----------
def plot_memd(tag, signals_dict, channel_names, fs_common, IMFs, outdir, show_imfs=4):
    """
    Plot multi-channel MEMD results
    """
    # Get first signal for time vector
    first_signal = next(iter(signals_dict.values()))
    t = np.arange(len(first_signal), dtype=float) / float(fs_common)
    
    n_channels = len(channel_names)
    n_imf = IMFs.shape[2]
    show = min(show_imfs, n_imf)

    nrows = n_channels + show  # original signals + selected IMFs
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 2.5*nrows), sharex=True)
    
    # Handle single subplot case
    if nrows == 1:
        axes = [axes]

    # Channel colors and labels
    colors = ['blue', 'navy', 'green', 'red', 'darkred']
    labels = {
        'ecg1': 'ECG Lead I',
        'ecg2': 'ECG Lead II',
        'icg': 'ICG Signal', 
        'radar_i': 'Radar I-component',
        'radar_q': 'Radar Q-component'
    }

    # Plot original signals
    for i, ch_name in enumerate(channel_names):
        color = colors[i % len(colors)]
        label = labels.get(ch_name, ch_name.upper())
        
        axes[i].plot(t, signals_dict[ch_name], color=color, linewidth=1)
        axes[i].set_title(label, fontsize=12)
        axes[i].grid(True, alpha=0.3)

    # Plot selected IMFs for all channels
    for k in range(show):
        ax = axes[n_channels + k]
        
        for i, ch_name in enumerate(channel_names):
            color = colors[i % len(colors)]
            label = labels.get(ch_name, ch_name.upper())
            ax.plot(t, IMFs[i, :, k], color=color, 
                   label=f"{label} IMF{k+1}", alpha=0.8, linewidth=1)
        
        ax.set_title(f"IMF {k+1} - All Channels", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    
    plt.suptitle(f"Multi-Channel MEMD Analysis: {tag}", fontsize=14)
    
    # Save the plot
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{tag}_memd_multichannel.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return out_png

# ---------- main flow ----------
def main():
    ap = argparse.ArgumentParser(description="Multi-Channel MEMD on All Physiological Signals")
    ap.add_argument("--out", type=str, default="plots_MEMD", help="Output directory root")
    ap.add_argument("--target_fs", type=float, default=200.0, help="Common resample rate (Hz)")
    ap.add_argument("--dirvec", type=int, default=64, help="MEMD direction vectors (K)")
    ap.add_argument("--max_files", type=int, default=0, help="Process only the first N files (0=all)")
    args = ap.parse_args()

    if not os.path.isdir(SUBJECT_DIR):
        raise SystemExit(f"Subject folder not found: {SUBJECT_DIR}")

    # Use plots_MEMD as the main output directory
    out_dir = os.path.join(args.out, os.path.basename(SUBJECT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Output directory: {out_dir}")

    files = [f for f in sorted(os.listdir(SUBJECT_DIR)) if f.lower().endswith(".mat")]
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit("No .mat files in " + SUBJECT_DIR)

    print(f"Found {len(files)} .mat files to process")

    successful_files = 0
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(SUBJECT_DIR, fname)
        tag = os.path.splitext(fname)[0]
        
        print(f"\nProcessing {i}/{len(files)}: {fname}")
        
        try:
            # Load all available signals
            sigs, fs = load_file(fpath)
            
            # Report loaded signals with their sampling rates
            for name in sigs.keys():
                if name in ["ecg1", "ecg2"]:
                    print(f"  Loaded {name}: {len(sigs[name])} samples @ {fs['ecg']} Hz")
                elif name == "icg":
                    print(f"  Loaded {name}: {len(sigs[name])} samples @ {fs['icg']} Hz")
                else:  # radar signals
                    print(f"  Loaded {name}: {len(sigs[name])} samples @ {fs['radar']} Hz")

            # Preprocessing: detrend + demean (keeps MEMD stable)
            for key in sigs.keys():
                sigs[key] = sigs[key] - np.mean(sigs[key])
                sigs[key] = sig.detrend(sigs[key], type="linear")

            # Resample to common sampling rate
            sigs_resampled, L = resample_signals(sigs, fs, args.target_fs)
            print(f"  Resampled to {args.target_fs} Hz: {L} samples")

            # Run multi-channel MEMD decomposition
            print("  Running multi-channel MEMD decomposition...")
            IMFs, channel_names = run_memd(sigs_resampled, dirvec=args.dirvec)
            n_imf = IMFs.shape[2]

            # Generate and save plot
            png_path = plot_memd(tag, sigs_resampled, channel_names, args.target_fs, IMFs, out_dir, show_imfs=4)

            # Save MAT file with IMFs + metadata
            out_mat = os.path.join(out_dir, f"{tag}_memd_multichannel.mat")
            meta = {
                "IMFs": IMFs.astype(np.float64),
                "nIMF": np.int32(n_imf),
                "fs_common": np.float64(args.target_fs),
                "channel_names": np.array(channel_names, dtype=object),
                "n_channels": np.int32(len(channel_names)),
                "dirvec": np.int32(args.dirvec),
                "original_fs_ecg": np.int32(fs["ecg"]),
                "original_fs_icg": np.int32(fs["icg"]),
                "original_fs_radar": np.int32(fs["radar"]),
                "n_samples": np.int32(L)
            }
            sio.savemat(out_mat, meta, do_compression=True)

            print(f"  ✓ Saved plot: {os.path.basename(png_path)}")
            print(f"  ✓ Saved data: {os.path.basename(out_mat)}")
            successful_files += 1
            
        except Exception as e:
            print(f"  ✗ ERROR processing {fname}: {e}")
            import traceback
            print(f"    {traceback.format_exc().splitlines()[-1]}")

    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful_files}/{len(files)} files")
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()