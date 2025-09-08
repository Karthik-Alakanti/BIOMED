import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy
import glob

# ---------- Frequency Feature Extraction ----------
def frequency_features(signal, fs=1000):
    f, Pxx = welch(signal, fs=fs, nperseg=fs*2)

    # Normalize for entropy
    Pxx_norm = Pxx / np.sum(Pxx)

    # Basic features
    dom_freq = f[np.argmax(Pxx)]
    spec_centroid = np.sum(f * Pxx_norm)
    spec_entropy = entropy(Pxx_norm)
    total_power = np.sum(Pxx)

    # Band power (example HRV bands)
    def bandpower(f, Pxx, fmin, fmax):
        idx = np.logical_and(f >= fmin, f <= fmax)
        return np.trapz(Pxx[idx], f[idx])

    vlf_power = bandpower(f, Pxx, 0.003, 0.04)
    lf_power  = bandpower(f, Pxx, 0.04, 0.15)
    hf_power  = bandpower(f, Pxx, 0.15, 0.40)
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    return {
        "dom_freq": dom_freq,
        "spec_centroid": spec_centroid,
        "spec_entropy": spec_entropy,
        "total_power": total_power,
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio
    }

# ---------- Process One .mat File ----------
def process_mat(file_path, condition, fs=1000):
    mat = sio.loadmat(file_path)
    results = []

    # Expected signals (change names according to your mat structure!)
    signals = ["ECG_Lead1", "ECG_Lead2", "ICG", "Radar_I", "Radar_Q"]
    
    for sig in signals:
        if sig in mat:
            data = mat[sig].flatten()
            features = frequency_features(data, fs)
            results.append({
                "Condition": condition,
                "Signal": sig,
                "IMF": "Original",
                **features
            })

    # Now for IMFs (assuming stored as IMF1, IMF2, etc.)
    for imf_num in range(1, 5):
        for sig in signals:
            imf_key = f"{sig}_IMF{imf_num}"
            if imf_key in mat:
                data = mat[imf_key].flatten()
                features = frequency_features(data, fs)
                results.append({
                    "Condition": condition,
                    "Signal": sig,
                    "IMF": f"IMF{imf_num}",
                    **features
                })

    return results

# ---------- Main Runner ----------
def main():
    fs = 1000  # change if needed
    all_results = []

    # Example: mat files named "resting.mat", "valsalva.mat", etc.
    files_conditions = {
        "Resting": "resting.mat",
        "Valsalva": "valsalva.mat",
        "TiltUp": "tiltup.mat",
        "TiltDown": "tiltdown.mat"
    }

    for condition, file_path in files_conditions.items():
        try:
            results = process_mat(file_path, condition, fs)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("frequency_features.csv", index=False)
    print("Features saved to frequency_features.csv")

if __name__ == "__main__":
    main()
