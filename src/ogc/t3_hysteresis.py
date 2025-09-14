import numpy as np
import json
from pathlib import Path
from scipy.signal import welch

def coherence_band(x, y, fs, band, nperseg=None):
    """Compute band-mean coherence between x and y."""
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    f, Pyy = welch(y, fs=fs, nperseg=nperseg)
    f, Pxy = welch(x, fs=fs, nperseg=nperseg)  # Placeholder: use real cross-spectral estimator
    coh = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-12)
    mask = (f >= band[0]) & (f <= band[1])
    return coh[mask].mean()

def run_hysteresis(x, y, fs, band_range=(0.5, 1.0), n_steps=20, nperseg=None, out_dir=None, seed=0):
    """
    Run hysteresis test: sweep band edge forward and backward.
    """
    np.random.seed(seed)
    band_min, band_max = band_range
    steps = np.linspace(band_min, band_max, n_steps)

    forward = []
    for b in steps:
        val = coherence_band(x, y, fs, (b, band_max), nperseg=nperseg)
        forward.append(val)

    backward = []
    for b in steps[::-1]:
        val = coherence_band(x, y, fs, (b, band_max), nperseg=nperseg)
        backward.append(val)

    # simple hysteresis metric = area between curves
    hysteresis_area = float(np.trapz(np.abs(np.array(forward) - np.array(backward)), steps))

    result = {
        "params": {
            "fs": fs,
            "band_range": band_range,
            "n_steps": n_steps,
            "nperseg": nperseg,
            "seed": seed
        },
        "result": {
            "forward": forward,
            "backward": backward,
            "hysteresis_area": hysteresis_area,
            "mode": "T3",
            "null_mode": "none"
        }
    }

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = Path(out_dir) / f"T3_seed{seed:03d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result

# CLI-style entrypoint (so it works with your cli.py setup)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run T3 Hysteresis test")
    parser.add_argument("--out", type=str, required=True, help="Output directory for JSONs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=float, default=20.0)
    parser.add_argument("--band_min", type=float, default=0.5)
    parser.add_argument("--band_max", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--nperseg", type=int, default=None)
    args = parser.parse_args()

    # Placeholder: replace with real data loading
    N = 2000
    t = np.arange(N) / args.fs
    x = np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.random.randn(N)
    y = np.sin(2 * np.pi * 0.8 * t + 0.2) + 0.1 * np.random.randn(N)

    run_hysteresis(x, y,
                   fs=args.fs,
                   band_range=(args.band_min, args.band_max),
                   n_steps=args.n_steps,
                   nperseg=args.nperseg,
                   out_dir=args.out,
                   seed=args.seed)
