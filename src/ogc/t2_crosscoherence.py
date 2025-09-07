import numpy as np
from scipy.signal import welch, csd

def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None):
    """
    Magnitude-squared coherence via Welch + CSD:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    Robust: passt nperseg/noverlap an die Signal-Länge an.
    """
    n = int(min(len(x), len(y)))
    nseg = int(min(nperseg, n))
    # Mindestgröße, aber nicht größer als n
    if nseg < 64:
        nseg = max(32, min(n, nseg))

    if noverlap is None:
        ov = nseg // 2
    else:
        ov = int(noverlap)
    # Overlap muss strikt kleiner als nperseg sein
    ov = max(0, min(ov, nseg - 1))

    f, Pxx = welch(x, fs=fs, nperseg=nseg, noverlap=ov, detrend="constant")
    _, Pyy = welch(y, fs=fs, nperseg=nseg, noverlap=ov, detrend="constant")
    _, Pxy = csd(x, y, fs=fs, nperseg=nseg, noverlap=ov, detrend="constant")

    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

def _phase_surrogate(sig, rng):
    """
    Phase-only surrogate: behält das Amplitudenspektrum, randomisiert Phasen.
    """
    X = np.fft.rfft(sig)
    amp = np.abs(X)
    ph = rng.uniform(0, 2*np.pi, size=amp.shape)
    # DC/Nyquist real lassen
    ph[0] = 0.0
    if (len(sig) % 2) == 0:
        ph[-1] = 0.0
    Xs = amp * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=len(sig))

# in t2_crosscoherence.py

def coherence_band(x, y, fs=1.0, band=(0.6, 1.0), nperseg=512, n_null=200,
                   rng=None, mode="peak", null_mode="flip", return_debug=False):
    rng = np.random.default_rng(rng)

    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    band_mask = (f >= band[0]) & (f <= band[1])
    C_band = C[band_mask]

    if C_band.size == 0:
        stat = 0.0
    else:
        if mode == "mean":
            stat = float(C_band.mean())
        else:
            stat = float(C_band.max())

    # Null-Verteilung
    null_stats = np.empty(n_null, dtype=float)
    for i in range(n_null):
        if null_mode == "phase":
            xs = _phase_surrogate(x, rng)
            ys = _phase_surrogate(y, rng)
        else:
            # "flip": zyklischer Random-Shift einer Serie -> zerstört Kopplung, erhält Spektren
            shift = rng.integers(0, len(y))
            ys = np.roll(y, shift)
            xs = x
        _, Cn = _mscoh(xs, ys, fs=fs, nperseg=nperseg)
        Cn_band = Cn[band_mask]
        null_stats[i] = (Cn_band.mean() if mode == "mean" else Cn_band.max()) if Cn_band.size else 0.0

    p_value = float((null_stats >= stat).mean())
    band_fraction = float(band_mask.mean())

    out = {
        "stat": stat,
        "band_fraction": band_fraction,
        "p_value": p_value,
        "mode": mode,
        "null_mode": null_mode,
    }
    if return_debug:
        out["debug"] = {
            "f": f.tolist(),
            "C": C.tolist(),
            "band_mask": band_mask.tolist(),
            "null_stats": null_stats.tolist(),
        }
    return out
