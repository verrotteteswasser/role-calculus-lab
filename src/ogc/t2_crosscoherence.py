import numpy as np
from scipy.signal import welch, csd

def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None):
    """
    Magnitude-squared coherence via Welch + CSD:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    # SciPy verlangt: noverlap < nperseg
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)

    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

def _phase_surrogate(sig, rng):
    """Phase-only surrogate: zufällige Phasen, Amplitudenspektrum bleibt."""
    X = np.fft.rfft(sig)
    amp = np.abs(X)
    ph = rng.uniform(0, 2*np.pi, size=amp.shape)
    # DC/Nyquist real lassen
    ph[0] = 0.0
    if (len(sig) % 2) == 0:
        ph[-1] = 0.0
    Xs = amp * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=len(sig))

def _stat_from_band(x, y, fs, nperseg, band, mode="mean"):
    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    Cb = C[mask]
    if Cb.size == 0:
        return 0.0, float(mask.mean())
    if mode == "peak":
        return float(np.max(Cb)), float(mask.mean())
    else:
        return float(np.mean(Cb)), float(mask.mean())

def coherence_band(
    x, y,
    fs=1.0,
    band=(0.7, 0.9),
    nperseg=512,
    n_null=500,
    rng=None,
    mode="mean",
    null_mode="flip",
):
    """
    Liefert Band-Statistik (mean/peak) + p-Wert ggü. Null:
      - null_mode="flip": Zeitumkehr eines Kanals (phasenkohärente Null)
      - null_mode="phase": Phase-only Surrogates (Spektrum erhalten)
      - null_mode="both": max(p_flip, p_phase) (konservativ)
    """
    rng = np.random.default_rng(rng)

    # Observed
    stat_obs, band_fraction = _stat_from_band(x, y, fs, nperseg, band, mode)

    def _null_stat_flip():
        xs = x[::-1]
        st, _ = _stat_from_band(xs, y, fs, nperseg, band, mode)
        return st

    def _null_stat_phase():
        xs = _phase_surrogate(x, rng)
        ys = _phase_surrogate(y, rng)
        st, _ = _stat_from_band(xs, ys, fs, nperseg, band, mode)
        return st

    p_flip = p_phase = None
    if null_mode in ("flip", "both"):
        null_vals = np.array([_null_stat_flip() for _ in range(n_null)])
        p_flip = float((null_vals >= stat_obs).mean())
    if null_mode in ("phase", "both"):
        null_vals = np.array([_null_stat_phase() for _ in range(n_null)])
        p_phase = float((null_vals >= stat_obs).mean())

    if null_mode == "flip":
        p_final = p_flip
    elif null_mode == "phase":
        p_final = p_phase
    else:  # both
        # konservativ: die größere (schwächere) Evidenz nehmen
        candidates = [p for p in (p_flip, p_phase) if p is not None]
        p_final = max(candidates) if candidates else None

    return {
        "stat": float(stat_obs),
        "band_fraction": float(band_fraction),
        "mode": mode,
        "null_mode": null_mode,
        "p_value_flip": p_flip,
        "p_value_phase": p_phase,
        "p_value_final": p_final,
    }
