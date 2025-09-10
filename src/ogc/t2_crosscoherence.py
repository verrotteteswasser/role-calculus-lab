import numpy as np
from scipy.signal import welch, csd

def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None, detrend="constant"):
    """
    Magnitude-squared coherence:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    # SciPy verlangt: noverlap < nperseg
    noverlap = min(noverlap, max(0, nperseg - 1))

    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

def _phase_surrogate(sig, rng):
    """
    Phase-only surrogate: random phases, preserve amplitude spectrum.
    """
    X = np.fft.rfft(sig)
    amp = np.abs(X)
    ph = rng.uniform(0, 2*np.pi, size=amp.shape)
    # DC und Nyquist real lassen
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
        return float(Cb.max()), float(mask.mean())
    else:
        return float(Cb.mean()), float(mask.mean())

def coherence_band(
    x, y,
    fs=1.0,
    band=(0.7, 0.9),
    nperseg=0,           # 0 / None => auto
    n_null=200,
    rng=None,
    mode="mean",         # "mean" oder "peak"
    null_mode="flip"     # "flip", "phase" oder "both"
):
    """
    Testet Band-Kohärenz via Surrogates.
    - null_mode="flip": y -> vorzeichenflip/permute (Phasenbezug zerstören, Spektrum ähnlich)
    - null_mode="phase": Phase-only Surrogates (Amplitude fix)
    - null_mode="both": beides und p_final = max(p_flip, p_phase) (konservativ)

    Rückgabe:
      dict(stat, band_fraction, mode, null_mode, p_value_*, p_value_final, decision_alpha_0.05)
    """
    # RNG
    rng = np.random.default_rng(rng)

    # Auto nperseg (≈6 Segmente), minimal 128 und gerade
    if nperseg in (None, 0):
        L = min(len(x), len(y))
        nperseg = max(128, L // 6)
    if nperseg % 2 == 1:
        nperseg += 1

    # beobachtete Statistik
    stat_obs, band_frac = _stat_from_band(x, y, fs, nperseg, band, mode=mode)

    # ---- Null 1: flip/permutation ----
    p_flip = None
    if null_mode in ("flip", "both"):
        nulls = np.empty(n_null, dtype=float)
        for i in range(n_null):
            # einfache Phasenzerstörung durch zufälliges +/- und Zirkularshift
            sign = -1.0 if rng.random() < 0.5 else 1.0
            shift = rng.integers(0, len(y))
            y_null = np.roll(sign * y, shift)
            nulls[i], _ = _stat_from_band(x, y_null, fs, nperseg, band, mode=mode)
        p_flip = float((nulls >= stat_obs).mean())

    # ---- Null 2: phase-surrogates ----
    p_phase = None
    if null_mode in ("phase", "both"):
        nulls = np.empty(n_null, dtype=float)
        for i in range(n_null):
            xs = _phase_surrogate(x, rng)
            ys = _phase_surrogate(y, rng)
            nulls[i], _ = _stat_from_band(xs, ys, fs, nperseg, band, mode=mode)
        p_phase = float((nulls >= stat_obs).mean())

    # Finales p
    if null_mode == "flip":
        p_final = p_flip
    elif null_mode == "phase":
        p_final = p_phase
    else:
        # konservativ: größeres p
        vals = [v for v in (p_flip, p_phase) if v is not None]
        p_final = float(max(vals)) if vals else None

    return {
        "stat": float(stat_obs),
        "band_fraction": float(band_frac),
        "mode": mode,
        "null_mode": null_mode,
        "p_value_flip": p_flip,
        "p_value_phase": p_phase,
        "p_value_final": p_final,
        "decision_alpha_0.05": (p_final is not None and p_final < 0.05),
    }
