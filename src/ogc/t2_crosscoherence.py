import numpy as np
from scipy.signal import welch, csd

def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

def _phase_surrogate(sig, rng):
    X = np.fft.rfft(sig)
    amp = np.abs(X)
    ph = rng.uniform(0, 2*np.pi, size=amp.shape)
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
        return 0.0, mask.mean()
    if mode == "peak":
        return float(Cb.max()), float(mask.mean())
    else:
        return float(Cb.mean()), float(mask.mean())

def coherence_band(
    x, y, fs=1.0, band=(0.6, 1.0), nperseg=512, n_null=200, rng=None,
    mode="mean", null_mode="phase"  # null_mode: "phase" | "flip" | "both"
):
    """
    Liefert Band-Statistik (mean/peak der Kohärenz) + Nulltests.
    - 'phase': Phase-only Surrogates beider Signale (strenger Test)
    - 'flip':  zirkularer Random-Shift von y ggü. x (spektral treu, Kopplung gebrochen)
               (Bezeichnung 'flip' bleibt zur Rückwärtskompatibilität)
    - 'both':  beides; liefert p_phase und p_flip
    """
    rng = np.random.default_rng(rng)

    stat, band_fraction = _stat_from_band(x, y, fs, nperseg, band, mode=mode)

    p_phase = None
    p_flip  = None

    # --- Phase-only Null ---
    if null_mode in ("phase", "both"):
        null_stats = []
        for _ in range(n_null):
            xs = _phase_surrogate(x, rng)
            ys = _phase_surrogate(y, rng)
            s, _ = _stat_from_band(xs, ys, fs, nperseg, band, mode=mode)
            null_stats.append(s)
        null_stats = np.asarray(null_stats, float)
        p_phase = float((null_stats >= stat).mean())

    # --- "Flip" Null = zirkularer Random-Shift ---
    if null_mode in ("flip", "both"):
        n = len(y)
        null_stats = []
        for _ in range(n_null):
            k = rng.integers(0, n)   # zufällige Zirkularverschiebung
            ys = np.roll(y, int(k))
            s, _ = _stat_from_band(x, ys, fs, nperseg, band, mode=mode)
            null_stats.append(s)
        null_stats = np.asarray(null_stats, float)
        p_flip = float((null_stats >= stat).mean())

    out = {
        "stat": stat,
        "band_fraction": band_fraction,
        "mode": mode,
    }
    if null_mode == "phase":
        out["p_value"] = p_phase
        out["null_mode"] = "phase"
    elif null_mode == "flip":
        out["p_value"] = p_flip
        out["null_mode"] = "flip"
    else:
        out["p_phase"] = p_phase
        out["p_flip"]  = p_flip
        out["null_mode"] = "both"

    return out
