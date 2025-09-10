import numpy as np
from scipy.signal import welch, csd

# ---------------------------
# Basics: Mag^2 Coherence
# ---------------------------
def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None):
    """
    Magnitude-squared coherence via Welch + CSD:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

# ---------------------------
# Surrogates (Null-Modelle)
# ---------------------------
def _phase_surrogate(sig, rng):
    """Phase-only surrogate: zufällige Phasen, Amplitudenspektrum bleibt."""
    X = np.fft.rfft(sig)
    amp = np.abs(X)
    ph = rng.uniform(0, 2*np.pi, size=amp.shape)
    ph[0] = 0.0
    if (len(sig) % 2) == 0:
        ph[-1] = 0.0
    Xs = amp * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=len(sig))

def _flip_surrogate(x, y, rng):
    """Zufälliges Vorzeichen-Flip von (x oder y oder beiden)."""
    sx = -1.0 if rng.random() < 0.5 else 1.0
    sy = -1.0 if rng.random() < 0.5 else 1.0
    return sx * x, sy * y

def _shift_surrogate(x, y, rng, min_shift_frac=0.1):
    """
    Zirkuläre Verschiebung von y um großen Offset (>= min_shift_frac * N).
    Sehr konservativ, bricht zeitgleiche Strukturen ohne Spektren zu zerstören.
    """
    n = len(y)
    min_shift = max(1, int(np.floor(min_shift_frac * n)))
    if n <= 2*min_shift:
        # Fallback: kleiner Shift
        k = int(rng.integers(1, max(2, n)))
    else:
        k = int(rng.integers(min_shift, n - min_shift))
    ys = np.roll(y, k)
    return x, ys

# ---------------------------
# Statistik im Band
# ---------------------------
def _stat_from_band(x, y, fs, nperseg, band, mode="mean", trim=0.1):
    """
    Liefert Statistik (peak | mean | trimmean) der Kohärenz im Frequenzband.
    Gibt (stat_value, band_fraction) zurück.
    """
    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    Cb = C[mask]
    if Cb.size == 0:
        return 0.0, float(mask.mean())

    if mode == "peak":
        val = float(Cb.max())
    elif mode == "trimmean":
        k = int(np.floor(trim * Cb.size))
        if k > 0 and (Cb.size - 2*k) > 0:
            Cb_sorted = np.sort(Cb)
            val = float(Cb_sorted[k:-k].mean())
        else:
            val = float(Cb.mean())
    else:  # "mean"
        val = float(Cb.mean())

    return val, float(mask.mean())

# ---------------------------
# Single-Null-Modus
# ---------------------------
def coherence_band(
    x, y,
    fs=1.0,
    band=(0.7, 0.9),
    nperseg=512,
    n_null=500,
    rng=None,
    mode="mean",
    trim=0.1,
    null_mode="flip",
    shift_min_frac=0.1,
):
    """
    Teste Band-Kohärenz gegen ein Nullmodell.
    null_mode: "flip" | "phase" | "shift"
    Return: dict mit stat, p_value, null_mean/std, z_score, usw.
    """
    rng = np.random.default_rng(rng)
    stat_obs, band_fraction = _stat_from_band(x, y, fs, nperseg, band, mode, trim)

    null_stats = np.empty(n_null, dtype=float)

    if null_mode == "flip":
        for i in range(n_null):
            xs, ys = _flip_surrogate(x, y, rng)
            null_stats[i], _ = _stat_from_band(xs, ys, fs, nperseg, band, mode, trim)

    elif null_mode == "phase":
        for i in range(n_null):
            xs = _phase_surrogate(x, rng)
            ys = _phase_surrogate(y, rng)
            null_stats[i], _ = _stat_from_band(xs, ys, fs, nperseg, band, mode, trim)

    elif null_mode == "shift":
        for i in range(n_null):
            xs, ys = _shift_surrogate(x, y, rng, min_shift_frac=shift_min_frac)
            null_stats[i], _ = _stat_from_band(xs, ys, fs, nperseg, band, mode, trim)

    else:
        raise ValueError(f"unknown null_mode: {null_mode}")

    # Right-tailed p: wie oft Null >= beobachtetes Stat
    p_value = float((null_stats >= stat_obs).mean())
    null_mu = float(null_stats.mean())
    null_sd = float(np.std(null_stats) + 1e-12)
    z_score = float((stat_obs - null_mu) / null_sd)

    return {
        "stat": float(stat_obs),
        "band_fraction": band_fraction,
        "mode": mode,
        "null_mode": null_mode,
        "p_value": p_value,
        "null_mean": null_mu,
        "null_std": null_sd,
        "z_score": z_score,
    }

# ---------------------------
# Kombi-Variante (flip+phase+shift)
# ---------------------------
def coherence_band_all(
    x, y, fs=1.0, band=(0.7, 0.9), nperseg=512, n_null=500, rng=None, mode="mean", trim=0.1
):
    """
    Führe drei Nullmodelle durch (flip, phase, shift) und liefere die konservative
    Entscheidung: p_final = max(p_flip, p_phase, p_shift).
    """
    kw = dict(fs=fs, band=band, nperseg=nperseg, n_null=n_null, rng=rng, mode=mode, trim=trim)

    r_flip  = coherence_band(x, y, null_mode="flip",  **kw)
    r_phase = coherence_band(x, y, null_mode="phase", **kw)
    r_shift = coherence_band(x, y, null_mode="shift", **kw)

    stat = r_flip["stat"]  # identisch über alle, weil gleiche Beobachtung
    band_fraction = r_flip["band_fraction"]

    p_flip, p_phase, p_shift = r_flip["p_value"], r_phase["p_value"], r_shift["p_value"]
    p_final = float(max(p_flip, p_phase, p_shift))

    # z als Referenz (flip), optional könnte man max-z o.ä. führen
    return {
        "stat": stat,
        "band_fraction": band_fraction,
        "mode": mode,
        "null_mode": "all",
        "p_value_flip": p_flip,
        "p_value_phase": p_phase,
        "p_value_shift": p_shift,
        "p_value_final": p_final,
        "z_score_flip": r_flip["z_score"],
        "z_score_phase": r_phase["z_score"],
        "z_score_shift": r_shift["z_score"],
    }
