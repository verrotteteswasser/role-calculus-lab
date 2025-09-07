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

def coherence_band(
    x,
    y,
    fs=1.0,
    band=(0.6, 1.0),
    nperseg=512,
    n_null=200,
    rng=None,
    mode="peak",
):
    """
    Kohärenz-Statistik im Frequenzband + p-Wert ggü. Phase-only Surrogates.
    mode: "peak" (max im Band) oder "mean" (Bandmittel, robuster).
    p = Anteil der Null-Statistiken >= beobachteter Statistik (right-tailed).
    """
    rng = np.random.default_rng(rng)

    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    band_mask = (f >= band[0]) & (f <= band[1])
    band_fraction = float(band_mask.mean())

    if not np.any(band_mask):
        return {"stat": 0.0, "band_fraction": 0.0, "p_value": 1.0, "mode": mode}

    C_band = C[band_mask]
    stat_fn = np.max if mode == "peak" else np.mean
    obs = float(stat_fn(C_band))

    null_vals = []
    for _ in range(n_null):
        xs = _phase_surrogate(x, rng)
        ys = _phase_surrogate(y, rng)
        _, Cn = _mscoh(xs, ys, fs=fs, nperseg=nperseg)
        Cn_band = Cn[band_mask]
        null_vals.append(float(stat_fn(Cn_band)))

    null_vals = np.asarray(null_vals, dtype=float)
    p_value = float((null_vals >= obs).mean())

    return {
        "stat": obs,
        "band_fraction": band_fraction,
        "p_value": p_value,
        "mode": mode,
    }

def coherence_band(
    x, y, fs=1.0, band=(0.6, 1.0), nperseg=512, n_null=200, rng=None,
    mode="mean",              # "mean" oder "peak"
    null_mode="flip"          # "flip" (empfohlen) oder "phase"
):
    rng = np.random.default_rng(rng)

    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    band_mask = (f >= band[0]) & (f <= band[1])
    if not np.any(band_mask):
        return {"stat": 0.0, "band_fraction": 0.0, "p_value": 1.0, "mode": mode}

    C_band = C[band_mask]
    stat_fn = np.max if mode == "peak" else np.mean
    obs = float(stat_fn(C_band))
    band_fraction = float(band_mask.mean())

    null_vals = []
    for _ in range(n_null):
        if null_mode == "phase":
            xs = _phase_surrogate(x, rng)
            ys = _phase_surrogate(y, rng)
            _, Cn = _mscoh(xs, ys, fs=fs, nperseg=nperseg)
        else:
            # Segment-Flip: y in Blöcken zufällig mit ±1 multiplizieren
            nseg = int(min(nperseg, len(y)))
            if nseg < 64:
                nseg = max(32, nseg)
            ov = nseg // 2
            ys = y.copy()
            start = 0
            step = max(1, nseg - ov)
            while start < len(ys):
                end = min(len(ys), start + nseg)
                if rng.random() < 0.5:
                    ys[start:end] *= -1.0
                start += step
            _, Cn = _mscoh(x, ys, fs=fs, nperseg=nperseg)

        Cn_band = Cn[band_mask]
        null_vals.append(float(stat_fn(Cn_band)))

    null_vals = np.asarray(null_vals, dtype=float)
    p_value = float((null_vals >= obs).mean())
    return {"stat": obs, "band_fraction": band_fraction, "p_value": p_value, "mode": mode}
