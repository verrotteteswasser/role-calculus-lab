import numpy as np
from scipy.signal import welch, csd

def _mscoh(x, y, fs=1.0, nperseg=512, noverlap=None):
    """
    Magnitude-squared coherence via Welch + CSD:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
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

def coherence_band(x, y, fs=1.0, band=(0.6, 1.0), nperseg=512, n_null=200, rng=None):
    """
    Liefert Peak-Kohärenz im Band + p-Wert ggü. Phase-only Surrogates.
    p = Anteil der Null-Peaks >= beobachtetem Peak (right-tailed).
    """
    rng = np.random.default_rng(rng)

    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    band_mask = (f >= band[0]) & (f <= band[1])
    C_band = C[band_mask]
    peak = float(C_band.max()) if C_band.size else 0.0

    # Null: Phasen getrennt randomisieren
    null_peaks = []
    for _ in range(n_null):
        xs = _phase_surrogate(x, rng)
        ys = _phase_surrogate(y, rng)
        _, Cn = _mscoh(xs, ys, fs=fs, nperseg=nperseg)
        Cn_band = Cn[band_mask]
        null_peaks.append(float(Cn_band.max()) if Cn_band.size else 0.0)

    null_peaks = np.array(null_peaks)
    # right-tailed p (wie "ist Null >= beobachtet?")
    p_value = float((null_peaks >= peak).mean())
    band_fraction = float(band_mask.mean())

    return {
        "peak": peak,
        "band_fraction": band_fraction,
        "p_value": p_value,
        # Optional: kurze Zusammenfassung anstatt die ganze Kurve zu dumpen
        # "summary": {"f_min": float(f[band_mask][0]), "f_max": float(f[band_mask][-1])}
    }
