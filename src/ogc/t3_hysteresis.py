# ogc/tests/t3_hysteresis.py

import numpy as np
from typing import Dict, Any, Tuple
from scipy.signal import welch, csd

def _mscoh(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnitude-squared coherence via Welch estimates:
      Cxy(f) = |Pxy|^2 / (Pxx * Pyy)
    Rückgabe: (f, Cxy)
    """
    noverlap = max(0, nperseg // 2)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    C = np.clip(C.real, 0.0, 1.0)
    return f, C

def _band_stat(x: np.ndarray, y: np.ndarray, fs: float, band: Tuple[float, float], nperseg: int, mode: str = "mean") -> float:
    f, C = _mscoh(x, y, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return 0.0
    Cb = C[mask]
    return float(Cb.max() if mode == "peak" else Cb.mean())

def hysteresis_loop(
    n: int = 300,
    u_min: float = 0.5,
    u_max: float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
    fs: float = 20.0,
    nperseg: int = 128,
    base_band: Tuple[float, float] = (0.78, 0.82),
    n_steps: int = 21,
    sweep: str = "low_edge",  # "low_edge" | "high_edge" | "width"
    mode: str = "mean",       # "mean" | "peak"
) -> Dict[str, Any]:
    """
    T3 Hysterese-Test:
      - Wir sweepen einen Band-Parameter (z.B. untere Kante) aufwärts und abwärts.
      - Messen pro Schritt die Band-Kohärenz.
      - Hysterese-Maß: Fläche zwischen Vorwärts- und Rückwärtskurve (A_loop).

    Rückgabe-Format (kompatibel mit deinem aggregate.py):
      result = {
        "forward": [...],
        "backward": [...],
        "u_grid": [...],      # Parameterwerte
        "A_loop": float,      # Fläche zwischen den Kurven
        "band_base": [f1,f2], # Basisband
        "sweep": "low_edge" | "high_edge" | "width",
        "mode": "mean" | "peak"
      }
    """
    rng = np.random.default_rng(seed)

    # --- Demo-Synthese (ersetzbar durch echte Daten, wenn vorhanden) ---
    # eine 0.8 Hz-Komponente + optionale Störung
    t = np.arange(n) / fs
    x = np.sin(2*np.pi*0.8*t) + noise * rng.standard_normal(n)
    y = np.sin(2*np.pi*0.8*t + 0.25) + noise * rng.standard_normal(n)

    # Parameter-Gitter
    u_grid = np.linspace(u_min, u_max, n_steps)

    f1, f2 = base_band
    forward, backward = [], []

    # Vorwärts-Sweep
    for u in u_grid:
        if sweep == "low_edge":
            band = (u, f2)
        elif sweep == "high_edge":
            band = (f1, u)
        else:  # "width"
            width = (f2 - f1) * u
            mid = 0.5 * (f1 + f2)
            band = (mid - 0.5 * width, mid + 0.5 * width)
        val = _band_stat(x, y, fs=fs, band=band, nperseg=nperseg, mode=mode)
        forward.append(val)

    # Rückwärts-Sweep
    for u in u_grid[::-1]:
        if sweep == "low_edge":
            band = (u, f2)
        elif sweep == "high_edge":
            band = (f1, u)
        else:
            width = (f2 - f1) * u
            mid = 0.5 * (f1 + f2)
            band = (mid - 0.5 * width, mid + 0.5 * width)
        val = _band_stat(x, y, fs=fs, band=band, nperseg=nperseg, mode=mode)
        backward.append(val)

    forward = np.asarray(forward, dtype=float)
    backward = np.asarray(backward, dtype=float)
    # Fläche zwischen den Kurven (numerische Integration)
    A_loop = float(np.trapz(np.abs(forward - backward), u_grid))

    return {
        "forward": forward.tolist(),
        "backward": backward.tolist(),
        "u_grid": u_grid.tolist(),
        "A_loop": A_loop,
        "band_base": [float(f1), float(f2)],
        "sweep": sweep,
        "mode": mode,
        # Meta hilfreich für spätere Plots:
        "fs": float(fs),
        "nperseg": int(nperseg),
        "n": int(n),
        "seed": int(seed),
        "noise": float(noise),
    }


