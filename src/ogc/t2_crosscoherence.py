\
import numpy as np
from .nulls import phase_only_surrogate

def cross_spectrum(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = min(len(x), len(y))
    x = x[:n]; y = y[:n]
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    Pxx = (X * np.conj(X)).real
    Pyy = (Y * np.conj(Y)).real
    Pxy = X * np.conj(Y)
    return Pxx, Pyy, Pxy

def coherence_band(x, y, rng=None, n_null=200):
    Pxx, Pyy, Pxy = cross_spectrum(x, y)
    denom = np.sqrt(Pxx * Pyy) + 1e-12
    C1 = np.abs(Pxy) / denom
    rng = np.random.default_rng(rng)
    peaks_null = np.empty(n_null)
    for i in range(n_null):
        y_null = phase_only_surrogate(y, rng=rng)
        _, _, Pxy_null = cross_spectrum(x, y_null)
        C1_null = np.abs(Pxy_null) / denom
        peaks_null[i] = C1_null.max()
    peak = float(C1.max())
    p_value = float((np.sum(peaks_null >= peak) + 1) / (n_null + 1))
    band_fraction = float(np.mean(C1 > 0.1))
    return {"peak": peak, "band_fraction": band_fraction, "p_value": p_value, "C1": C1.tolist()}
