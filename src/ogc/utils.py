\
import numpy as np

def bootstrap_ci(x, func=np.mean, n_boot=1000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    boots = np.empty(n_boot)
    n = x.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = func(x[idx])
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return lo, hi

def radial_profile(r, v_r, r_split=None):
    """
    Compute interior slope (regression through origin).
    Caller should pass interior-only r, v_r arrays.
    """
    r = np.asarray(r)
    v_r = np.asarray(v_r)
    s_in = (r @ v_r) / (r @ r)
    return s_in

def identity_residual(a, s_in, C_out, D=3):
    target = a**D
    return abs(C_out / s_in - target) / target
