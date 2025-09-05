\
import numpy as np
from .utils import radial_profile, identity_residual

def synthetic_stack(a=1.0, n_points=1000, D=3, noise=0.0, rng=None):
    """
    Generate synthetic radial field with interior slope +1 and exterior -(D-1) decay.
    Returns r, v_r, and C_out (const for exterior r^(D-1)*v_r).
    """
    rng = np.random.default_rng(rng)
    r = np.linspace(0.01, 4*a, n_points)
    v_r = np.empty_like(r)
    s_in_true = 1.0
    interior = r <= a
    v_r[interior] = s_in_true * r[interior]
    C_out_true = a**D * s_in_true
    v_r[~interior] = C_out_true / (r[~interior]**(D-1))
    if noise > 0:
        v_r += rng.normal(0, noise*np.std(v_r), size=v_r.shape)
    return r, v_r, C_out_true

def fit_t1(r, v_r, a, D=3):
    interior = r <= a
    exterior = r > a
    s_in = radial_profile(r[interior], v_r[interior])
    C_out = np.median((r[exterior]**(D-1)) * v_r[exterior])
    delta_I = identity_residual(a, s_in, C_out, D=D)
    return {"s_in": float(s_in), "C_out": float(C_out), "delta_I": float(delta_I)}
