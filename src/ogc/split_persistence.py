import numpy as np

def split_persistence(values_A, values_B, tol=1e-3):
    """
    Vergleicht zwei Splits (z.B. Hardware/Data/Operator).
    Rückgabe: pass=True/False + metrische Abweichung.
    """
    A = np.array(values_A, dtype=float).ravel()
    B = np.array(values_B, dtype=float).ravel()
    if A.size != B.size: 
        return {"pass": False, "reason": "shape_mismatch", "diff": None}
    diff = float(np.mean(np.abs(A - B)))
    return {"pass": bool(diff <= tol), "diff": diff, "tol": float(tol)}
