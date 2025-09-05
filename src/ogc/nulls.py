\
import numpy as np

def label_shuffle(labels, rng=None):
    rng = np.random.default_rng(rng)
    labels = np.asarray(labels)
    return rng.permutation(labels)

def shell_randomization(values, rng=None):
    """
    Permute radial shells while preserving counts (1D array of shells).
    """
    rng = np.random.default_rng(rng)
    values = np.asarray(values)
    return rng.permutation(values)

def phase_only_surrogate(x, rng=None):
    """
    Real-valued 1D signal -> phase-randomized surrogate with amplitude preserved.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    phase = np.angle(X)
    rand_phase = rng.uniform(0, 2*np.pi, size=phase.shape)
    # keep DC and (if exists) Nyquist phases
    rand_phase[0] = phase[0]
    if (x.shape[0] % 2) == 0:
        rand_phase[-1] = phase[-1]
    Y = amp * np.exp(1j * rand_phase)
    y = np.fft.irfft(Y, n=x.shape[0])
    return y

def degree_preserving_rewire(adj, n_swap=1000, rng=None):
    """
    Simple Maslov-Sneppen rewiring for undirected binary graph (numpy array).
    """
    rng = np.random.default_rng(rng)
    A = adj.copy()
    n = A.shape[0]
    edges = np.transpose(np.triu(A, 1).nonzero())
    m = edges.shape[0]
    if m < 2:
        return A
    for _ in range(n_swap):
        i1, i2 = rng.integers(0, m, size=2)
        (a, b) = edges[i1]
        (c, d) = edges[i2]
        if len({a,b,c,d}) < 4:
            continue
        # proposed swaps: (a,d) and (c,b)
        if A[a, d] or A[c, b]:
            continue
        if a==d or c==b:
            continue
        A[a, b] = A[b, a] = 0
        A[c, d] = A[d, c] = 0
        A[a, d] = A[d, a] = 1
        A[c, b] = A[b, c] = 1
        edges[i1] = [min(a,d), max(a,d)]
        edges[i2] = [min(c,b), max(c,b)]
    return A
