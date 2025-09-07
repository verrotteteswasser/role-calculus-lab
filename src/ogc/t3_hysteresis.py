import numpy as np

def hysteresis_loop(n=300, u_min=0.0, u_max=2.0, noise=0.0, seed=0):
    """
    T3: Hysterese/Plateaus – generiert Up/Down Kurven + A_loop.
    (Deckt dein bestehendes Ergebnis ab, hier als Wrapper.)
    """
    rng = np.random.default_rng(seed)
    u_up   = np.linspace(u_min, u_max, n)
    u_down = u_up[::-1]
    # Beispiel: S-Kurve + ggf. Rauschen
    def s_curve(u, th=0.6, k=10.0): return 1.0/(1.0 + np.exp(-k*(u-th)))
    y_up   = s_curve(u_up) + rng.normal(0, noise, size=n)
    y_down = s_curve(u_down*1.05) + rng.normal(0, noise, size=n)  # leichte Verschiebung => Hysterese
    # Shoelace/Polygon area:
    x = np.concatenate([u_up,  u_down])
    y = np.concatenate([y_up, -y_down])
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return {
        "Theta_up": float(u_up[np.argmax(np.gradient(y_up))]),
        "Theta_down": float(u_down[np.argmax(np.gradient(y_down))]),
        "A_loop": float(area),
        "u_up": u_up.tolist(), "y_up": y_up.tolist(),
        "u_down": u_down.tolist(), "y_down": y_down.tolist()
    }

