\
import numpy as np

def bistable_response(u, a=1.0, b=1.0, c=0.0, noise=0.0, rng=None):
    """
    Simple saturating cubic with relaxation; b>0 dämpft.
    """
    rng = np.random.default_rng(rng)
    y = 0.0
    ys = []
    for ui in u:
        for _ in range(30):
            y = 0.8*y + 0.2*(a*ui - b*(y**3) + c)
            y = float(np.clip(y, -5.0, 5.0))  # sanfte Sättigung
        if noise > 0:
            y += rng.normal(0, noise)
        ys.append(y)
    return np.array(ys)

def hysteresis_loop_area(x_up, y_up, x_down, y_down):
    xs = np.concatenate([x_up, x_down[::-1]])
    ys = np.concatenate([y_up, y_down[::-1]])
    area = 0.5*np.abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    return area

def simulate_hysteresis(n=200, u_min=0.0, u_max=2.0, noise=0.0, rng=None):
    rng = np.random.default_rng(rng)
    u_up = np.linspace(u_min, u_max, n)
    y_up = bistable_response(u_up, noise=noise, rng=rng)
    u_down = np.linspace(u_max, u_min, n)
    y_down = bistable_response(u_down, noise=noise, rng=rng)
    area = hysteresis_loop_area(u_up, y_up, u_down, y_down)
    med = (y_up.min() + y_up.max())/2
    th_up = float(u_up[np.argmax(y_up>med)])
    th_down = float(u_down[np.argmax(y_down>med)])
    return {"Theta_up": th_up, "Theta_down": th_down, "A_loop": float(area),
            "u_up": u_up.tolist(), "y_up": y_up.tolist(),
            "u_down": u_down.tolist(), "y_down": y_down.tolist()}
