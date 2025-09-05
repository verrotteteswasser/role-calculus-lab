\
import argparse, json
import numpy as np
from ogc.t1_orientation import synthetic_stack, fit_t1
from ogc.t2_crosscoherence import coherence_band
from ogc.t3_hysteresis import simulate_hysteresis

def cmd_t1(args):
    r, v_r, C_true = synthetic_stack(a=args.a, D=args.D, noise=args.noise, rng=args.seed)
    res = fit_t1(r, v_r, a=args.a, D=args.D)
    params = {k: v for k, v in vars(args).items() if k != "func"}
    out = {"params": params, "result": res}
    print(json.dumps(out, indent=2))

def cmd_t2(args):
    import numpy as np
    from scipy.signal import resample_poly
    from ogc.t2_crosscoherence import coherence_band

    n = args.n
    T = 10.0
    fs = n / T
    t = np.linspace(0, T, n, endpoint=False)

    # Synthese: gemeinsame ~0.8 Hz-Komponente
    x = np.sin(2*np.pi*0.8*t) + 0.5*np.sin(2*np.pi*2.0*t)
    y = np.sin(2*np.pi*0.8*t + 0.6) + 0.1*np.random.default_rng(args.seed).normal(0, 1, n)

    # --- Downsampling auf ~20 Hz (Anti-Alias via resample_poly) ---
    target_fs = 20.0
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim

    # Welch-Parameter: df ~ 0.04 Hz (512er Segmente bei 20 Hz)
    nperseg = 512

    res = coherence_band(
        x_ds, y_ds,
        fs=fs_ds,
        band=(0.7, 0.9),
        nperseg=nperseg,
        n_null=args.n_null,
        rng=args.seed,
        mode="mean"  # robuster als "peak"
    )

    params = {k: v for k, v in vars(args).items() if k != "func"}
    out = {"params": params, "result": res}
    print(json.dumps(out, indent=2))

def cmd_t3(args):
    res = simulate_hysteresis(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, rng=args.seed)
    params = {k: v for k, v in vars(args).items() if k != "func"}
    out = {"params": params, "result": res}
    print(json.dumps(out, indent=2))

def main():
    p = argparse.ArgumentParser(description="OGC Testlab CLI (T1/T2/T3)")
    sub = p.add_subparsers()

    p1 = sub.add_parser("t1", help="Run T1 orientation test on synthetic stack")
    p1.add_argument("--a", type=float, default=1.0)
    p1.add_argument("--D", type=int, default=3)
    p1.add_argument("--noise", type=float, default=0.0)
    p1.add_argument("--seed", type=int, default=0)
    p1.set_defaults(func=cmd_t1)

    p2 = sub.add_parser("t2", help="Run T2 cross-coherence test with phase-shuffle null")
    p2.add_argument("--n", type=int, default=2048)
    p2.add_argument("--n-null", dest="n_null", type=int, default=200)
    p2.add_argument("--seed", type=int, default=0)
    p2.set_defaults(func=cmd_t2)

    p3 = sub.add_parser("t3", help="Run T3 hysteresis simulation and metrics")
    p3.add_argument("--n", type=int, default=300)
    p3.add_argument("--u-min", dest="u_min", type=float, default=0.0)
    p3.add_argument("--u-max", dest="u_max", type=float, default=2.0)
    p3.add_argument("--noise", type=float, default=0.0)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=cmd_t3)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
