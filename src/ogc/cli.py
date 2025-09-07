import argparse, json

def cmd_t1(args):
    from ogc.tests.t1_orientation import orientation_identity
    res = orientation_identity(None, None)  # später echte Inputs
    print(json.dumps({"params": {}, "result": res}, ensure_ascii=False, indent=2))

def cmd_t2(args):
    # Deine bestehende T2 (Cross-Coherence) weiterverwenden:
    from ogc.t2_crosscoherence import coherence_band
    import numpy as np
    n = args.n; T = 10.0; fs = n / T
    t = np.linspace(0, T, n, endpoint=False)
    x = np.sin(2*np.pi*0.8*t) + 0.5*np.sin(2*np.pi*2.0*t)
    y = np.sin(2*np.pi*0.8*t + 0.6) + 0.1*np.random.default_rng(args.seed).normal(0, 1, n)
    res = coherence_band(x, y, fs=fs, band=(0.7,0.9), nperseg=min(512, n//4), n_null=args.n_null, rng=args.seed, mode="mean")
    print(json.dumps({"params": {"n": n, "n_null": args.n_null, "seed": args.seed}, "result": res}, ensure_ascii=False, indent=2))

def cmd_t3(args):
    from ogc.tests.t3_hysteresis import hysteresis_loop
    res = hysteresis_loop(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, seed=args.seed)
    print(json.dumps({"params": vars(args), "result": res}, ensure_ascii=False, indent=2))

def cmd_s_margin(args):
    from ogc.tests.s_margin import safety_margin
    res = safety_margin(loss_rate=args.loss, window=args.window)
    print(json.dumps({"params": vars(args), "result": res}, ensure_ascii=False, indent=2))

def cmd_split(args):
    from ogc.tests.split_persistence import split_persistence
    # Demo: zwei numerische Listen einlesen (Komma-separiert), später ersetzen durch echte Läufe
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    res = split_persistence(A, B, tol=args.tol)
    print(json.dumps({"params": vars(args), "result": res}, ensure_ascii=False, indent=2))

def cmd_cstar(args):
    from ogc.tests.cstar_longreturn import cstar_return_indicator
    # Demo: synthetische 0/1 Events; später echte C-Zählserie einspeisen
    import numpy as np
    rng = np.random.default_rng(args.seed)
    base = rng.binomial(1, 0.05, size=args.n).astype(float)
    # füge schwache Langzeitstruktur hinzu
    if args.inject_echo:
        for k in range(args.echo_every, args.n, args.echo_every):
            base[k:min(k+3, args.n)] += 0.3
        base = np.clip(base, 0, 1)
    res = cstar_return_indicator(base, max_lag=args.max_lag, rng=args.seed)
    print(json.dumps({"params": vars(args), "result": res}, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()

    # T1
    p1 = sub.add_parser("t1");           p1.set_defaults(func=cmd_t1)

    # T2
    p2 = sub.add_parser("t2")
    p2.add_argument("--n", type=int, default=4096)
    p2.add_argument("--n-null", type=int, default=300)
    p2.add_argument("--seed", type=int, default=0)
    p2.set_defaults(func=cmd_t2)

    # T3
    p3 = sub.add_parser("t3")
    p3.add_argument("--n", type=int, default=300)
    p3.add_argument("--u-min", type=float, default=0.0)
    p3.add_argument("--u-max", type=float, default=2.0)
    p3.add_argument("--noise", type=float, default=0.0)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=cmd_t3)

    # Safety margin
    ps = sub.add_parser("s_margin")
    ps.add_argument("--loss", type=float, default=0.2)
    ps.add_argument("--window", type=float, default=3.0)
    ps.set_defaults(func=cmd_s_margin)

    # Split persistence
    pp = sub.add_parser("split")
    pp.add_argument("--values-a", type=str, required=True)
    pp.add_argument("--values-b", type=str, required=True)
    pp.add_argument("--tol", type=float, default=1e-3)
    pp.set_defaults(func=cmd_split)

    # C* long return
    pc = sub.add_parser("cstar")
    pc.add_argument("--n", type=int, default=5000)
    pc.add_argument("--max-lag", type=int, default=200)
    pc.add_argument("--inject-echo", action="store_true")
    pc.add_argument("--echo-every", type=int, default=250)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_cstar)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
