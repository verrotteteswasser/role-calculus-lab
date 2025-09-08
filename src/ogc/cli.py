import argparse, json
import os, time

def _ts():
    return time.strftime("%Y%m%d-%H%M%S")

def _ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _save_json(payload, out_dir, subcmd):
    """
    Speichert payload als JSON unter:
      {out_dir}/{subcmd}/{YYYYmmdd-HHMMSS}.json
    und gibt den Pfad zurück.
    """
    folder = os.path.join(out_dir, subcmd)
    _ensure_dir(folder)
    path = os.path.join(folder, f"{_ts()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def _clean_params(args):
    """Entfernt nicht-serialisierbare Dinge (z.B. args.func) aus argparse-Params."""
    p = dict(vars(args))
    p.pop("func", None)
    for k, v in list(p.items()):
        try:
            json.dumps(v)
        except Exception:
            p[k] = str(v)
    return p

def cmd_t1(args):
    from ogc.tests.t1_orientation import orientation_identity
    res = orientation_identity(None, None)  # später echte Inputs
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "t1")
    print(f"[saved] {save_path}")

def cmd_t2(args):
    import numpy as np
    from scipy.signal import resample_poly
    from ogc.t2_crosscoherence import coherence_band

    # Synthese
    n = args.n
    T = 30.0
    fs = n / T
    t = np.linspace(0, T, n, endpoint=False)

    rng = np.random.default_rng(args.seed)
    x = np.sin(2*np.pi*0.8*t) + 0.5*np.sin(2*np.pi*2.0*t) + 0.05 * rng.normal(0, 1, n)
    y = np.sin(2*np.pi*0.8*t + 0.6) + 0.30 * rng.normal(0, 1, n)

    # Downsampling auf ~20 Hz
    target_fs = 20.0
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim
    L = len(x_ds)

    # Welch: mehrere Segmente
    K = 6
    nperseg = max(128, (L // K))
    if nperseg % 2 == 1:
        nperseg += 1

    # Band aus CLI-Parametern
    band = (args.band_min, args.band_max)

    # Null-Hypothesen behandeln
    if args.null_mode == "both":
        res_flip = coherence_band(
            x_ds, y_ds,
            fs=fs_ds, band=band, nperseg=nperseg,
            n_null=args.n_null, rng=args.seed,
            mode="mean", null_mode="flip"
        )
        res_phase = coherence_band(
            x_ds, y_ds,
            fs=fs_ds, band=band, nperseg=nperseg,
            n_null=args.n_null, rng=args.seed,
            mode="mean", null_mode="phase"
        )
        p_flip = res_flip["p_value"]
        p_phase = res_phase["p_value"]
        p_final = max(p_flip, p_phase)  # konservativ (beide Nullmodelle bestehen)

        res = {
            "stat": res_flip["stat"],
            "band_fraction": res_flip["band_fraction"],
            "mode": "mean",
            "null_mode": "both",
            "p_value_flip": p_flip,
            "p_value_phase": p_phase,
            "p_value_final": p_final,
            "decision_alpha_0.05": (p_final < 0.05),
        }
    else:
        res = coherence_band(
            x_ds, y_ds,
            fs=fs_ds, band=band, nperseg=nperseg,
            n_null=args.n_null, rng=args.seed,
            mode="mean", null_mode=args.null_mode
        )

    # Ausgabe + Save
    params = _clean_params(args)
    params.update({
        "fs_ds": fs_ds,
        "nperseg": nperseg,
        "band": list(band),
    })
    out = {"params": params, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "t2")
    print(f"[saved] {save_path}")

def cmd_t3(args):
    from ogc.tests.t3_hysteresis import hysteresis_loop
    res = hysteresis_loop(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, seed=args.seed)
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "t3")
    print(f"[saved] {save_path}")

def cmd_s_margin(args):
    from ogc.tests.s_margin import safety_margin
    res = safety_margin(loss_rate=args.loss, window=args.window)
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "s_margin")
    print(f"[saved] {save_path}")

def cmd_split(args):
    from ogc.tests.split_persistence import split_persistence
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    res = split_persistence(A, B, tol=args.tol)
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "split")
    print(f"[saved] {save_path}")

def cmd_cstar(args):
    from ogc.tests.cstar_longreturn import cstar_return_indicator
    import numpy as np
    rng = np.random.default_rng(args.seed)
    base = rng.binomial(1, 0.05, size=args.n).astype(float)
    if args.inject_echo:
        for k in range(args.echo_every, args.n, args.echo_every):
            base[k:min(k+3, args.n)] += 0.3
        base = np.clip(base, 0, 1)
    res = cstar_return_indicator(base, max_lag=args.max_lag, rng=args.seed)
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = _save_json(out, args.out_dir, "cstar")
    print(f"[saved] {save_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="result", help="Basisordner für JSON-Outputs")
    sub = p.add_subparsers()

    # T1
    p1 = sub.add_parser("t1")
    p1.set_defaults(func=cmd_t1)

    # T2
    p2 = sub.add_parser("t2")
    p2.add_argument("--n", type=int, default=4096)
    p2.add_argument("--n-null", type=int, default=300)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--band-min", type=float, default=0.7)
    p2.add_argument("--band-max", type=float, default=0.9)
    p2.add_argument(
        "--null-mode",
        choices=["flip", "phase", "both"],
        default="flip",
        help="Null-Erzeugung: 'flip' (Segment-Flips), 'phase' (Phase-only Surrogates) oder 'both'"
    )
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
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()