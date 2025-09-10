import argparse, json, os, datetime
import numpy as np
from scipy.signal import resample_poly
from ogc.t2_crosscoherence import coherence_band

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _save_json(obj, out_dir, sub):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(out_dir, sub)
    _ensure_dir(folder)
    path = os.path.join(folder, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")

# ----------------- T2 -----------------
def cmd_t2(args):
    # Synthese
    n = args.n
    T = 30.0
    fs = n / T
    t = np.linspace(0, T, n, endpoint=False)

    rng = np.random.default_rng(args.seed)
    x = np.sin(2*np.pi*0.8*t) + 0.5*np.sin(2*np.pi*2.0*t) + 0.05 * rng.normal(0, 1, n)
    y = np.sin(2*np.pi*0.8*t + 0.6) + 0.30 * rng.normal(0, 1, n)

    # Downsampling ~20 Hz
    target_fs = 20.0 if args.target_fs is None else float(args.target_fs)
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim

    # nperseg: 0 => auto
    nperseg = args.nperseg
    if nperseg in (None, 0):
        L = len(x_ds)
        nperseg = max(128, L // 6)
    if nperseg % 2 == 1:
        nperseg += 1

    band = (args.band_min, args.band_max)

    res = coherence_band(
        x_ds, y_ds,
        fs=fs_ds,
        band=band,
        nperseg=nperseg,
        n_null=args.n_null,
        rng=args.seed,
        mode=args.mode,
        null_mode=args.null_mode
    )

    out = {
        "params": {
            "out_dir": args.out_dir,
            "n": args.n,
            "n_null": args.n_null,
            "seed": args.seed,
            "null_mode": args.null_mode,
            "band_min": args.band_min,
            "band_max": args.band_max,
            "nperseg": nperseg,
            "target_fs": target_fs,
            "fs_ds": fs_ds,
            "band": list(band)
        },
        "result": res
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(out, args.out_dir, "t2")

# ----------------- T3 (unverändert) -----------------
def cmd_t3(args):
    # einfache Demo-Ausgabe: dein vorhandener Hysterese-Code
    from ogc.tests.t3_hysteresis import hysteresis_loop
    res = hysteresis_loop(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, seed=args.seed)
    out = {"params": {"n": args.n, "u_min": args.u_min, "u_max": args.u_max, "noise": args.noise, "seed": args.seed}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(out, args.out_dir, "t3")

# ----------------- S-Margin (unverändert) -----------------
def cmd_s_margin(args):
    from ogc.tests.s_margin import safety_margin
    res = safety_margin(loss_rate=args.loss, window=args.window)
    out = {"params": {"loss": args.loss, "window": args.window}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(out, args.out_dir, "s_margin")

# ----------------- SPLIT (unverändert) -----------------
def cmd_split(args):
    from ogc.tests.split_persistence import split_persistence
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    res = split_persistence(A, B, tol=args.tol)
    out = {"params": {"values_a": args.values_a, "values_b": args.values_b, "tol": args.tol}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(out, args.out_dir, "split")

# ----------------- C* (unverändert) -----------------
def cmd_cstar(args):
    from ogc.tests.cstar_longreturn import cstar_return_indicator
    rng = np.random.default_rng(args.seed)
    base = rng.binomial(1, 0.05, size=args.n).astype(float)
    if args.inject_echo:
        for k in range(args.echo_every, args.n, args.echo_every):
            base[k:min(k+3, args.n)] += 0.3
        base = np.clip(base, 0, 1)
    res = cstar_return_indicator(base, max_lag=args.max_lag, rng=args.seed)
    out = {"params": {"n": args.n, "max_lag": args.max_lag, "inject_echo": args.inject_echo, "echo_every": args.echo_every, "seed": args.seed}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(out, args.out_dir, "cstar")

# ----------------- MAIN -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=None, help="optional: Ergebnisse als JSON ablegen in diesem Ordner")

    sub = p.add_subparsers(dest="cmd", required=True)

    # T2
    p2 = sub.add_parser("t2")
    p2.add_argument("--n", type=int, default=12288)
    p2.add_argument("--n-null", type=int, default=2000)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--mode", type=str, default="mean", choices=["mean", "peak"])
    p2.add_argument("--null-mode", type=str, default="both", choices=["flip", "phase", "both"])
    p2.add_argument("--band-min", type=float, default=0.7)
    p2.add_argument("--band-max", type=float, default=0.9)
    p2.add_argument("--nperseg", type=int, default=0, help="0 = auto (≈ len/6), sonst fixer Wert")
    p2.add_argument("--target-fs", type=float, default=20.0, help="Downsample-Ziel (Hz)", dest="target_fs")
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

    # C*
    pc = sub.add_parser("cstar")
    pc.add_argument("--n", type=int, default=8000)
    pc.add_argument("--max-lag", type=int, default=400)
    pc.add_argument("--inject-echo", action="store_true")
    pc.add_argument("--echo-every", type=int, default=240)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_cstar)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
