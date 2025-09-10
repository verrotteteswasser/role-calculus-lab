import argparse, json, os, datetime
import numpy as np
from scipy.signal import resample_poly
from ogc.t2_crosscoherence import coherence_band

def _clean_params(args):
    p = dict(vars(args))
    p.pop("func", None)
    return p

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def cmd_t2(args):
    # Synthese
    n = args.n
    T = 30.0
    fs = n / T
    t = np.linspace(0, T, n, endpoint=False)

    rng = np.random.default_rng(args.seed)
    x = np.sin(2*np.pi*0.8*t) + 0.5*np.sin(2*np.pi*2.0*t) + 0.05 * rng.normal(0, 1, n)
    y = np.sin(2*np.pi*0.8*t + 0.6) + 0.30 * rng.normal(0, 1, n)

    # Downsampling auf ~target_fs
    target_fs = args.target_fs
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim

    # Welch nperseg
    nperseg = args.nperseg
    if nperseg is None:
        # ca. 5 Segmente
        L = len(x_ds)
        nperseg = max(128, (L // 6))
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
            "n": n,
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

    # speichern
    if args.out_dir:
        dest = os.path.join(args.out_dir, "t2")
        _ensure_dir(dest)
        fname = os.path.join(dest, f"{_nowstamp()}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {fname}")

def cmd_t3(args):
    # Minimal-Dummy: liefert deterministische Hysterese-Kurve (wie vorher)
    n = args.n
    u = np.linspace(args.u_min, args.u_max, n)
    # simple modellierte Schleife:
    theta_up = 0.6020066889632107
    theta_down = 2.0
    # konstruierte y (nur Demo)
    y_up = 1 - np.exp(-5*u)
    y_down = y_up[::-1]
    A_loop = float(np.trapz(y_up, u) + np.trapz(y_down, u))  # pseudo-area

    res = {
        "Theta_up": theta_up,
        "Theta_down": theta_down,
        "A_loop": A_loop,
        "u_up": u.tolist(),
        "y_up": y_up.tolist(),
        "u_down": u[::-1].tolist(),
        "y_down": y_down.tolist(),
    }

    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_dir:
        dest = os.path.join(args.out_dir, "t3")
        _ensure_dir(dest)
        fname = os.path.join(dest, f"{_nowstamp()}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {fname}")

def cmd_s_margin(args):
    S = (1 - args.loss) * args.window
    res = {"S": S, "loss_rate": args.loss, "window": args.window}
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_dir:
        dest = os.path.join(args.out_dir, "s_margin")
        _ensure_dir(dest)
        fname = os.path.join(dest, f"{_nowstamp()}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {fname}")

def cmd_split(args):
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    diff = float(np.mean([(a-b)**2 for a,b in zip(A,B)]))
    res = {"pass": (diff <= args.tol), "diff": diff, "tol": args.tol}
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_dir:
        dest = os.path.join(args.out_dir, "split")
        _ensure_dir(dest)
        fname = os.path.join(dest, f"{_nowstamp()}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {fname}")

def cmd_cstar(args):
    rng = np.random.default_rng(args.seed)
    base = rng.binomial(1, 0.05, size=args.n).astype(float)
    if args.inject_echo:
        for k in range(args.echo_every, args.n, args.echo_every):
            base[k:min(k+3, args.n)] += 0.3
        base = np.clip(base, 0, 1)

    # Simple Indikator: Tail-Mean der ACF
    x = base - base.mean()
    acf = np.correlate(x, x, mode="full")[len(x)-1:] / (np.var(x)*len(x) + 1e-12)
    tail = acf[1:args.max_lag+1]
    stat = float(np.mean(tail))
    # Null: Shuffle
    null = []
    for _ in range(1000):
        xs = rng.permutation(base)
        xx = xs - xs.mean()
        acfs = np.correlate(xx, xx, mode="full")[len(xx)-1:] / (np.var(xx)*len(xx) + 1e-12)
        null.append(float(np.mean(acfs[1:args.max_lag+1])))
    null = np.array(null)
    p_value = float((null >= stat).mean())

    res = {"stat": stat, "p_value": p_value, "tail_mean_acf": stat}
    out = {"params": _clean_params(args), "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_dir:
        dest = os.path.join(args.out_dir, "cstar")
        _ensure_dir(dest)
        fname = os.path.join(dest, f"{_nowstamp()}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {fname}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=None)

    sub = p.add_subparsers()

    # T2
    p2 = sub.add_parser("t2")
    p2.add_argument("--n", type=int, default=12288)
    p2.add_argument("--n-null", type=int, default=5000)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--mode", type=str, default="mean", choices=["mean","peak"])
    p2.add_argument("--null-mode", type=str, default="both", choices=["flip","phase","both"])
    p2.add_argument("--band-min", type=float, default=0.7)
    p2.add_argument("--band-max", type=float, default=0.9)
    p2.add_argument("--nperseg", type=int, default=128)
    p2.add_argument("--target-fs", type=float, default=20.0)
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
    pc.add_argument("--n", type=int, default=8000)
    pc.add_argument("--max-lag", type=int, default=400)
    pc.add_argument("--inject-echo", action="store_true")
    pc.add_argument("--echo-every", type=int, default=240)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_cstar)

    args = p.parse_args()
    # out_dir am Top-Level:
    if hasattr(args, "out_dir"):
        # subparser-args erben das Attribut; nichts weiter nötig
        pass
    args.func(args)

if __name__ == "__main__":
    main()
