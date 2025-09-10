import argparse, json, os, time
from datetime import datetime

def _nowstamp():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _save_json(out_dir, sub, payload):
    _ensure_dir(os.path.join(out_dir, sub))
    path = os.path.join(out_dir, sub, f"{_nowstamp()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")

# ---------------------------------------------------------
# T1 (stub bleibt)
# ---------------------------------------------------------
def cmd_t1(args):
    res = {"ok": True}
    out = {"params": {}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        _save_json(args.out_dir, "t1", out)

# ---------------------------------------------------------
# T2 – Cross-Coherence
# ---------------------------------------------------------
def cmd_t2(args):
    import numpy as np
    from scipy.signal import resample_poly
    from ogc.t2_crosscoherence import (
        coherence_band,
        coherence_band_all,
    )

    # ---------- Synthese ----------
    n = args.n
    T = 30.0
    fs = n / T
    t = np.linspace(0, T, n, endpoint=False)
    rng = np.random.default_rng(args.seed)

    # SNR-Parameter aus CLI
    x = (
        np.sin(2*np.pi*0.8*t)
        + 0.5*np.sin(2*np.pi*2.0*t)
        + args.snr_x * rng.normal(0, 1, n)
    )
    y = (
        np.sin(2*np.pi*0.8*t + 0.6)
        + args.snr_y * rng.normal(0, 1, n)
    )

    # ---------- Downsampling ----------
    target_fs = float(args.target_fs)
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim
    L = len(x_ds)

    # ---------- nperseg wählen + Mindest-Binanzahl im Band ----------
    nperseg = int(args.nperseg)
    band = (args.band_min, args.band_max)

    def _bins_in_band(nps):
        fres = fs_ds / nps
        return int(np.floor((band[1] - band[0]) / fres))

    need_bins = max(1, args.min_bins)
    while _bins_in_band(nperseg) < need_bins and nperseg < L:
        nperseg *= 2
    if nperseg > L:
        nperseg = L // 2 if L >= 4 else max(2, L)

    # ---------- Auswertung ----------
    mode = args.mode
    trim = args.trim
    null_mode = args.null_mode

    if null_mode in ("all", "both"):
        # "both" (legacy) wird hier als "all" (flip+phase+shift) interpretiert
        res = coherence_band_all(
            x_ds, y_ds, fs=fs_ds, band=band, nperseg=nperseg,
            n_null=args.n_null, rng=args.seed, mode=mode, trim=trim
        )
    else:
        res = coherence_band(
            x_ds, y_ds, fs=fs_ds, band=band, nperseg=nperseg,
            n_null=args.n_null, rng=args.seed, mode=mode, trim=trim,
            null_mode=null_mode, shift_min_frac=args.shift_min_frac
        )

    # Entscheidung bei alpha=0.05 (falls p_final vorhanden, sonst p_value)
    p_final = res.get("p_value_final", res.get("p_value", 1.0))
    decision = bool(p_final < 0.05)

    out = {
        "params": {
            "out_dir": args.out_dir,
            "tag": args.tag,
            "n": n,
            "n_null": args.n_null,
            "seed": args.seed,
            "null_mode": null_mode,
            "mode": mode,
            "trim": trim,
            "nperseg": nperseg,
            "min_bins": need_bins,
            "target_fs": target_fs,
            "fs_ds": float(fs_ds),
            "band": [band[0], band[1]],
            "snr_x": args.snr_x,
            "snr_y": args.snr_y,
        },
        "result": {
            **res,
            "decision_alpha_0.05": decision,
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        # tag-Unterordner, wenn gesetzt
        save_root = args.out_dir if not args.tag else os.path.join(args.out_dir, args.tag)
        _save_json(save_root, "t2", out)

# ---------------------------------------------------------
# T3  (dein vorhandener Code; unverändert außer Save/Params)
# ---------------------------------------------------------
def cmd_t3(args):
    from ogc.tests.t3_hysteresis import hysteresis_loop
    res = hysteresis_loop(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, seed=args.seed)
    out = {"params": {"n": args.n, "u_min": args.u_min, "u_max": args.u_max, "noise": args.noise, "seed": args.seed},
           "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        save_root = args.out_dir if not args.tag else os.path.join(args.out_dir, args.tag)
        _save_json(save_root, "t3", out)

# ---------------------------------------------------------
# s_margin (unverändert, plus Save)
# ---------------------------------------------------------
def cmd_s_margin(args):
    from ogc.tests.s_margin import safety_margin
    res = safety_margin(loss_rate=args.loss, window=args.window)
    out = {"params": {"loss": args.loss, "window": args.window}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        save_root = args.out_dir if not args.tag else os.path.join(args.out_dir, args.tag)
        _save_json(save_root, "s_margin", out)

# ---------------------------------------------------------
# split (unverändert, plus Save)
# ---------------------------------------------------------
def cmd_split(args):
    from ogc.tests.split_persistence import split_persistence
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    res = split_persistence(A, B, tol=args.tol)
    out = {"params": {"values_a": args.values_a, "values_b": args.values_b, "tol": args.tol}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        save_root = args.out_dir if not args.tag else os.path.join(args.out_dir, args.tag)
        _save_json(save_root, "split", out)

# ---------------------------------------------------------
# cstar (unverändert, plus Save)
# ---------------------------------------------------------
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
    out = {"params": {"n": args.n, "max_lag": args.max_lag, "inject_echo": args.inject_echo,
                      "echo_every": args.echo_every, "seed": args.seed}, "result": res}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_dir:
        save_root = args.out_dir if not args.tag else os.path.join(args.out_dir, args.tag)
        _save_json(save_root, "cstar", out)

# ---------------------------------------------------------
# Parser
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="", help="Schreibe JSONs hierhin (optional).")
    p.add_argument("--tag", type=str, default="", help="Unterordner unter out-dir (optional).")
    sub = p.add_subparsers()

    # T1
    p1 = sub.add_parser("t1");           p1.set_defaults(func=cmd_t1)
    p1.add_argument("--dummy", action="store_true")

    # T2
    p2 = sub.add_parser("t2");           p2.set_defaults(func=cmd_t2)
    p2.add_argument("--n", type=int, default=12288)
    p2.add_argument("--n-null", type=int, default=5000)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--mode", type=str, default="trimmean", choices=["mean","peak","trimmean"])
    p2.add_argument("--trim", type=float, default=0.10)
    p2.add_argument("--null-mode", type=str, default="all", choices=["flip","phase","shift","both","all"])
    p2.add_argument("--shift-min-frac", type=float, default=0.10)
    p2.add_argument("--band-min", type=float, default=0.7)
    p2.add_argument("--band-max", type=float, default=0.9)
    p2.add_argument("--nperseg", type=int, default=128)
    p2.add_argument("--min-bins", type=int, default=4, help="minimale #Frequenzbins im Band")
    p2.add_argument("--target-fs", type=float, default=20.0)
    p2.add_argument("--snr-x", type=float, default=0.05)
    p2.add_argument("--snr-y", type=float, default=0.30)

    # T3
    p3 = sub.add_parser("t3");           p3.set_defaults(func=cmd_t3)
    p3.add_argument("--n", type=int, default=300)
    p3.add_argument("--u-min", type=float, default=0.0)
    p3.add_argument("--u-max", type=float, default=2.0)
    p3.add_argument("--noise", type=float, default=0.0)
    p3.add_argument("--seed", type=int, default=0)

    # Safety margin
    ps = sub.add_parser("s_margin");     ps.set_defaults(func=cmd_s_margin)
    ps.add_argument("--loss", type=float, default=0.2)
    ps.add_argument("--window", type=float, default=3.0)

    # Split
    pp = sub.add_parser("split");        pp.set_defaults(func=cmd_split)
    pp.add_argument("--values-a", type=str, required=True)
    pp.add_argument("--values-b", type=str, required=True)
    pp.add_argument("--tol", type=float, default=1e-3)

    # C*
    pc = sub.add_parser("cstar");        pc.set_defaults(func=cmd_cstar)
    pc.add_argument("--n", type=int, default=8000)
    pc.add_argument("--max-lag", type=int, default=400)
    pc.add_argument("--inject-echo", action="store_true")
    pc.add_argument("--echo-every", type=int, default=240)
    pc.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
