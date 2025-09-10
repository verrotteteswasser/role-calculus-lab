# src/ogc/cli.py
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _save_json(obj: dict, out_dir: str, tag: str, task: str) -> str:
    base = os.path.join(out_dir, tag, task)
    _ensure_dir(base)
    path = os.path.join(base, f"{_timestamp()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[saved] {os.path.relpath(path)}")
    return path

def _clean_params(args):
    p = dict(vars(args))
    p.pop("func", None)
    # robust gegen Nicht-JSON-Typen
    for k, v in list(p.items()):
        try:
            json.dumps(v)
        except Exception:
            p[k] = str(v)
    return p

# ---------------- T1 ----------------
def cmd_t1(args):
    # Platzhalter (kein eigener Test im Repo)
    res = {"ok": True}
    payload = {
        "schema_version": "t1-v1",
        "task": "t1",
        "params": _clean_params(args),
        "result": res,
    }
    _save_json(payload, args.out_dir, args.tag, "t1")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# ---------------- T2 ----------------
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

    # Downsampling ~20 Hz
    target_fs = 20.0
    decim = max(1, int(round(fs / target_fs)))
    x_ds = resample_poly(x, up=1, down=decim)
    y_ds = resample_poly(y, up=1, down=decim)
    fs_ds = fs / decim
    L = len(x_ds)

    # Welch-Setup: mehrere Segmente
    K = 6
    nperseg = max(128, (L // K))
    if nperseg % 2 == 1:
        nperseg += 1

    band = (args.band_min, args.band_max)

    # Null-Modi ausführen
    def run(mode: str):
        return coherence_band(
            x_ds, y_ds,
            fs=fs_ds,
            band=band,
            nperseg=nperseg,
            n_null=args.n_null,
            rng=args.seed,
            mode="mean",
            null_mode=mode
        )

    if args.null_mode == "flip":
        res_flip = run("flip")
        res_phase = {}
        p_final = res_flip.get("p_value")
        schema = "t2flip-v1"
    elif args.null_mode == "phase":
        res_phase = run("phase")
        res_flip = {}
        p_final = res_phase.get("p_value")
        schema = "t2phase-v1"
    else:  # both
        res_flip = run("flip")
        res_phase = run("phase")
        # konservativ: nimm die größere p-Value (= strengere Evidenz)
        p_flip = res_flip.get("p_value")
        p_phase = res_phase.get("p_value")
        # Falls phase-Null technisch extrem (0.0), lassen wir sie trotzdem im JSON.
        p_final = max([p for p in [p_flip, p_phase] if p is not None])
        schema = "t2both-v1"

    # „stat“ = Band-Statistik (mean Kohärenz)
    stat = None
    for r in (res_flip, res_phase):
        if r and "peak" in r or "stat" in r:
            # unser t2_crosscoherence gibt „peak“/„stat“; wir nutzen „stat“ wenn vorhanden,
            # ansonsten „peak“ (zur Kompatibilität)
            stat = r.get("stat", r.get("peak"))
            break

    payload = {
        "schema_version": schema,
        "task": "t2",
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
            "fs_ds": float(fs_ds),
            "band": [float(band[0]), float(band[1])],
        },
        "result": {
            "stat": stat,
            "band_fraction": res_flip.get("band_fraction") or res_phase.get("band_fraction"),
            "mode": "mean",
            "null_mode": args.null_mode,
            "p_value_flip": res_flip.get("p_value"),
            "p_value_phase": res_phase.get("p_value"),
            "p_value_final": p_final,
            "decision_alpha_0.05": (p_final is not None and p_final < 0.05),
        }
    }

    _save_json(payload, args.out_dir, args.tag, "t2")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# ---------------- T3 ----------------
def cmd_t3(args):
    # Hysterese-Test — erzeugt Loop mit Kennzahlen
    from ogc.tests.t3_hysteresis import hysteresis_loop
    res = hysteresis_loop(n=args.n, u_min=args.u_min, u_max=args.u_max, noise=args.noise, seed=args.seed)

    payload = {
        "schema_version": "t3-v1",
        "task": "t3",
        "params": _clean_params(args),
        "result": res,
    }
    _save_json(payload, args.out_dir, args.tag, "t3")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# --------------- Safety Margin ---------------
def cmd_s_margin(args):
    from ogc.tests.s_margin import safety_margin
    res = safety_margin(loss_rate=args.loss, window=args.window)
    payload = {
        "schema_version": "safety-v1",
        "task": "s_margin",
        "params": _clean_params(args),
        "result": res,
    }
    _save_json(payload, args.out_dir, args.tag, "s_margin")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# --------------- Split Persistence ---------------
def cmd_split(args):
    from ogc.tests.split_persistence import split_persistence
    A = [float(v) for v in args.values_a.split(",")]
    B = [float(v) for v in args.values_b.split(",")]
    res = split_persistence(A, B, tol=args.tol)
    payload = {
        "schema_version": "split-v1",
        "task": "split",
        "params": _clean_params(args),
        "result": res,
    }
    _save_json(payload, args.out_dir, args.tag, "split")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# --------------- C* Long Return ---------------
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
    payload = {
        "schema_version": "cstar-v1",
        "task": "cstar",
        "params": _clean_params(args),
        "result": res,
    }
    _save_json(payload, args.out_dir, args.tag, "cstar")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

# --------------- CLI ---------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="result")
    p.add_argument("--tag", type=str, default="latest", help="Unterordner unter out-dir (z.B. v2025-09-08)")
    sub = p.add_subparsers()

    p1 = sub.add_parser("t1");           p1.set_defaults(func=cmd_t1)

    p2 = sub.add_parser("t2")
    p2.add_argument("--n", type=int, default=4096)
    p2.add_argument("--n-null", type=int, default=300)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--null-mode", choices=["flip","phase","both"], default="both")
    p2.add_argument("--band-min", type=float, default=0.7)
    p2.add_argument("--band-max", type=float, default=0.9)
    p2.set_defaults(func=cmd_t2)

    p3 = sub.add_parser("t3")
    p3.add_argument("--n", type=int, default=300)
    p3.add_argument("--u-min", type=float, default=0.0)
    p3.add_argument("--u-max", type=float, default=2.0)
    p3.add_argument("--noise", type=float, default=0.0)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=cmd_t3)

    ps = sub.add_parser("s_margin")
    ps.add_argument("--loss", type=float, default=0.2)
    ps.add_argument("--window", type=float, default=3.0)
    ps.set_defaults(func=cmd_s_margin)

    pp = sub.add_parser("split")
    pp.add_argument("--values-a", type=str, required=True)
    pp.add_argument("--values-b", type=str, required=True)
    pp.add_argument("--tol", type=float, default=1e-3)
    pp.set_defaults(func=cmd_split)

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
