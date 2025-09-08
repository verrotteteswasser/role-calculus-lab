python aggregate.py          # wertet result/ aus
python aggregate.py --root result  # (optional) explizit

# aggregate.py
# Stdlib-only Auswertung für result/ JSONs (T2, T3, C*).
import argparse, json, os, csv
from pathlib import Path
from statistics import mean, pstdev

def walk_jsons(root: Path):
    for p in root.rglob("*.json"):
        # nur echte Messergebnisse (keine evtl. Config-Dateien)
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "result" in data:
                yield p, data
        except Exception:
            continue

def kind_of(sample: dict) -> str:
    r = sample.get("result", {})
    # T2: Kohärenz
    if "band_fraction" in r and ("p_value" in r or "p_value_final" in r or "mode" in r):
        return "t2"
    # T3: Hysterese
    if {"A_loop", "Theta_up", "Theta_down"} <= set(r.keys()):
        return "t3"
    # C*: Long-Return
    if {"tail_mean_acf", "p_value"} <= set(r.keys()):
        return "cstar"
    return "other"

def to_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def summarize_t2(rows):
    # rows: list of (path, data)
    vals = []
    for p, d in rows:
        res = d["result"]
        par = d.get("params", {})
        stat = to_float(res.get("stat"))
        band_fraction = to_float(res.get("band_fraction"))
        # p-Wert bevorzugt: p_value_final > p_value_flip > p_value
        p_final = (res.get("p_value_final", res.get("p_value_flip", res.get("p_value"))))
        p_final = to_float(p_final)
        decision = res.get("decision_alpha_0.05")
        if decision is None and p_final is not None:
            decision = bool(p_final < 0.05)
        vals.append({
            "path": str(p),
            "seed": par.get("seed"),
            "n": par.get("n"),
            "n_null": par.get("n_null"),
            "fs_ds": par.get("fs_ds"),
            "nperseg": par.get("nperseg"),
            "band": par.get("band") or [par.get("band_min"), par.get("band_max")],
            "stat": stat,
            "band_fraction": band_fraction,
            "p_value": p_final,
            "sig@0.05": decision,
            "null_mode": res.get("null_mode") or par.get("null_mode"),
            "mode": res.get("mode"),
        })
    if not vals:
        return {"count": 0}, vals

    def mget(key):
        arr = [to_float(v[key]) for v in vals if to_float(v[key]) is not None]
        return (mean(arr), pstdev(arr)) if arr else (None, None)

    m_stat, s_stat = mget("stat")
    m_bandf, s_bandf = mget("band_fraction")
    m_p, s_p = mget("p_value")
    sig_rate = None
    sigs = [1 for v in vals if v["sig@0.05"] is True]
    if vals:
        sig_rate = len(sigs) / len(vals)

    # Band-Info versuchen
    bands = [tuple(v["band"]) for v in vals if v.get("band") and all(x is not None for x in v["band"])]
    band_str = None
    if bands:
        # wenn alle gleich, zeig eins
        if len(set(bands)) == 1:
            band_str = str(bands[0])
        else:
            band_str = f"{min(b[0] for b in bands):.3f}..{max(b[1] for b in bands):.3f}"

    summary = {
        "count": len(vals),
        "band": band_str,
        "stat_mean": m_stat, "stat_std": s_stat,
        "band_fraction_mean": m_bandf, "band_fraction_std": s_bandf,
        "p_mean": m_p, "p_std": s_p,
        "sig_rate@0.05": sig_rate,
        "null_modes": sorted(set(str(v.get("null_mode")) for v in vals)),
        "modes": sorted(set(str(v.get("mode")) for v in vals)),
    }
    return summary, vals

def summarize_t3(rows):
    vals = []
    for p, d in rows:
        r = d["result"]
        par = d.get("params", {})
        vals.append({
            "path": str(p),
            "seed": par.get("seed"),
            "n": par.get("n"),
            "noise": par.get("noise"),
            "A_loop": to_float(r.get("A_loop")),
            "Theta_up": to_float(r.get("Theta_up")),
            "Theta_down": to_float(r.get("Theta_down")),
        })
    if not vals:
        return {"count": 0}, vals

    def mget(key):
        arr = [to_float(v[key]) for v in vals if to_float(v[key]) is not None]
        return (mean(arr), pstdev(arr)) if arr else (None, None)

    mA, sA = mget("A_loop")
    mUp, sUp = mget("Theta_up")
    mDn, sDn = mget("Theta_down")

    summary = {
        "count": len(vals),
        "A_loop_mean": mA, "A_loop_std": sA,
        "Theta_up_mean": mUp, "Theta_up_std": sUp,
        "Theta_down_mean": mDn, "Theta_down_std": sDn,
    }
    return summary, vals

def summarize_cstar(rows):
    vals = []
    for p, d in rows:
        r = d["result"]
        par = d.get("params", {})
        vals.append({
            "path": str(p),
            "seed": par.get("seed"),
            "n": par.get("n"),
            "max_lag": par.get("max_lag"),
            "stat": to_float(r.get("stat")),
            "p_value": to_float(r.get("p_value")),
            "tail_mean_acf": to_float(r.get("tail_mean_acf")),
        })
    if not vals:
        return {"count": 0}, vals

    def mget(key):
        arr = [to_float(v[key]) for v in vals if to_float(v[key]) is not None]
        return (mean(arr), pstdev(arr)) if arr else (None, None)

    mstat, sstat = mget("stat")
    mp, sp = mget("p_value")
    mt, st = mget("tail_mean_acf")
    sig_rate = None
    ps = [v for v in vals if v["p_value"] is not None and v["p_value"] < 0.05]
    sig_rate = len(ps) / len(vals)

    summary = {
        "count": len(vals),
        "stat_mean": mstat, "stat_std": sstat,
        "p_mean": mp, "p_std": sp,
        "sig_rate@0.05": sig_rate,
        "tail_mean_acf_mean": mt, "tail_mean_acf_std": st,
    }
    return summary, vals

def write_csv(path: Path, rows, header):
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result", help="Wurzelordner mit JSON-Ergebnissen")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[!] Ordner nicht gefunden: {root}")
        return

    buckets = {"t2": [], "t3": [], "cstar": [], "other": []}
    for p, data in walk_jsons(root):
        buckets[kind_of(data)].append((p, data))

    summary_dir = root / "summary"
    ensure_dir(summary_dir)

    # T2
    t2_summary, t2_rows = summarize_t2(buckets["t2"])
    write_csv(summary_dir / "t2_rows.csv", t2_rows,
              ["path","seed","n","n_null","fs_ds","nperseg","band","stat","band_fraction","p_value","sig@0.05","null_mode","mode"])
    # T3
    t3_summary, t3_rows = summarize_t3(buckets["t3"])
    write_csv(summary_dir / "t3_rows.csv", t3_rows,
              ["path","seed","n","noise","A_loop","Theta_up","Theta_down"])
    # C*
    cs_summary, cs_rows = summarize_cstar(buckets["cstar"])
    write_csv(summary_dir / "cstar_rows.csv", cs_rows,
              ["path","seed","n","max_lag","stat","p_value","tail_mean_acf"])

    # Quick report
    report = []
    report.append("# Quick Report\n")
    report.append("## T2 – Cross-Coherence\n")
    report.append(json.dumps(t2_summary, indent=2, ensure_ascii=False))
    report.append("\n## T3 – Hysteresis\n")
    report.append(json.dumps(t3_summary, indent=2, ensure_ascii=False))
    report.append("\n## C* – Long Return\n")
    report.append(json.dumps(cs_summary, indent=2, ensure_ascii=False))
    txt = "\n".join(report)

    with (summary_dir / "quick_report.txt").open("w", encoding="utf-8") as f:
        f.write(txt)

    print(f"[ok] Dateien geschrieben nach: {summary_dir}")
    print(" - t2_rows.csv, t3_rows.csv, cstar_rows.csv")
    print(" - quick_report.txt")
    print("\nKurzfassung:")
    print(txt)

if __name__ == "__main__":
    main()
