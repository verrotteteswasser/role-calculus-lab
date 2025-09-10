# aggregate.py (Repo-Root oder beliebig, wird mit Pfad ausgeführt)
from __future__ import annotations
import os, json, argparse, glob
from statistics import mean
from datetime import datetime

def _read_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ----------------- T2 -----------------
def get_t2_row(path: str, since: datetime|None, accept_versions: set[str]) -> dict|None:
    try:
        data = _read_json(path)
    except Exception:
        return None

    schema = data.get("schema_version")
    if accept_versions and schema not in accept_versions:
        return None

    if since:
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(path))
            if ts < since:
                return None
        except Exception:
            pass

    params = data.get("params", {}) or {}
    res = data.get("result", {}) or {}

    # p_final robust bestimmen
    p_final = res.get("p_value_final")
    if p_final is None:
        # Fallback-Kaskade
        for k in ("p_value_flip", "p_value_phase", "p_value"):
            if res.get(k) is not None:
                p_final = res.get(k); break

    if p_final is None:
        return None

    stat = res.get("stat", res.get("peak"))
    return {
        "file": os.path.basename(path),
        "seed": params.get("seed"),
        "n": params.get("n"),
        "null_mode": params.get("null_mode"),
        "band": params.get("band"),
        "stat": stat,
        "p_value": p_final,
    }

def collect_t2(root: str, tag: str|None, since: datetime|None):
    base = os.path.join(root, tag, "t2") if tag else os.path.join(root, "t2")
    rows = []
    for p in glob.glob(os.path.join(base, "*.json")):
        row = get_t2_row(p, since, {"t2both-v1","t2flip-v1","t2phase-v1"})
        if row:
            rows.append(row)
    return rows

# ----------------- T3 -----------------
def get_t3_row(path: str, since: datetime|None, accept_versions: set[str]) -> dict|None:
    try:
        data = _read_json(path)
    except Exception:
        return None

    if accept_versions and data.get("schema_version") not in accept_versions:
        return None

    if since:
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(path))
            if ts < since:
                return None
        except Exception:
            pass

    params = data.get("params", {}) or {}
    res = data.get("result", {}) or {}
    return {
        "file": os.path.basename(path),
        "seed": params.get("seed"),
        "n": params.get("n"),
        "Theta_up": res.get("Theta_up"),
        "Theta_down": res.get("Theta_down"),
        "A_loop": res.get("A_loop"),
    }

def collect_t3(root: str, tag: str|None, since: datetime|None):
    base = os.path.join(root, tag, "t3") if tag else os.path.join(root, "t3")
    rows = []
    for p in glob.glob(os.path.join(base, "*.json")):
        row = get_t3_row(p, since, {"t3-v1"})
        if row:
            rows.append(row)
    return rows

# ----------------- CSTAR -----------------
def get_cstar_row(path: str, since: datetime|None, accept_versions: set[str]) -> dict|None:
    try:
        data = _read_json(path)
    except Exception:
        return None

    if accept_versions and data.get("schema_version") not in accept_versions:
        return None

    if since:
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(path))
            if ts < since:
                return None
        except Exception:
            pass

    params = data.get("params", {}) or {}
    res = data.get("result", {}) or {}
    return {
        "file": os.path.basename(path),
        "seed": params.get("seed"),
        "n": params.get("n"),
        "max_lag": params.get("max_lag"),
        "stat": res.get("stat"),
        "p_value": res.get("p_value"),
        "tail_mean_acf": res.get("tail_mean_acf"),
    }

def collect_cstar(root: str, tag: str|None, since: datetime|None):
    base = os.path.join(root, tag, "cstar") if tag else os.path.join(root, "cstar")
    rows = []
    for p in glob.glob(os.path.join(base, "*.json")):
        row = get_cstar_row(p, since, {"cstar-v1"})
        if row:
            rows.append(row)
    return rows

# ----------------- IO helpers -----------------
def _write_csv(path: str, rows: list[dict], header: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = []
            for h in header:
                v = r.get(h)
                if isinstance(v, list):
                    v = ";".join(str(x) for x in v)
                vals.append("" if v is None else str(v))
            f.write(",".join(vals) + "\n")

def _quick_stats_t2(rows: list[dict]) -> str:
    if not rows: return "T2: keine Dateien gefunden.\n"
    ps = [r["p_value"] for r in rows if r.get("p_value") is not None]
    s = [r["stat"] for r in rows if r.get("stat") is not None]
    return (
        f"T2: n_files={len(rows)}, mean p={mean(ps):.3g}, "
        f"min p={min(ps):.3g}, max p={max(ps):.3g}, "
        f"mean stat={mean(s) if s else float('nan'):.3g}\n"
    )

def _quick_stats_t3(rows: list[dict]) -> str:
    if not rows: return "T3: keine Dateien gefunden.\n"
    areas = [r["A_loop"] for r in rows if r.get("A_loop") is not None]
    return f"T3: n_files={len(rows)}, mean A_loop={mean(areas):.3g}\n"

def _quick_stats_cstar(rows: list[dict]) -> str:
    if not rows: return "C*: keine Dateien gefunden.\n"
    ps = [r["p_value"] for r in rows if r.get("p_value") is not None]
    return (
        f"C*: n_files={len(rows)}, mean p={mean(ps):.3g}, "
        f"min p={min(ps):.3g}, max p={max(ps):.3g}\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="result", help="Basis-Ordner der Runs")
    ap.add_argument("--tag", type=str, default=None, help="Nur Dateien unter result/<tag> lesen")
    ap.add_argument("--since", type=str, default=None, help="nur Dateien ab Datum (YYYY-MM-DD)")
    args = ap.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None

    t2_rows = collect_t2(args.root, args.tag, since)
    t3_rows = collect_t3(args.root, args.tag, since)
    cstar_rows = collect_cstar(args.root, args.tag, since)

    out_root = os.path.join(args.root, args.tag, "aggregate") if args.tag else os.path.join(args.root, "aggregate")
    _ensure_dir(out_root)

    _write_csv(os.path.join(out_root, "t2_rows.csv"), t2_rows, ["file","seed","n","null_mode","band","stat","p_value"])
    _write_csv(os.path.join(out_root, "t3_rows.csv"), t3_rows, ["file","seed","n","Theta_up","Theta_down","A_loop"])
    _write_csv(os.path.join(out_root, "cstar_rows.csv"), cstar_rows, ["file","seed","n","max_lag","stat","p_value","tail_mean_acf"])

    report = (
        _quick_stats_t2(t2_rows) +
        _quick_stats_t3(t3_rows) +
        _quick_stats_cstar(cstar_rows)
    )
    with open(os.path.join(out_root, "quick_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)

if __name__ == "__main__":
    main()
