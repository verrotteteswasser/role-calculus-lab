# aggregate.py
import argparse, json, os, datetime as dt
from collections import defaultdict

def _is_json(fname: str) -> bool:
    return fname.lower().endswith(".json")

def _load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__bad__": True, "__error__": str(e), "__path__": path}

def _mtime_iso(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return ""

def _within_since(path: str, since: str | None) -> bool:
    if not since:
        return True
    try:
        # akzeptiert "YYYY-MM-DD" oder "YYYY-MM-DD HH:MM[:SS]"
        try:
            since_dt = dt.datetime.fromisoformat(since)
        except ValueError:
            since_dt = dt.datetime.strptime(since, "%Y-%m-%d")
        return dt.datetime.fromtimestamp(os.path.getmtime(path)) >= since_dt
    except Exception:
        return True

def _row_t2(js, path):
    p = js.get("params", {})
    r = js.get("result", {})
    # robust: nimm p_value_final (null_mode=both), fallback auf p_value
    p_final = r.get("p_value_final", r.get("p_value"))
    band = r.get("band", p.get("band"))
    if not band and "band_min" in p and "band_max" in p:
        band = [p.get("band_min"), p.get("band_max")]
    return {
        "file": path,
        "mtime": _mtime_iso(path),
        "seed": p.get("seed"),
        "n": p.get("n"),
        "n_null": p.get("n_null"),
        "fs_ds": p.get("fs_ds"),
        "nperseg": p.get("nperseg"),
        "band_min": (band[0] if isinstance(band, (list, tuple)) and len(band) >= 1 else None),
        "band_max": (band[1] if isinstance(band, (list, tuple)) and len(band) >= 2 else None),
        "mode": r.get("mode"),
        "null_mode": r.get("null_mode"),
        "stat": r.get("stat"),
        "band_fraction": r.get("band_fraction"),
        "p_value_flip": r.get("p_value_flip"),
        "p_value_phase": r.get("p_value_phase"),
        "p_value_final": p_final,
        "decision_alpha_0.05": r.get("decision_alpha_0.05"),
    }

def _row_t3(js, path):
    p = js.get("params", {})
    r = js.get("result", {})
    return {
        "file": path,
        "mtime": _mtime_iso(path),
        "seed": p.get("seed"),
        "n": p.get("n"),
        "u_min": p.get("u_min"),
        "u_max": p.get("u_max"),
        "noise": p.get("noise"),
        "Theta_up": r.get("Theta_up"),
        "Theta_down": r.get("Theta_down"),
        "A_loop": r.get("A_loop"),
    }

def _row_cstar(js, path):
    p = js.get("params", {})
    r = js.get("result", {})
    return {
        "file": path,
        "mtime": _mtime_iso(path),
        "seed": p.get("seed"),
        "n": p.get("n"),
        "max_lag": p.get("max_lag"),
        "inject_echo": p.get("inject_echo"),
        "echo_every": p.get("echo_every"),
        "stat": r.get("stat"),
        "p_value": r.get("p_value"),
        "tail_mean_acf": r.get("tail_mean_acf"),
    }

def _task_from_path(path: str) -> str | None:
    # erwartet .../<root>/<task>/<timestamp>.json  -> task ∈ {t2,t3,cstar}
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        t = parts[-2].lower()
        if t in {"t2", "t3", "cstar"}:
            return t
    # fallback: anhand der keys im json -> eher langsam, vermeiden wir
    return None

def _write_csv(rows, out_path):
    if not rows:
        return
    # Felder vereinheitlichen
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(all_keys) + "\n")
        for r in rows:
            vals = []
            for k in all_keys:
                v = r.get(k)
                if isinstance(v, str):
                    # rudimentäre CSV-Escape (ohne Quote-Minimalismus)
                    vals.append('"' + v.replace('"', '""') + '"')
                elif v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

def _fmt_stats(vals):
    if not vals:
        return "n=0"
    return f"n={len(vals)}, mean={sum(vals)/len(vals):.3g}, min={min(vals):.3g}, max={max(vals):.3g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result", help="Wurzelordner der JSON-Runs (z.B. result oder src\\ogc\\result\\v2025-09-08)")
    ap.add_argument("--tag", default=None, help="Freitext-Tag für Report-Header")
    ap.add_argument("--since", default=None, help="Nur Dateien mit mtime >= since (YYYY-MM-DD[ HH:MM[:SS]])")
    ap.add_argument("--csv", action="store_true", help="CSV-Dateien (t2_rows.csv, t3_rows.csv, cstar_rows.csv) schreiben")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"{root}: kein Verzeichnis gefunden.")
        return

    rows = {"t2": [], "t3": [], "cstar": []}
    bad = []

    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not _is_json(fn):
                continue
            fp = os.path.join(dirpath, fn)
            if not _within_since(fp, args.since):
                continue
            js = _load(fp)
            if js.get("__bad__"):
                bad.append(js)
                continue
            task = _task_from_path(fp)
            if task == "t2":
                rows["t2"].append(_row_t2(js, fp))
            elif task == "t3":
                rows["t3"].append(_row_t3(js, fp))
            elif task == "cstar":
                rows["cstar"].append(_row_cstar(js, fp))
            else:
                # Unklassifizierbar -> ignorieren still
                pass

    # sortieren nach mtime
    for k in rows:
        rows[k].sort(key=lambda r: r.get("mtime", ""))

    # Report bauen
    report_lines = []
    hdr = f"AGGREGATE REPORT  ({dt.datetime.now().isoformat(timespec='seconds')})"
    if args.tag:
        hdr += f"  —  tag={args.tag}"
    hdr += f"\nroot = {os.path.abspath(root)}\n"
    report_lines.append(hdr)

    # T2 Stats
    t2 = rows["t2"]
    if t2:
        pvals = [r.get("p_value_final") for r in t2 if isinstance(r.get("p_value_final"), (int, float))]
        report_lines.append(f"T2  files={len(t2)}    p_value_final: {_fmt_stats(pvals)}")
        # Seed-Übersicht
        seeds = sorted(set([r.get("seed") for r in t2 if r.get("seed") is not None]))
        report_lines.append(f"     seeds: {seeds}")
        # Letzte 3 Zeilen als quick peek
        report_lines.append("     last 3:")
        for r in t2[-3:]:
            report_lines.append(
                f"       {r.get('mtime')} seed={r.get('seed')} stat={r.get('stat'):.6g} p_final={r.get('p_value_final')}"
            )
    else:
        report_lines.append("T2: keine Dateien gefunden.")

    # T3 Stats
    t3 = rows["t3"]
    if t3:
        areas = [r.get("A_loop") for r in t3 if isinstance(r.get("A_loop"), (int, float))]
        report_lines.append(f"T3  files={len(t3)}    A_loop: {_fmt_stats(areas)}")
    else:
        report_lines.append("T3: keine Dateien gefunden.")

    # C* Stats
    cs = rows["cstar"]
    if cs:
        pvals_c = [r.get("p_value") for r in cs if isinstance(r.get("p_value"), (int, float))]
        report_lines.append(f"C*  files={len(cs)}    p_value: {_fmt_stats(pvals_c)}")
    else:
        report_lines.append("C*: keine Dateien gefunden.")

    if bad:
        report_lines.append(f"\nWARN: {len(bad)} JSON-Dateien konnten nicht gelesen werden.")

    report_txt = "\n".join(report_lines)
    print(report_txt)

    # quick_report.txt im Repo-Root ablegen
    with open("quick_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")

    # CSVs
    if args.csv:
        _write_csv(rows["t2"], "t2_rows.csv")
        _write_csv(rows["t3"], "t3_rows.csv")
        _write_csv(rows["cstar"], "cstar_rows.csv")

if __name__ == "__main__":
    main()
