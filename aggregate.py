#!/usr/bin/env python3
"""
Aggregate OGC experiment outputs.

Usage:
  python aggregate.py                 # scan ./result
  python aggregate.py --root result   # explicit root
Creates:
  result/summary/t2_rows.csv
  result/summary/t3_rows.csv
  result/summary/cstar_rows.csv
  result/summary/quick_report.txt
"""

from __future__ import annotations
import argparse, csv, json, os, sys, math
from datetime import datetime
from typing import Dict, Any, List, Tuple

def _flatten(prefix: str, obj: Any, out: Dict[str, Any]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}.", v, out)
    elif isinstance(obj, list):
        out[prefix[:-1]] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix[:-1]] = obj

def load_rows(root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    t2_rows: List[Dict[str, Any]] = []
    t3_rows: List[Dict[str, Any]] = []
    cstar_rows: List[Dict[str, Any]] = []

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fn)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[skip] cannot parse {fpath}: {e}", file=sys.stderr)
                continue

            row: Dict[str, Any] = {"file": os.path.relpath(fpath, root)}
            try:
                st = os.stat(fpath)
                row["mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
            except:
                pass

            if isinstance(data, dict):
                if "params" in data:
                    _flatten("params.", data["params"], row)
                if "result" in data:
                    _flatten("result.", data["result"], row)

            dlow = dirpath.lower()
            if "t2" in dlow or "p_value_flip" in json.dumps(data, ensure_ascii=False):
                t2_rows.append(row)
            elif "t3" in dlow or "Theta_up" in json.dumps(data, ensure_ascii=False):
                t3_rows.append(row)
            elif "cstar" in dlow or "tail_mean_acf" in json.dumps(data, ensure_ascii=False):
                cstar_rows.append(row)
            else:
                if any(k.startswith("result.p_value") for k in row):
                    t2_rows.append(row)
                elif "result.A_loop" in row:
                    t3_rows.append(row)
                else:
                    cstar_rows.append(row)

    return t2_rows, t3_rows, cstar_rows

def _write_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _nanmean(xs: List[float]) -> float:
    xs2 = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs2)/len(xs2) if xs2 else float("nan")

def summarize_t2(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "T2: no files found."
    pcol = "result.p_value_final"
    if pcol not in rows[0]:
        if "result.p_value" in rows[0]: pcol = "result.p_value"
        elif "result.p_value_flip" in rows[0]: pcol = "result.p_value_flip"
    ps = [float(r.get(pcol, "nan")) for r in rows]
    sig = [(str(r.get("result.decision_alpha_0.05","")).lower()=="true") or (float(r.get(pcol,1.0))<0.05) for r in rows]
    stat_mean = _nanmean([float(r.get("result.stat","nan")) for r in rows])
    return (
        f"T2 files: {len(rows)}\n"
        f"  mean stat: {stat_mean:.6f}\n"
        f"  mean {pcol}: {_nanmean(ps):.6f}\n"
        f"  min p: {min(ps):.6f}  max p: {max(ps):.6f}\n"
        f"  significant @0.05: {sum(1 for s in sig if s)}/{len(sig)} ({100*sum(1 for s in sig if s)/len(sig):.1f}%)\n"
    )

def summarize_t3(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "T3: no files found."
    A = [float(r.get("result.A_loop","nan")) for r in rows]
    return (
        f"T3 files: {len(rows)}\n"
        f"  mean A_loop: {_nanmean(A):.6f}\n"
        f"  min A_loop: {min(A):.6f}  max A_loop: {max(A):.6f}\n"
        f"  Theta_up range: {min(float(r.get('result.Theta_up','nan')) for r in rows):.6f} .. "
        f"{max(float(r.get('result.Theta_up','nan')) for r in rows):.6f}\n"
    )

def summarize_cstar(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "C*: no files found."
    p = [float(r.get("result.p_value","nan")) for r in rows]
    stat = [float(r.get("result.stat","nan")) for r in rows]
    tail = [float(r.get("result.tail_mean_acf","nan")) for r in rows]
    return (
        f"C* files: {len(rows)}\n"
        f"  mean stat: {_nanmean(stat):.6f}\n"
        f"  mean p_value: {_nanmean(p):.6f}\n"
        f"  mean tail_mean_acf: {_nanmean(tail):.6f}\n"
        f"  significant @0.05: {sum(1 for x in p if x<0.05)}/{len(p)} "
        f"({100*sum(1 for x in p if x<0.05)/len(p):.1f}%)\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result", help="Root directory to scan")
    args = ap.parse_args()

    root = args.root
    t2_rows, t3_rows, cstar_rows = load_rows(root)

    summary_dir = os.path.join(root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    _write_csv(t2_rows,   os.path.join(summary_dir, "t2_rows.csv"))
    _write_csv(t3_rows,   os.path.join(summary_dir, "t3_rows.csv"))
    _write_csv(cstar_rows,os.path.join(summary_dir, "cstar_rows.csv"))

    report = "\n".join([
        f"Scan root: {os.path.abspath(root)}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        summarize_t2(t2_rows),
        summarize_t3(t3_rows),
        summarize_cstar(cstar_rows),
    ])

    rpt_path = os.path.join(summary_dir, "quick_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"[saved] {rpt_path}")

if __name__ == "__main__":
    main()
