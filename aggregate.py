import argparse, json, os, datetime
from glob import glob
from statistics import mean

def _ts():
    return datetime.datetime.now().isoformat(timespec="seconds")

def _list(dirpath, sub):
    p = os.path.join(dirpath, sub, "*.json")
    return sorted(glob(p))

def _safe_get_pfinal(obj):
    res = obj.get("result", {})
    if "p_value_final" in res and res["p_value_final"] is not None:
        return float(res["p_value_final"])
    # fallback (ältere Läufe)
    if "p_value" in res and res["p_value"] is not None:
        return float(res["p_value"])
    return None

def _load_row_t2(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    prm = j.get("params", {})
    res = j.get("result", {})
    return {
        "file": os.path.basename(path),
        "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "seed": prm.get("seed"),
        "stat": res.get("stat"),
        "p_value_flip": res.get("p_value_flip"),
        "p_value_phase": res.get("p_value_phase"),
        "p_value_final": _safe_get_pfinal(j),
    }

def _load_row_t3(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    prm = j.get("params", {})
    res = j.get("result", {})
    return {
        "file": os.path.basename(path),
        "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "A_loop": res.get("A_loop"),
    }

def _load_row_cstar(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    prm = j.get("params", {})
    res = j.get("result", {})
    return {
        "file": os.path.basename(path),
        "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "stat": res.get("stat"),
        "p_value": res.get("p_value"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="result")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--since", type=str, default=None)  # optional ISO-Filter (nicht zwingend benutzt)
    args = ap.parse_args()

    root = args.root
    print(f"AGGREGATE REPORT  ({_ts()})" + (f"  —  tag={args.tag}" if args.tag else ""))
    print(f"root = {os.path.abspath(root)}\n")

    # --- T2
    t2_files = _list(root, "t2")
    if t2_files:
        rows = [_load_row_t2(p) for p in t2_files]
        pvals = [r["p_value_final"] for r in rows if r["p_value_final"] is not None]
        seeds = [r["seed"] for r in rows if r["seed"] is not None]
        if pvals:
            print(f"T2  files={len(rows)}    p_value_final: n={len(pvals)}, mean={round(mean(pvals), 4)}, min={round(min(pvals),4)}, max={round(max(pvals),4)}")
            if seeds:
                print(f"     seeds: {seeds[:50] if len(seeds)<=50 else seeds[:50] + ['...']}")
            print("     last 3:")
            for r in rows[-3:]:
                print(f"       {r['mtime']} seed={r.get('seed')} stat={round(r.get('stat'),6)} p_final={r.get('p_value_final')}")
        else:
            print("T2: keine p-Werte gefunden (p_value_final/p_value).")
    else:
        print("T2: keine Dateien gefunden.")

    # --- T3
    t3_files = _list(root, "t3")
    if t3_files:
        rows = [_load_row_t3(p) for p in t3_files]
        aloops = [r["A_loop"] for r in rows if r["A_loop"] is not None]
        if aloops:
            print(f"T3  files={len(rows)}    A_loop: n={len(aloops)}, mean={round(mean(aloops), 2)}, min={round(min(aloops),2)}, max={round(max(aloops),2)}")
        else:
            print("T3: keine A_loop gefunden.")
    else:
        print("T3: keine Dateien gefunden.")

    # --- C*
    cs_files = _list(root, "cstar")
    if cs_files:
        rows = [_load_row_cstar(p) for p in cs_files]
        pvals = [r["p_value"] for r in rows if r["p_value"] is not None]
        if pvals:
            print(f"C*  files={len(rows)}    p_value: n={len(pvals)}, mean={round(mean(pvals),3)}, min={round(min(pvals),3)}, max={round(max(pvals),3)}")
        else:
            print("C*: keine p-Werte gefunden.")
    else:
        print("C*: keine Dateien gefunden.")

if __name__ == "__main__":
    main()
