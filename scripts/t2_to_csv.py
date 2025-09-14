import json, os, csv, argparse
from glob import glob
from pathlib import Path

def rows(folder):
    out=[]
    for p in sorted(glob(os.path.join(folder, "t2", "*.json"))):
        j=json.loads(Path(p).read_text(encoding="utf-8"))
        prm, res = j.get("params", {}), j.get("result", {})
        out.append({
            "file": os.path.basename(p),
            "seed": prm.get("seed"),
            "stat": res.get("stat"),
            "p_phase": res.get("p_value_phase"),
            "p_flip": res.get("p_value_flip"),
            "p_final": res.get("p_value_final"),
            "mode": res.get("mode"),
            "null_mode": res.get("null_mode"),
            "band_min": prm.get("band_min"),
            "band_max": prm.get("band_max"),
            "nperseg": prm.get("nperseg"),
            "fs_ds": prm.get("fs_ds"),
        })
    return out

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = ["file","seed","stat","p_phase","p_flip","p_final","mode","null_mode","band_min","band_max","nperseg","fs_ds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--both", required=True)   # e.g. result\v2025-09-15_woop\both
    ap.add_argument("--phase", required=True)  # e.g. result\v2025-09-15_woop\phase
    args=ap.parse_args()
    write_csv(os.path.join(args.both,  "summary","t2_rows.csv"), rows(args.both))
    write_csv(os.path.join(args.phase, "summary","t2_rows.csv"), rows(args.phase))
