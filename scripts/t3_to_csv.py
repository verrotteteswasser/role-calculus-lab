import json, os, csv, argparse
from glob import glob
from pathlib import Path

def rows(root):
    out=[]
    for p in sorted(glob(os.path.join(root, "t3", "*.json"))):
        j=json.loads(Path(p).read_text(encoding="utf-8"))
        prm, res = j.get("params", {}), j.get("result", {})
        out.append({
            "file": os.path.basename(p),
            "seed": prm.get("seed"),
            "A_loop": res.get("A_loop"),
            "n_steps": prm.get("n_steps") or res.get("u_grid") and len(res["u_grid"]),
            "sweep": res.get("sweep"),
            "mode": res.get("mode"),
        })
    return out

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = ["file","seed","A_loop","n_steps","sweep","mode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)   # e.g. result\v2025-09-15_woop
    args=ap.parse_args()
    write_csv(os.path.join(args.root, "summary","t3_rows.csv"), rows(args.root))
