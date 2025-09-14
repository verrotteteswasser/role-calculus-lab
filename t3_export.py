# t3_export.py
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_rows(root: str):
    rows = []
    for p in sorted(glob(os.path.join(root, "t3", "*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        res = j.get("result", {})
        prm = j.get("params", {})
        rows.append({"path": p, "u_grid": res.get("u_grid", []),
                     "forward": res.get("forward", []),
                     "backward": res.get("backward", []),
                     "A_loop": res.get("A_loop"),
                     "seed": prm.get("seed")})
    return rows

def plot_loop(u, fwd, bwd, out_png):
    fig = plt.figure()
    plt.plot(u, fwd, label="forward")
    plt.plot(u, bwd, label="backward")
    plt.xlabel("u"); plt.ylabel("band-mean coherence")
    plt.title("T3 Hysteresis loop")
    plt.legend()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. result\\v2025-09-15_powerT3")
    ap.add_argument("--out-fig", default="figure/fig_T3_loop.png")
    ap.add_argument("--out-tex", default="figure/T3_figure_snippet.tex")
    args = ap.parse_args()

    rows = load_rows(args.root)
    if not rows:
        print("No T3 JSONs found.")
        return

    # Nimm den ersten als Figure-Beispiel (oder nimm Median A_loop)
    rows_sorted = sorted([r for r in rows if r["A_loop"] is not None], key=lambda r: r["A_loop"])
    r = rows_sorted[len(rows_sorted)//2] if rows_sorted else rows[0]

    u = np.array(r["u_grid"], dtype=float)
    fwd = np.array(r["forward"], dtype=float)
    bwd = np.array(r["backward"], dtype=float)

    # ensure figure dir
    Path(os.path.dirname(args.out_fig) or ".").mkdir(parents=True, exist_ok=True)
    plot_loop(u, fwd, bwd, args.out_fig)

    A = float(r["A_loop"]) if r["A_loop"] is not None else float(np.trapz(np.abs(fwd-bwd), u))
    tex = f"""
% Auto-generated: T3 hysteresis loop
\\begin{{figure}}[ht]
  \\centering
  \\includegraphics[width=0.72\\linewidth]{{{args.out_fig}}}
  \\caption{{Hysteresis in band-mean coherence during parameter sweep: forward vs backward. Loop area $A_{{\\mathrm{{loop}}}} = {A:.3g}$.}}
  \\label{{fig:T3-loop}}
\\end{{figure}}
""".strip()

    Path(os.path.dirname(args.out_tex) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text(tex, encoding="utf-8")

if __name__ == "__main__":
    main()
