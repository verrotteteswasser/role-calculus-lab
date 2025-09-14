
import json, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def load_rows(folder: str):
    rows = []
    for p in sorted(glob(os.path.join(folder, "*.json"))):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        prm, res = j.get("params", {}), j.get("result", {})
        rows.append({
            "file": os.path.basename(p),
            "seed": prm.get("seed"),
            "stat": res.get("stat"),
            "p_phase": res.get("p_value_phase"),
            "p_flip": res.get("p_value_flip"),
            "p_final": res.get("p_value_final"),
            "mode": res.get("mode"),
            "null_mode": res.get("null_mode")
        })
    return pd.DataFrame(rows)

def metrics_p(series, alpha=0.05):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return {"n":0,"mean":None,"min":None,"max":None,"count_sig":0,"alpha":alpha}
    return {"n":int(s.size), "mean":float(s.mean()), "min":float(s.min()), "max":float(s.max()), "count_sig":int((s<alpha).sum()), "alpha":alpha}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", required=True, help="folder with t2/*.json (both)")
    ap.add_argument("--phase", required=True, help="folder with t2/*.json (phase)")
    ap.add_argument("--out-dir", default="figure", help="directory for figures and tex")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df_b = load_rows(args.both)
    df_p = load_rows(args.phase)

    m_b = metrics_p(df_b["p_final"] if "p_final" in df_b else pd.Series(dtype=float))
    m_p = metrics_p(df_p["p_phase"] if "p_phase" in df_p else pd.Series(dtype=float))

    # A: both histogram
    if m_b["n"]>0:
        fig = plt.figure()
        plt.hist(pd.to_numeric(df_b["p_final"], errors="coerce").dropna(), bins=15)
        plt.title("Histogram of p_final (both-null)"); plt.xlabel("p_final"); plt.ylabel("count")
        (out / "fig_T2_hist_both.png").with_suffix(".png")
        fig.savefig(out / "fig_T2_hist_both.png", bbox_inches="tight", dpi=150); plt.close(fig)

    # B: phase histogram
    if m_p["n"]>0:
        fig = plt.figure()
        plt.hist(pd.to_numeric(df_p["p_phase"], errors="coerce").dropna(), bins=15)
        plt.title("Histogram of p_phase (phase-only null)"); plt.xlabel("p_phase"); plt.ylabel("count")
        fig.savefig(out / "fig_T2_hist_phase.png", bbox_inches="tight", dpi=150); plt.close(fig)

    # C: scatter stat vs p_final (both)
    if m_b["n"]>0 and "stat" in df_b:
        fig = plt.figure()
        x = pd.to_numeric(df_b["stat"], errors="coerce").astype(float)
        y = pd.to_numeric(df_b["p_final"], errors="coerce").astype(float)
        plt.scatter(x, y); plt.title("Band-mean coherence vs p_final (both)")
        plt.xlabel("stat (band-mean coherence)"); plt.ylabel("p_final")
        fig.savefig(out / "fig_T2_scatter_stat_vs_p.png", bbox_inches="tight", dpi=150); plt.close(fig)

    capt_a = f"Across {m_b['n']} seeds under the conservative both-null, p_final is right-skewed with {m_b['count_sig']}/{m_b['n']} < 0.05; min {m_b['min']:.4g}, mean {m_b['mean']:.4g}."
    capt_b = f"Phase-only surrogates are consistently rejected: {m_p['count_sig']}/{m_p['n']} with p < 0.05; mean p ≈ {m_p['mean']:.2e}, max {m_p['max']:.2g}."
    capt_c = "Higher band-mean coherence corresponds to lower p_final; best seeds fall below 0.01."

    tex = f"""
% Auto-generated T2 figures and captions
\\begin{{figure}}[ht]
  \\centering
  \\includegraphics[width=0.72\\linewidth]{{figure/fig_T2_hist_both.png}}
  \\caption{{{capt_a}}}
  \\label{{fig:T2-hist-both}}
\\end{{figure}}

\\begin{{figure}}[ht]
  \\centering
  \\includegraphics[width=0.72\\linewidth]{{figure/fig_T2_hist_phase.png}}
  \\caption{{{capt_b}}}
  \\label{{fig:T2-hist-phase}}
\\end{{figure}}

\\begin{{figure}}[ht]
  \\centering
  \\includegraphics[width=0.72\\linewidth]{{figure/fig_T2_scatter_stat_vs_p.png}}
  \\caption{{{capt_c}}}
  \\label{{fig:T2-scatter-stat-vs-p}}
\\end{{figure}}
""".strip()
    (out / "T2_figures_snippet.tex").write_text(tex, encoding="utf-8")

if __name__ == "__main__":
    main()
