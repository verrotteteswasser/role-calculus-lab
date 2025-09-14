# WOOP — One-Loop Pipeline (T2 + T3)

Usage (from your lab root, with your existing ogc.cli):
```powershell
# Copy ogc/tests/t3_hysteresis.py into your repo under ogc/tests/
# Copy scripts/*.py into your repo root (or adjust paths).
python scripts/woop.py --out-root result\v2025-09-15_woop --seeds 0-49 --t3-seeds 0-9
```

Outputs:
- Figures: figure/fig_T2_hist_both.png, fig_T2_hist_phase.png, fig_T2_scatter_stat_vs_p.png, fig_T3_loop.png
- LaTeX: figure/T2_figures_snippet.tex, figure/T3_figure_snippet.tex
- JSONs under result\v2025-09-15_woop\{phase,both,t3}
