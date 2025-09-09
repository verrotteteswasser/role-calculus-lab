# aggregate.py
import json, os, glob, math
from pathlib import Path
from datetime import datetime

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- T2: Cross-Coherence ----------
def _extract_t2_row(js, path):
    P = js.get("params", {}) or {}
    R = js.get("result", {}) or {}

    # null_mode kann in result oder params stecken
    null_mode = (R.get("null_mode") or P.get("null_mode") or "flip")

    # p-Wert korrekt wählen
    if "p_value_final" in R:
        p = R["p_value_final"]
    elif null_mode == "both":
        # Fallback: streng nach Definition = max(flip, phase)
        p = max(
            R.get("p_value_flip", float("nan")),
            R.get("p_value_phase", float("nan")),
        )
    else:
        # Einzel-Null: nimm vorhandenes Feld
        p = (
            R.get("p_value",
            R.get("p_value_flip",
            R.get("p_value_phase", float("nan"))))
        )

    # Band-Grenzen robust ziehen (neuere CLI hat band_min/max; sonst 'band')
    band_min = P.get("band_min")
    band_max = P.get("band_max")
    if (band_min is None or band_max is None) and isinstance(P.get("band"), (list, tuple)) and len(P["band"]) == 2:
        band_min, band_max = P["band"]

    return {
        "file": str(path),
        "seed": P.get("seed"),
        "n": P.get("n"),
        "n_null": P.get("n_null"),
        "null_mode": null_mode,
        "mode": R.get("mode") or P.get("mode"),
        "stat": R.get("stat"),
        "band_fraction": R.get("band_fraction"),
        "p_value": p,                          # <- zentrale Spalte für Auswertung
        "p_value_flip": R.get("p_value_flip"),
        "p_value_phase": R.get("p_value_phase"),
        "p_value_final": R.get("p_value_final"),
        "decision_alpha_0.05": R.get("decision_alpha_0.05"),
        "fs_ds": P.get("fs_ds"),
        "target_fs": P.get("target_fs"),
        "nperseg": P.get("nperseg"),
        "band_min": band_min,
        "band_max": band_max,
    }

# ---------- T3: Hysteresis ----------
def _extract_t3_row(js, path):
    P = js.get("params", {}) or {}
    R = js.get("result", {}) or {}

    return {
        "file": str(path),
        "seed": P.get("seed"),
        "n": P.get("n"),
        "u_min": P.get("u_min"),
        "u_max": P.get("u_max"),
        "noise": P.get("noise"),
        "Theta_up": R.get("Theta_up"),
        "Theta_down": R.get("Theta_down"),
        "A_loop": R.get("A_loop"),
    }

# ---------- C*: Long-return ----------
def _extract_cstar_row(js, path):
    P = js.get("params", {}) or {}
    R = js.get("result", {}) or {}

    return {
        "file": str(path),
        "seed": P.get("seed"),
        "n": P.get("n"),
        "max_lag": P.get("max_lag"),
        "inject_echo": P.get("inject_echo"),
        "echo_every": P.get("echo_every"),
        "stat": R.get("stat"),
        "tail_mean_acf": R.get("tail_mean_acf"),
        "p_value": R.get("p_value"),
    }

def _detect_kind(js, path):
    """Erkenne JSON-Typ robust über Felder und (optional) Pfad."""
    R = (js.get("result") or {})
    # Feld-basierte Heuristik
    if "A_loop" in R or ("Theta_up" in R and "Theta_down" in R):
        return "t3"
    if "tail_mean_acf" in R:
        return "cstar"
    if ("band_fraction" in R and "stat" in R) or ("p_value_final" in R or "p_value_flip" in R or "p_value_phase" in R):
        return "t2"
    # Pfad-Fallback
    p = str(path).lower()
    if "/t2/" in p or "\\t2\\" in p: return "t2"
    if "/t3/" in p or "\\t3\\" in p: return "t3"
    if "/cstar/" in p or "\\cstar\\" in p: return "cstar"
    return "unknown"

def _fmt_float(x, digits=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{digits}g}"

def main(root="result"):
    root = Path(root)
    t2_rows, t3_rows, cstar_rows = [], [], []

    # Alle JSONs einsammeln
    for path in root.rglob("*.json"):
        try:
            js = _load_json(path)
        except Exception:
            continue
        kind = _detect_kind(js, path)
        try:
            if kind == "t2":
                t2_rows.append(_extract_t2_row(js, path))
            elif kind == "t3":
                t3_rows.append(_extract_t3_row(js, path))
            elif kind == "cstar":
                cstar_rows.append(_extract_cstar_row(js, path))
        except Exception:
            # skip kaputte Einträge, aber nicht crashen
            continue

    # Ausgabeordner
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # CSVs schreiben
    def _write_csv(rows, out_path, header=None):
        if not rows:
            return
        if header is None:
            # Header aus Union aller Keys
            keys = []
            seen = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k); keys.append(k)
        else:
            keys = header
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                vals = []
                for k in keys:
                    v = r.get(k)
                    if isinstance(v, float):
                        vals.append(_fmt_float(v))
                    else:
                        vals.append("" if v is None else str(v))
                f.write(",".join(vals) + "\n")

    if t2_rows:
        _write_csv(t2_rows, summary_dir / "t2_rows.csv")
    if t3_rows:
        _write_csv(t3_rows, summary_dir / "t3_rows.csv")
    if cstar_rows:
        _write_csv(cstar_rows, summary_dir / "cstar_rows.csv")

    # Quick-Report
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Quick report ({now})\n")

    if t2_rows:
        ps = [r["p_value"] for r in t2_rows if isinstance(r.get("p_value"), (int, float)) and not math.isnan(r["p_value"])]
        alpha_hits = sum(1 for r in t2_rows if r.get("decision_alpha_0.05") is True)
        lines += [
            "== T2 Cross-Coherence ==",
            f"runs: {len(t2_rows)}",
            f"mean p: {_fmt_float(sum(ps)/len(ps)) if ps else 'nan'}",
            f"min p: {_fmt_float(min(ps)) if ps else 'nan'}",
            f"max p: {_fmt_float(max(ps)) if ps else 'nan'}",
            f"alpha=0.05 rejects: {alpha_hits}/{len(t2_rows)}",
            "",
        ]

    if t3_rows:
        As = [r["A_loop"] for r in t3_rows if isinstance(r.get("A_loop"), (int, float))]
        lines += [
            "== T3 Hysteresis ==",
            f"runs: {len(t3_rows)}",
            f"mean A_loop: {_fmt_float(sum(As)/len(As)) if As else 'nan'}",
            f"min A_loop: {_fmt_float(min(As)) if As else 'nan'}",
            f"max A_loop: {_fmt_float(max(As)) if As else 'nan'}",
            "",
        ]

    if cstar_rows:
        ps = [r["p_value"] for r in cstar_rows if isinstance(r.get("p_value"), (int, float))]
        lines += [
            "== C* Long-return ==",
            f"runs: {len(cstar_rows)}",
            f"mean p: {_fmt_float(sum(ps)/len(ps)) if ps else 'nan'}",
            f"min p: {_fmt_float(min(ps)) if ps else 'nan'}",
            f"max p: {_fmt_float(max(ps)) if ps else 'nan'}",
            "",
        ]

    with open(summary_dir / "quick_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ok] wrote:\n  {summary_dir/'t2_rows.csv' if t2_rows else '-'}\n  {summary_dir/'t3_rows.csv' if t3_rows else '-'}\n  {summary_dir/'cstar_rows.csv' if cstar_rows else '-'}\n  {summary_dir/'quick_report.txt'}")

if __name__ == "__main__":
    # optional: root via ENV oder Default "result"
    main(root=os.environ.get("OGC_RESULT_ROOT", "result"))

