\# Role Calculus – Repro Pack (T2 Power \& Nulls)



\## Requirements

\- Python 3.10+ (getestet mit 3.13)

\- numpy, scipy



```bash

pip install -U numpy scipy


# Role-Calculus Lab – T2 Coherence Tests

## Quick start
```powershell
# Beispiel: 50 seeds, konservativer "both"-Test, 5k Surrogates
$OUT = "result\v2025-09-10_powerT2_both"
foreach ($s in 0..49) {
  python -m ogc.cli --out-dir $OUT t2 `
    --n 24576 --n-null 5000 --seed $s `
    --null-mode both `
    --band-min 0.78 --band-max 0.82
}

# Phase-only Vergleich
$OUTP = "result\v2025-09-10_powerT2_phase"
foreach ($s in 0..49) {
  python -m ogc.cli --out-dir $OUTP t2 `
    --n 24576 --n-null 5000 --seed $s `
    --null-mode phase `
    --band-min 0.78 --band-max 0.82
}

