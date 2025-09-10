# === Konfig ===
$Tag    = "v2025-09-08"     # <— anpassen falls du willst
$OutDir = "result"

# ---- T2: seeds 0..19 ----
$N        = 12288
$NNull    = 5000
$BandMin  = 0.7
$BandMax  = 0.9
$NullMode = "both"

foreach ($s in 0..19) {
  python -m ogc.cli --out-dir $OutDir --tag $Tag t2 `
    --n $N --n-null $NNull --seed $s `
    --null-mode $NullMode --band-min $BandMin --band-max $BandMax
}

# ---- T3: seeds 0..19 ----
$N_t3   = 300
$UMin   = 0.0
$UMax   = 2.0
$Noise  = 0.05

foreach ($s in 0..19) {
  python -m ogc.cli --out-dir $OutDir --tag $Tag t3 `
    --n $N_t3 --u-min $UMin --u-max $UMax --noise $Noise --seed $s
}

# ---- optional: C* (ein paar Seeds) ----
$N_c   = 8000
$Lag   = 400
foreach ($s in 0..5) {
  python -m ogc.cli --out-dir $OutDir --tag $Tag cstar `
    --n $N_c --max-lag $Lag --seed $s
}

# ---- Aggregation für genau diesen Tag ----
python aggregate.py --root $OutDir --tag $Tag
