
import argparse, subprocess, sys, os, shlex, pathlib

def run(cmd):
    print(">>", cmd)
    return subprocess.run(cmd, shell=True, check=True)

def parse_seeds(spec: str):
    if "-" in spec:
        a,b = spec.split("-",1)
        return list(range(int(a), int(b)+1))
    return [int(x) for x in spec.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser(description="WOOP: one-loop pipeline for T2+T3")
    p.add_argument("--out-root", required=True, help="result\\vYYYY-MM-DD_name")
    p.add_argument("--seeds", default="0-49")
    p.add_argument("--n", type=int, default=24576)
    p.add_argument("--n-null", type=int, default=5000, dest="n_null")
    p.add_argument("--band-min", type=float, default=0.78)
    p.add_argument("--band-max", type=float, default=0.82)
    p.add_argument("--target-fs", type=float, default=20.0)
    p.add_argument("--nperseg", type=int, default=0)
    p.add_argument("--mode", type=str, default="mean", choices=["mean","peak"])
    p.add_argument("--t3-n", type=int, default=300)
    p.add_argument("--t3-noise", type=float, default=0.05)
    p.add_argument("--t3-seeds", default="0-9")
    args = p.parse_args()

    out = pathlib.Path(args.out_root)
    # 1) T2 phase
    out_phase = out / "phase"
    out_phase.mkdir(parents=True, exist_ok=True)
    for s in parse_seeds(args.seeds):
        cmd = f'python -m ogc.cli --out-dir "{out_phase}" t2 --n {args.n} --n-null {args.n_null} --seed {s} --null-mode phase --band-min {args.band_min} --band-max {args.band_max} --nperseg {args.nperseg} --target-fs {args.target_fs} --mode {args.mode}'
        run(cmd)

    # 2) T2 both
    out_both = out / "both"
    out_both.mkdir(parents=True, exist_ok=True)
    for s in parse_seeds(args.seeds):
        cmd = f'python -m ogc.cli --out-dir "{out_both}" t2 --n {args.n} --n-null {args.n_null} --seed {s} --null-mode both --band-min {args.band_min} --band-max {args.band_max} --nperseg {args.nperseg} --target-fs {args.target_fs} --mode {args.mode}'
        run(cmd)

    # 3) Export T2
    cmd_export_t2 = f'python scripts/t2_export.py --both "{out_both}/t2" --phase "{out_phase}/t2" --out-dir figure'
    run(cmd_export_t2)

    # 4) T3 minimal (using existing CLI flags)
    out_t3 = out / "t3"
    out_t3.mkdir(parents=True, exist_ok=True)
    for s in parse_seeds(args.t3_seeds):
        cmd = f'python -m ogc.cli --out-dir "{out}" t3 --n {args.t3_n} --u-min 0.5 --u-max 1.0 --noise {args.t3_noise} --seed {s}'
        run(cmd)

    # 5) Export T3
    cmd_export_t3 = f'python scripts/t3_export.py --root "{out}" --out-fig figure/fig_T3_loop.png --out-tex figure/T3_figure_snippet.tex'
    run(cmd_export_t3)

    print("\n[WOOP] Done. Figures in ./figure and JSONs in", out)

if __name__ == "__main__":
    main()
