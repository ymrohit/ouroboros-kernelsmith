"""SHAPE-GRID re-bench (V2): do the wins survive away from the headline shape?

The v1 claim ("beats max-autotune") was measured at ONE fixed shape per op. This script
re-benchmarks every archived best kernel across a GRID of (M, N) x dtype cells — through
the SAME hardened harness (CUDA events, warmup both paths, median-of-N, anti-memoization
poke, verify-after-bench, max-autotune baseline) — and reports:

  - per-op GEOMEAN speedup vs max-autotune across the grid (the robust claim)
  - per-op win-rate (cells > 1.0) and the explicit LOSS REGIONS (reported plainly)
  - an optional rotate-buffer cross-check at the headline shape (L2-residency control)

A win that only exists at 8192x4096 is an overfit schedule, not a discovery. This script
is where that distinction is measured instead of argued.

Usage:
  python rebench_shapes.py                       # all kernels in outputs/best_kernels
  python rebench_shapes.py --kernels seed_kernels --ops rmsnorm,softmax
  python rebench_shapes.py --rotate              # add the cache-cold cross-check column
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import evaluate
from specs import _ROPE_FAMILY, SPECS, _grid_kind

# (M, N) grid for matrix-family ops. Headline 8192x4096 is one cell among equals: tall-thin,
# square, wide, decode-like small-M, and prefill-like big-M are all represented.
MATRIX_GRID = [(1024, 4096), (4096, 4096), (8192, 4096), (4096, 8192), (16384, 2048), (256, 16384)]
# (M, D) grid for rope-family ops (D = head dim, even).
ROPE_GRID = [(8192, 128), (32768, 128), (65536, 64), (16384, 256)]
DTYPES = ["fp16", "bf16"]
HEADLINE = (8192, 4096)


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=str(HERE / "outputs" / "best_kernels"),
                    help="dir of <op>.py kernels to grid-bench")
    ap.add_argument("--ops", default="", help="comma subset (default: every kernel in the dir)")
    ap.add_argument("--dtypes", default="fp16,bf16")
    ap.add_argument("--n-iters", type=int, default=50, dest="n_iters",
                    help="timed iterations per cell (median); autotune compile dominates anyway")
    ap.add_argument("--rotate", action="store_true",
                    help="ALSO re-run the headline cell with rotating buffers (cache-cold check)")
    ap.add_argument("--out", default=str(HERE / "reports" / "rebench_shapes.json"))
    args = ap.parse_args()

    kdir = Path(args.kernels)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()] or sorted(
        p.stem for p in kdir.glob("*.py") if not p.stem.endswith(("_wrong", "_cheat_shape", "_cheat_memo", "_mutate")))
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    print(f"SHAPE-GRID rebench | {len(ops)} ops | dtypes={dtypes} | kernels={kdir}\n" + "=" * 90)

    results = {}
    t0 = time.time()
    for op in ops:
        if op not in SPECS:
            print(f"  -- {op}: not a registered spec, skipping"); continue
        try:
            kind = _grid_kind(op)
        except KeyError:
            print(f"  -- {op}: no grid builder, skipping"); continue
        src = (kdir / f"{op}.py").read_text()
        grid = ROPE_GRID if op in _ROPE_FAMILY else MATRIX_GRID
        cells = []
        for (M, N) in grid:
            for dt in dtypes:
                r = evaluate(src, op, strong=True, n_iters=args.n_iters,
                             bench_override=(M, N, dt))
                cell = {"M": M, "N": N, "dtype": dt, "status": r.status,
                        "vs_maxauto": r.speedup_maxauto, "vs_compile": r.speedup_compile,
                        "latency_ms": r.latency_ms}
                cells.append(cell)
                mark = ("WIN " if r.speedup_maxauto > 1.0 else "loss") if r.status == "ok" else "ERR "
                print(f"  {op:24} {M:>6}x{N:<6} {dt}  -> {r.status:12} "
                      f"{r.speedup_maxauto:.3f}x MA  {r.speedup_compile:.3f}x compile  [{mark}]",
                      flush=True)
        ok_cells = [c for c in cells if c["status"] == "ok" and c["vs_maxauto"] > 0]
        gm = geomean([c["vs_maxauto"] for c in ok_cells])
        wins = sum(1 for c in ok_cells if c["vs_maxauto"] > 1.0)
        losses = [c for c in ok_cells if c["vs_maxauto"] <= 1.0]
        rec = {"cells": cells, "geomean_vs_maxauto": round(gm, 4),
               "win_rate": round(wins / len(ok_cells), 3) if ok_cells else 0.0,
               "n_cells_ok": len(ok_cells), "n_cells": len(cells),
               "losses": [{k: c[k] for k in ("M", "N", "dtype", "vs_maxauto")} for c in losses]}
        if args.rotate:
            HM, HN = HEADLINE if op not in _ROPE_FAMILY else ROPE_GRID[1]
            rh = evaluate(src, op, strong=True, n_iters=args.n_iters,
                          bench_override=(HM, HN, "fp16"), rotate=True)
            rec["headline_rotated_vs_maxauto"] = rh.speedup_maxauto
            print(f"  {op:24} headline ROTATED (cache-cold) -> {rh.speedup_maxauto:.3f}x MA", flush=True)
        results[op] = rec
        print(f"  {op:24} GEOMEAN {gm:.3f}x MA | win-rate {rec['win_rate']:.0%} "
              f"({wins}/{len(ok_cells)}) | losses: "
              + (", ".join(f"{c['M']}x{c['N']}/{c['dtype']}={c['vs_maxauto']:.2f}" for c in losses) or "none"),
              flush=True)

    overall = geomean([r["geomean_vs_maxauto"] for r in results.values() if r["geomean_vs_maxauto"] > 0])
    summary = {"overall_geomean_vs_maxauto": round(overall, 4),
               "ops_geomean_above_1": sum(1 for r in results.values() if r["geomean_vs_maxauto"] > 1.0),
               "n_ops": len(results), "grid": {"matrix": MATRIX_GRID, "rope": ROPE_GRID},
               "dtypes": dtypes, "n_iters": args.n_iters, "elapsed_min": round((time.time() - t0) / 60, 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_op": results}, indent=2))
    print("=" * 90)
    print(f"OVERALL geomean vs max-autotune: {overall:.3f}x | "
          f"{summary['ops_geomean_above_1']}/{summary['n_ops']} ops geomean > 1.0 | -> {out}")
    print("Honest read: an op whose geomean > 1.0 wins ACROSS the regime, not at one point;\n"
          "losses are listed per-cell above and in the report — they are part of the result.")


if __name__ == "__main__":
    main()
