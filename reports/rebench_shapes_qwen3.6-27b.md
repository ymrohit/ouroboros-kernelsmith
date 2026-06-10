# Shape-grid re-bench — do the 32 wins survive away from the headline shape?

**Setup.** Every archived best kernel (the 32 from the recorded SFT→RL runs) re-benchmarked
through the V2-hardened harness across a **(M,N) × {fp16,bf16} grid** — matrix ops over
{1024×4096, 4096×4096, 8192×4096, 4096×8192, 16384×2048, 256×16384}, rope ops over
{8192×128, 32768×128, 65536×64, 16384×256} — vs `torch.compile` max-autotune *recompiled
per cell*. Plus a **cache-cold cross-check**: the headline cell re-run with 4 rotating input
clone-sets so nothing stays L2-resident. H200, 50-iter medians, anti-memoization poke +
verify-after-bench active in every cell. 376 cells total, 81.9 min.
Raw: `rebench_shapes_qwen3.6-27b.json` (every cell), volume `reports/rebench_shapes.json`.

## Verdict

- **32/32 ops have geomean > 1.0 vs max-autotune across the grid. Overall geomean: 1.494×.**
- **32/32 survive the cache-cold (rotated-buffer) check** at the headline shape
  (1.10×–1.83×) — the wins are not an L2-residency artifact.
- **38 of 376 cells (10%) are losses, and they are reported per-cell.** The v1 headline
  ("beats max-autotune at 8192×4096") is now the *weakest* version of the claim.

## Per-op geomeans (top/bottom)

| op | geomean ×MA | win-rate | cache-cold |
|---|---:|---:|---:|
| softmax | 1.927 | 100% | 1.833 |
| softmax_scale | 1.732 | 100% | 1.518 |
| rmsnorm_tanh | 1.619 | 100% | 1.457 |
| rmsnorm_relu | 1.568 | 100% | 1.348 |
| … (full table in the JSON) | | | |
| add_layernorm_sigmoid | 1.347 | 83% | 1.227 |
| qknorm_rope | 1.324 | 75% | 1.158 |
| add_rmsnorm_rope | 1.320 | 75% | 1.100 |

Every op not shown sits between 1.32× and 1.62× geomean. 14/32 ops win **all 12 cells**.

## The loss regions (the honest part — and they have a clean structure)

- **34 of the 38 loss cells are at 16384×2048** — the many-rows/short-rows regime. The
  kernels were discovered at N=4096; at N=2048 their row-per-program schedules leave SMs
  underutilized and inductor's split reductions win by 1–39%. This is a *regime boundary*,
  not noise — and it is exactly where the next RL round should add a bench shape.
- The remaining 4: both rope fusions lose at 65536×64 (tiny head-dim, huge M — same
  story), and two `add_*` ops dip ~4–6% at 4096×8192/fp16.
- Worst single cell: `add_layernorm_sigmoid` at 16384×2048/bf16 = 0.687×.

## Read

The claim upgrade this report buys: **"beats the incumbent autotuner at one tuned shape" →
"beats it in geomean across a 12-cell shape/dtype grid, cache-cold verified, with a
characterized 10% loss region concentrated in one regime (short rows)."** The loss region
is itself a result: it tells the next training round precisely which bench shape to add
(N≤2048) and predicts the fix (split-row/persistent schedules — the rmsnorm_wide lesson).

_Generated 2026-06-10 · H200 · V2 harness (bench-shape-in-sweep, poke+verify, mutation
check, rotate) · max-autotune recompiled per cell · losses are part of the result._
