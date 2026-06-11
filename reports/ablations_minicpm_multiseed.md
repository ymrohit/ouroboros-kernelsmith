# Multi-seed ablation on MiniCPM5-1B (the single-seed caveat, retired)

**Setup.** OpenBMB **MiniCPM5-1B** (1B, Llama arch), SFT'd to the validity gate (100% valid
on all 6 ops in 2 epochs), then four ablation arms each run at **3 independent seeds**
(genuine `--seed` on python+torch RNG), 6 ops, 18 RL rounds, group 8, exit validation fresh
+ max-autotune. All on a single local **RTX 4090, $0**. 12 runs total.
Raw: `reports/abl_{arm}_s{0,1,2}.json`.

## The 1B writes compiler-beating kernels (the headline)

Across all 12 runs, **every arm beats `torch.compile` max-autotune** (geomean $>1$, beat-rate
83--100%). A **1-billion-parameter** open model, trained for free on a consumer GPU, writes
Triton kernels that beat the strongest compiler baseline. This is the OpenBMB + Tiny Titan
result, independent of the ablation.

## The ablation: arms are statistically TIED (3 seeds)

| arm | removes | geomean vs MA | valid-rate | beat-rate | per-seed geomeans |
|---|---|---:|---:|---:|---|
| control | --- (full loop) | **1.102 ± 0.028** | 0.972 | 100% | 1.141, 1.088, 1.077 |
| no-feedback | referee errors in prompt | 1.090 ± 0.010 | 0.991 | 94% | 1.102, 1.089, 1.078 |
| distill-only | policy-gradient term | 1.057 ± 0.030 | 0.944 | 83% | 1.091, 1.063, 1.017 |
| no-learn | all weight updates | 1.079 ± 0.014 | 0.975 | 94% | 1.076, 1.098, 1.064 |

**Between-arm spread = 0.045; largest within-arm seed std = 0.030.** The spread across arms
is smaller than two seed-standard-deviations, so the arms are statistically indistinguishable.
`no-learn` --- pure best-of-$N$ from the frozen SFT model, *no learning at all* --- lands at
$1.079$, within noise of `control`'s $1.102$ (98%).

## The decisive cross-check: distill-only flipped

This is the cleanest possible demonstration that single-seed ablation orderings are noise:

| | distill-only vs control |
|---|---|
| Modal 27B, **single seed** | distill-only **BEAT** control (1.361 vs 1.338) --- "GRPO earns nothing" |
| MiniCPM 1B, **3 seeds** | distill-only is **WORST** (1.057 vs control 1.102) |

The same arm was best in one single-seed run and worst in a three-seed average. **Neither
ordering is real.** The honest, defensible claim that survives error bars is the weaker but
correct one:

> On familiar operators, no learning ingredient --- feedback, the policy-gradient term, or
> learning at all --- separates beyond seed noise. **Search against the referee is the
> dominant force where competence already exists.** Learning's value lives on *foreign*
> operators (the 37-op discovery and invention runs), not here.

## Why this matters for the paper

The Modal single-seed ablation (paper Table 2) invited the objection "n=1 per arm." This
multi-seed replication retires that caveat and *strengthens* the conclusion: the arms
overlap, so the paper no longer claims a fragile ordering (distill-only > control), only the
robust one (search dominates; learning ingredients are within noise on familiar ops). That
it reproduces on a **1B** model at a different scale, on free hardware, is a bonus the 27B
run could not provide.

_2026-06-11 · RTX 4090 · MiniCPM5-1B (OpenBMB) · 3 seeds × 4 arms · exit validation fresh +
max-autotune · $0._
