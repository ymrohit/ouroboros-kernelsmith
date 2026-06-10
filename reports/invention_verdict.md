# Invention experiment — verdict (judged honestly, kernels read, not just scored)

**Setup.** `rl_adapter_v2` + 42 RL rounds on 7 problems it had never trained on: `cumsum`
(prefix-scan — a different parallel algorithm class), `entropy` + `kl_div` (double-reduction
logit fusions), and 4 `*_short` ops pinned to 16384×2048 — the ONE regime where every
kernel in the 69-kernel product loses. Explore arm 0.0: the model wrote every candidate.
Raw: `kernelsmith_invent.json`. Exit validation = fresh measurement + max-autotune.

## Scoreboard: 7/7 validated correct, 7/7 beat max-autotune, 5/7 model-authored

| op | vs max-autotune | vs compile | author | context |
|---|---:|---:|---|---|
| softmax_short | **1.803** | 2.086 | **LM** | loss-regime target |
| layernorm_gelu_short | **1.450** | 1.595 | **LM** | was a **0.88× LOSS** in the grid |
| rmsnorm_short | **1.386** | 1.458 | **LM** | loss-regime target |
| add_layernorm_sigmoid_short | **1.222** | 1.332 | **LM** | was the product's WORST cell (**0.69×**) |
| cumsum | **1.292** | 1.896 | **LM** | new algorithm class |
| entropy | 1.477 | 1.808 | seed | model verified kernels but never beat the gold |
| kl_div | 1.225 | 1.295 | seed | same |

LM valid-rate on foreign problems: **49.7%** (167/336) vs ~69% on familiar families — an
honest difficulty gradient, not a collapse. 8 archive lead-takes.

## What the kernels actually show (the judgment)

**1. The model fixed the product's only characterized weakness — with a simpler idea than
mine.** I predicted the 16384×2048 regime needed split-row or multi-row-per-program
schedules. Wrong. The model's winning kernels are clean **whole-row single-block**
schedules (`BLOCK = next_power_of_2(N)`): at N=2048 the row fits one block, 16384 programs
already saturate the SMs, and it was the *looped* template style (tuned at N=4096) causing
the losses — not row-per-program itself. The verifier found the simpler truth and
overruled the human diagnosis. Flipping 0.69× → 1.22× and 0.88× → 1.45× in the exact
cells we published as losses is the strongest possible answer to "do the loss regions
persist?" — no: **point the loop at them and they become wins.**

**2. cumsum is invention-lite, judged precisely.** The model did NOT reinvent carry
propagation — it did something better for this regime: recognized that `tl.cumsum` (a
primitive it knew from pretraining, never shown in any prompt for this op) covers a whole
row in one block, and emitted a loop-free whole-row scan that **beats the human
carry-loop gold by ~47% (1.29× vs 0.88× MA)** and runs 1.90× over compile. Correct across
the full adversarial sweep including N=8192. Honest label: *primitive selection + schedule
simplification, verified* — not a new parallel algorithm.

**3. The hard fusions resisted.** On `entropy` and `kl_div` (coupled double reductions)
the model produced many verified kernels but never beat the hand-written two/three-pass
golds in 6 passes/op. The product still wins (the golds beat max-autotune 1.48×/1.23×) but
authorship stays human. That is the current capability boundary, stated plainly.

## Net judgment

Genuine, verifier-certified adaptation: the model closed the published loss region and
out-engineered the human seed on a foreign algorithm class — but via apt schedule/primitive
*selection*, not novel algorithm *invention*; and the hardest multi-reduction fusions still
belong to the human golds. Product grows to **76 verified kernels** (74 pending the 5×
stability gate on the 7 new). The loss-region flip belongs in the paper as the closer of
the shape-grid story; the wrong-human-diagnosis anecdote belongs in it as evidence the
referee outranks intuition — everyone's.

_2026-06-10 · H200 · V2 harness · exit validation fresh + max-autotune · kernels read before judging._
