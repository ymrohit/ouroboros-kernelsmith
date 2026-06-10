# OUROBOROS — a model that writes the GPU kernels it runs on

**Claim, in one sentence:** a Qwen3.6-27B fine-tuned only by an *immutable verifier*
(compile → `allclose` vs PyTorch across adversarial shapes/dtypes/magnitudes → CUDA-event
benchmark) writes Triton fusion kernels that **reproducibly beat `torch.compile`
max-autotune** — 32/32 ops on H200 in the recorded run — and match-or-beat expert
hand-written Triton on the comparable ops.

**The harness is the product.** The model is replaceable; the thing that cannot be faked is
the referee. Correctness is a boolean, speed is a number, and neither the proposer nor any
trainer can touch how either is measured.

## Verified results (recorded runs; every number is harness-emitted)

| evidence | result | report |
|---|---|---|
| 5× stability re-bench vs max-autotune (H200) | **32/32 ops beat it reproducibly** (mean−spread > 1.0); 16 orig + 16 new, 30/32 model-authored | `reports/rebench_stability_qwen3.6-27b.md` |
| Head-to-head vs expert Triton (Liger/Unsloth/tutorials) | ours faster on all 5 comparable ops (fixed-schedule condition) | `reports/headtohead_experts_qwen3.6-27b.md` |
| RL vs continue-SFT on 16 unseen ops | RL self-distill taught them (16/16); continue-SFT stalled and was stopped | `reports/discovery_newops_qwen3.6-27b.md` |
| Harness selftest (V2, 30 cases) | 14 gold pass · 13 negative controls rejected · **3 anti-gaming cheats rejected** | `ouroboros/reports/harness_selftest.json` |
| E2E composed MLP block (V2, 4090 seeds) | correct + 1.085× eager / 1.301× compile-MA at block level (Amdahl bound ~1.114× stated) | `ouroboros/reports/e2e_block.json` |

**Honest bounds, up front.** These are reproducible *scheduling* wins (~10–85%) over the
incumbent autotuner on bandwidth-bound fusion ops at the benchmarked shapes — NOT wins over
cuBLAS/FlashAttention-class kernels, and NOT new algorithm classes. The V2 shape-grid
re-bench measures (instead of argues) which wins survive across (M,N)×dtype; its per-cell
losses are part of the result. The forward-only cross-entropy comparison is unfair to Liger
(their CE computes grads in the forward) and is flagged wherever it appears.

## What V2 hardened (the anti-gaming work)

The v1 harness had three theoretical exploits — a kernel could special-case the public
fixed bench shape, memoize its output by input pointer, or mutate its inputs in place. All
three are now (a) impossible by construction (bench shape joins the correctness sweep;
anti-memoization poke + verify-after-bench; mutation check) and (b) *proven* impossible by
three new negative-control kernels that implement the exploits and must be REJECTED for the
selftest to pass. A `rotate` bench mode breaks L2 residency as a cache-cold cross-check.

## The op suite — 101 verified specs

16 explicit ops (norms, softmax family, GLU family, RoPE, dequant) + a generative fusion
grammar `[+residual] → {rms,layer}norm → epilogue` over **19 epilogues** (76 chain ops) + 5
V2 standalone ops (`softcap_softmax`, `rmsnorm_gemma`, `glu`, `rope_interleaved`,
`cross_entropy`). Every op enters the loop only after its gold seed passes the harness AND
its negative control is rejected. (`l1norm` was evaluated and *excluded*: its outputs sit
below fp16 atol, so allclose would be vacuous there.)

## Reproduce

```bash
.venv-modal/bin/modal token new                      # auth (HF secret optional, for pushes)
.venv-modal/bin/modal run modal_app.py::selftest     # the gate: 30 cases must be ALL GREEN
.venv-modal/bin/modal run modal_app.py::verify_chains          # 64-op grammar gate (L4)
.venv-modal/bin/modal run --detach modal_app.py::train_all     # corpus → SFT → RL → rebench
.venv-modal/bin/modal run --detach modal_app.py::rebench_shapes  # V2 shape-grid (geomean + losses)
.venv-modal/bin/modal run --detach modal_app.py::bench_experts   # V2 two-condition head-to-head
.venv-modal/bin/modal run --detach modal_app.py::e2e             # composed-block demo
```

Local (any CUDA GPU with the train venv): `python ouroboros/harness.py` runs the full
selftest; `python ouroboros/rebench_shapes.py --kernels ouroboros/seed_kernels` grids the
gold seeds.

## Repo map

```
modal_app.py               # all Modal entrypoints (selftest/sft/rl/rebench/rebench_shapes/
                           #   bench_experts/e2e/verify_chains/kernelbench_eval/train_all)
ouroboros/
  harness.py               # THE PRODUCT: immutable compile→allclose→bench referee (V2-hardened)
  specs.py                 # 101 OpSpecs + shape-grid input builders (additive)
  chains.py                # the fusion grammar (19 epilogues × 2 norms × ±residual)
  seed_kernels/            # gold kernels + negative controls (incl. 3 anti-gaming cheats)
  teacher_kernels.py       # structurally-diverse teacher corpus, harness-filtered
  sft_train.py             # PHASE 1: 0% → ~100% valid Triton, gated on measured valid-rate
  rl_kernelsmith.py        # PHASE 2: GRPO + self-distill, reward = measured speedup
  rebench_stability.py     # 5× stability gate   rebench_shapes.py  # V2 shape grid
  e2e_block_bench.py       # V2 composed-block demo   external_harvest.py  # expert kernels (2 conditions)
  CLAUDE.md                # the doctrine — read before touching anything
reports/                   # the recorded H200 run    volume_backup/  # mirror of the Modal volume
```

## Integrity notes (what was refused, and why it's in the repo)

- Stopped a 4.5h continue-SFT when it interfered with proven ops; reported instead of buried.
- KernelBench: `kernelbench_eval` exists for the *base-model* condition only. Our specialist
  scores ~0 there by construction (foreign format, generalist ops) and we declined the
  prompt-prefill bridge as a headline number — the code stays so the honest low number can
  be reproduced, not so a shortcut can be laundered.
- The v1 "5/11 expert kernels non-robust" claim is **retired in V2**: through Liger's own
  public API all 5 recipes pass our adversarial sweep — the failures were artifacts of our
  fixed-schedule extraction wrappers, and the two conditions are now reported side by side.
- The RL "KL" guard is honestly a |Δ seq-logprob| drift penalty, and is labeled as such.

Artifacts: 🤗 `YMRohit/ouroboros-kernelsmith-qwen3.6-27b` (adapters, best kernels, reports)
· `YMRohit/ouroboros-kernel-corpus` (verified corpus). License: MIT.
