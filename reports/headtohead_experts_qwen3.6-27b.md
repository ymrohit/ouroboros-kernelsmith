# Head-to-head: OUROBOROS kernels vs hand-written expert Triton

**The "are we actually good?" test.** Our kernels vs hand-written Triton from **Liger-Kernel**,
**Unsloth**, and the **Triton tutorials**, all run through the SAME immutable harness on the same
H200, same inputs — correctness (`allclose`, adversarial shapes/dtypes/magnitudes) then CUDA-event
latency, and speedup vs `torch.compile` max-autotune (MA). No knob-tuning, no autotune laundering
(expert kernels' `@triton.autotune` is stripped so we compare fixed schedules).

## Result — on every op where both pass our harness, ours is faster

| op | ours (ms) | ours ×MA | best expert (ms) | expert ×MA | ours faster | our author |
|----|----------:|---------:|------------------|-----------:|------------:|------------|
| softmax   | **0.0535** | 1.846 | tutorial 0.0544 | 1.779 | **+1.7%**  | **MODEL** |
| swiglu    | **0.0660** | 1.235 | liger 0.0697    | 1.142 | **+5.3%**  | **MODEL** |
| rmsnorm   | **0.0530** | 1.256 | unsloth 0.0598  | 1.166 | **+11.4%** | gold seed |
| relu2     | **0.0515** | 1.259 | liger 0.0547    | 1.186 | **+5.9%**  | gold seed |
| layernorm | **0.0614** | 1.164 | unsloth 0.0618 / tutorial 0.0667 | 1.109 / 1.063 | **+0.6%** | gold seed |

## Robustness — our adversarial harness exposed non-robust expert kernels
Of 11 expert recipes harvested, **5 failed our harness** on the odd shape (37 × 4097, fp16) —
`liger/rms_norm`, `liger/fused_add_rms_norm`, `liger/geglu`, `liger/layer_norm`,
`unsloth/geglu_exact` (CompilationError / TypeError on non-power-of-2 N). Our kernels handle these
shapes. 6 verified: liger relu2/swiglu, tutorial softmax/layernorm, unsloth rms/layernorm.

## Honest reading
- **Model-authored wins (the thesis):** `softmax` (1.85×MA, +1.7% vs the Triton-tutorial kernel)
  and `swiglu` (+5.3% vs Liger) — written by Qwen3.6-27B, beat the human kernels.
- **Gold-seed wins:** on `rmsnorm`/`relu2`/`layernorm` our hand-written (Opus) seeds beat
  Liger/Unsloth/tutorial — fair hand-written-vs-hand-written.
- **Margins vary:** `layernorm` is +0.6% (within noise — call it a tie); `rmsnorm` +11.4% is solid.
  These are bandwidth-bound ops at one fixed bench shape; the win is "competitive-to-better than
  expert Triton, and more robust," NOT "categorically faster everywhere."
- Bound unchanged: we beat the compiler/autotuner AND now match-or-beat library Triton on these
  ops; we still do not claim to beat cuBLAS/FlashAttention-class kernels.

_Generated 2026-06-10. H200; harness strong-mode (5-sample bench, max-autotune baseline). App ap-d8PMRQ8xgAeky7ImPR0rpa._
