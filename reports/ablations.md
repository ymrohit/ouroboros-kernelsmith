# Ablation suite — which ingredients of the loop actually matter?

**Setup.** Four arms, otherwise identical: same 8 familiar ops (`rmsnorm_gelu,
layernorm_gelu, add_rmsnorm_silu, softmax_scale, geglu, swiglu, add_layernorm,
qknorm_rope`), same starting adapter (`sft_adapter`), 24 rounds, group 8, explore 0.0,
H200, exit validation fresh + max-autotune. One seed per arm (caveat below).
Raw: `ablation_{control,nofeedback,distillonly,nolearn}.json`.

| arm | removes | valid-rate | lead-takes | beat MA | geomean vs MA |
|---|---|---:|---:|---:|---:|
| **control** | — (full loop) | 1.000 | 21 | 8/8 | 1.338 |
| **no-feedback** | harness feedback in prompt | 0.995 | 19 | 8/8 | 1.309 |
| **distill-only** | GRPO advantage term | 0.948 | 23 | 8/8 | **1.361** |
| **no-learn** | all weight updates (frozen best-of-N) | 0.969 | 18 | 8/8 | 1.302 |

*(no-learn provenance: headline read from its completed JSON before a concurrent-save race
overwrote the volume copy with a mid-run checkpoint; per-op rows lost; race documented and
fixed — `_save` now only commits container-modified files. The 8 archived kernels remain
in `outputs/abl_nolearn_kernels/` and can be re-validated if per-op detail is ever needed.)*

## Adjudication of the pre-registered prediction — **mostly FALSIFIED** (catalog entry #9)

*Predicted:* control > distill-only > no-feedback > no-learn on discoveries; feedback
matters most for valid-rate; GRPO matters most for speedup.
*Measured:* **all four arms got 8/8 discoveries** (the bar doesn't discriminate on familiar
ops), valid-rates all ≥0.948 (ceiling — feedback's effect unmeasurable here), and on
geomean **distill-only BEAT control** (1.361 vs 1.338) while frozen best-of-N captured
~97% of control's geomean.

## Honest reading

1. **On familiar ops, the verifier + sampling does most of the work.** A competent SFT
   model plus 24 rounds of verified search nearly matches the full RL loop — learning's
   marginal value on home turf is a few geomean points.
2. **The GRPO advantage term earns nothing here — and may cost a little** (distill-only
   highest geomean, control's higher valid-rate suggesting GRPO trades exploration for
   conformity). Single-seed, so treat the ordering among 1.30–1.36 as suggestive, not
   established; multi-seed repeats are the obvious upgrade before the paper states this.
3. **Where learning demonstrably matters is FOREIGN ops** — the V2 discovery run
   (valid-rate climbing 50%→68% across passes, the softplus/mish idiom going 1–3/8 → 8/8
   via self-distill) and the invention run are the controlled contrast: same loop, unseen
   problems, learning visibly load-bearing.
4. Combined paper claim: *self-distillation on verified winners is the load-bearing
   learning ingredient; the policy-gradient term adds little on familiar ops; search
   against the referee is the dominant force where competence already exists.*

_2026-06-10 · H200 · single seed per arm · all numbers winner's-curse-corrected exit
validations · prediction adjudicated against docs/KEY_FINDINGS.md ledger._
