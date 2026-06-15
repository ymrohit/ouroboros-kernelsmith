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
| **no-learn** | all weight updates (frozen best-of-N) | 0.995 | 23 | 8/8 | 1.340 |

*(no-learn provenance — CORRECTED 2026-06-11: this row originally read 1.302 / 0.969 / 18,
transcribed from a pre-clobber read after a concurrent-save race was believed to have
overwritten the volume copy with a mid-run checkpoint. The Modal volume in fact holds the
**completed** JSON — `args` confirm the arm (`no_learn: true`, same 8 ops, 24/24 rounds),
with full per-op rows and attribution — now restored as `ablation_nolearn.json`; the
locally clobbered round-10 checkpoint is preserved as
`ablation_nolearn_clobbered_checkpoint.json`. The durable artifact supersedes the
transcribed read, which had **understated** no-learn: 1.302→1.340, 0.969→0.995, 18→23.
`make_numbers.py` no longer carries a hardcoded fallback for this row. The 8 archived
kernels remain in `outputs/abl_nolearn_kernels/`.)*

## Adjudication of the pre-registered prediction — **mostly FALSIFIED** (catalog entry #9)

*Predicted:* control > distill-only > no-feedback > no-learn on discoveries; feedback
matters most for valid-rate; GRPO matters most for speedup.
*Measured:* **all four arms got 8/8 discoveries** (the bar doesn't discriminate on familiar
ops), valid-rates all ≥0.948 (ceiling — feedback's effect unmeasurable here), and on
geomean **distill-only BEAT control** (1.361 vs 1.338) while frozen best-of-N matched
control outright (1.340 vs 1.338 — ~100%, post-correction; the pre-correction read said
~97%).

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
validations · prediction adjudicated against the internal findings ledger._
