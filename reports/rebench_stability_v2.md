# 5× stability re-bench — ALL 69 kernels (v1 + V2 discovery run)

**The durable, un-gameable verdict on the full product.** Every kernel in
`outputs/best_kernels/` re-benchmarked **5× fresh** through the V2-hardened harness
(strong mode: max-autotune baseline; anti-memoization poke + verify-after-bench live in
every run). Discovery bar: **mean − spread > 1.0 vs max-autotune** — reproducible, not a
lucky sample. H200. Raw 5-sample data per op: `rebench_stability_v2.json`.

## Verdict: **69 / 69 DISCOVERIES — zero failures, zero marginals**

- Mean of per-op means: **1.299× vs max-autotune**; max spread across all 69: ±0.052.
- Strongest: `cross_entropy` **2.038 ± 0.031** (model-authored), `softmax` 1.848 ± 0.028,
  `softcap_softmax` 1.777 ± 0.018, `rmsnorm_mish` 1.556 ± 0.008.
- Weakest (still clears the bar): `add_layernorm_gelu_erf` 1.113 ± 0.015,
  `add_layernorm_silu` 1.121 ± 0.004.
- 67/69 winning kernels are model-authored (`softcap_softmax`, `rope_interleaved` are
  gold-seed-owned; stated, not hidden).

## Provenance note (why this run exists)

The first 69-op stability run completed all-green, but `rebench_stability.py` was
print-only and Modal's CLI windows the logs — the complete verdict was not durably
recorded, which violates this repo's rule that every claimed number be traceable to a
harness-emitted JSON. The script was patched to emit JSON and the gate re-run in full.
This file and its JSON are that record.

_Generated 2026-06-10 · H200 · V2 harness · 5× median-of-N CUDA-event bench · clone/setup
outside the timed window · stale-cache and bench-shape exploits rejected by construction._
