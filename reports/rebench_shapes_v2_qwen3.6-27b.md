# Shape-grid re-bench — the 37 V2 discovery-run kernels

Same protocol as the v1-kernel grid (`rebench_shapes_qwen3.6-27b.md`): every kernel from
the V2 discovery run across the (M,N) × {fp16,bf16} grid vs `torch.compile` max-autotune
recompiled per cell, V2-hardened harness in every cell, plus the cache-cold rotating-buffer
check at the headline shape. H200, 440 cells, 89.4 min.
Raw: `rebench_shapes_v2_qwen3.6-27b.json`.

## Verdict

- **37/37 ops geomean > 1.0 vs max-autotune. Overall geomean: 1.480×.**
- **37/37 survive the cache-cold rotated check.**
- **51/440 cells (11.6%) are losses — 46 of the 51 at 16384×2048**, the same short-row
  regime where the v1 kernels lose. Two independent training runs, identical loss
  boundary: this is a characterized property of row-per-program schedules, not noise.

| top | geomean ×MA |   | bottom | geomean ×MA |
|---|---:|---|---|---:|
| cross_entropy (LM) | 1.972 | | rmsnorm_gelu_erf | 1.367 |
| softcap_softmax (seed) | 1.840 | | add_rmsnorm_softplus | 1.362 |
| layernorm_hardtanh (LM) | 1.563 | | add_layernorm_gelu_erf | 1.331 |
| rmsnorm_gemma (LM) | 1.557 | | add_rmsnorm_gelu_erf | 1.325 |
| layernorm_relu6 (LM) | 1.554 | | rope_interleaved (seed) | 1.269 |

Even the WORST of the 37 holds a 1.27× geomean across the regime.

## Read

The V2 discovery kernels — written by the model for ops it first saw the same day, with
the explore arm contributing zero — hold up under exactly the scrutiny that the v1 audit
demanded: per-cell max-autotune recompiles, both dtypes, cache-cold, with the loss region
stated. Combined with the v1 grid: **69/69 kernels with geomean > 1.0** across their grids
(overall 1.494× / 1.480×), loss cells 10–12% and confined to one explainable regime.
Next-round target is unchanged and now doubly confirmed: add an N≤2048 bench shape so the
search can find split-row schedules.

_Generated 2026-06-10 · H200 · V2 harness in every cell · losses are part of the result._
