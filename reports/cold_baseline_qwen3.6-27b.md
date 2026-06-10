# Cold-baseline valid-rate — Qwen/Qwen3.6-27B (pre-SFT)

**Model:** `Qwen/Qwen3.6-27B` (text-only, `language_model_only`) · H200 · transformers 5.10.2 · torch 2.10
**Eval:** harness correctness-only, per op, temps 0.3 / 0.5 / 0.8, k=8 samples · max_new=1024
**Source:** run `ap-8EwHhnLx0ks2smxXCpSMxn` (crashed later in SFT backward — a `fla` Hopper/Triton bug, fixed by `tilelang`).
**Status:** ⚠️ PARTIAL — **8 of 31 ops**. The remaining 23 ops scrolled out of Modal's retained log
window before they were captured (the run wasn't saving `sft.json` yet — the crash hit before the
SFT save step). We keep `--skip-baseline` going forward, so this pre-measurement is not re-run.

| # | op | valid@0.3 | valid@0.5 | valid@0.8 | distinct structs |
|---|----|----------:|----------:|----------:|-----------------:|
| 1 | add_layernorm        | 75%  | 100% | 75%  | 5 |
| 2 | add_layernorm_gelu   | 0%   | 0%   | 0%   | 0 |
| 3 | add_layernorm_relu2  | 88%  | 50%  | 38%  | 6 |
| 4 | add_layernorm_silu   | 50%  | 75%  | 62%  | 4 |
| 5 | add_rmsnorm          | 100% | 100% | 100% | 4 |
| 6 | add_rmsnorm_gelu     | 0%   | 0%   | 0%   | 0 |
| 7 | add_rmsnorm_relu2    | 100% | 88%  | 88%  | 8 |
| 8 | add_rmsnorm_rope     | 0%   | 0%   | 0%   | 0 |

## Read
Even **cold (no training)**, Qwen3.6-27B already writes valid Triton at **75–100%** for several ops
(`add_rmsnorm` 100%, `add_rmsnorm_relu2` 88–100%, `add_layernorm` 75–100%) — it's a very strong coder.
But it hits **0%** on specific fused ops (`*_gelu` fusions, `add_rmsnorm_rope`) — which is exactly what
SFT exists to teach. Mean valid@0.3 over these 8 ≈ **51%** (< the 0.8 gate), so SFT does real work
(it must lift the 0% fused ops), rather than no-op.

_Captured 2026-06-09; partial by necessity (log retention)._
