# OUROBOROS @ Qwen3.6-27B on Modal — Results

A small-model-writes-GPU-kernels run **scaled to Qwen3.6-27B** on Modal H200, with an immutable
harness as the only arbiter. **Correctness is a boolean (allclose vs PyTorch, adversarial
shapes/dtypes/magnitudes); speed is a number (CUDA-event median vs `torch.compile` max-autotune).**
Everything below is verified by that harness — no self-reported numbers.

Artifacts (private): 🤗 `YMRohit/ouroboros-kernelsmith-qwen3.6-27b` + local `./volume_backup/`.

---

## The verified stack (the dependency saga, resolved)
- **Model:** `Qwen/Qwen3.6-27B`, text-only (`language_model_only`), LoRA rank 128, attn=sdpa.
- **Image:** `nvidia/cuda:12.8.0-devel` · torch 2.10 · triton 3.6 · transformers 5.10.2 · peft 0.19.1.
- **Linear-attention (Gated DeltaNet) path:** `flash-linear-attention 0.5.0` + `causal_conv1d
  1.6.2.post1` (prebuilt cu12torch2.10 wheel) + **`tilelang`** — required because triton 3.2–3.6 on
  Hopper (sm_90) computes the GDN backward incorrectly; fla 0.5.0 routes it to a TileLang kernel.
  Verified `READY_TO_TRAIN=True` on H100 (real GDN fwd+bwd, grad 1.86M) before any training.
- **GPU:** H200 (150 GB) — RL peaked ~110 GB, would OOM an 80 GB H100.

## Headline
- **32 / 32 fusion ops beat `torch.compile` max-autotune reproducibly** (5× stability rebench,
  mean − spread > 1.0): 16 original + 16 new — **every one model-authored.**
- **Beat hand-written expert Triton on all 5 comparable ops** (Liger / Unsloth / Triton tutorial).
- Honest bound: these are reproducible **scheduling** wins (~10–85%) over the incumbent autotuner on
  bandwidth-bound ops — NOT wins over cuBLAS/FlashAttention-class kernels, NOT novel algorithms.

---

## Step 1 — SFT→RL on the original suite  ✅
Adapters `sft_adapter` → `rl_adapter` (the kernel-writer). **16/16 robust discoveries** vs
max-autotune. Standouts: `softmax 1.84×`, `softmax_scale 1.44×`, `rmsnorm_gelu 1.49×`,
`layernorm_gelu 1.43×`. → `reports/rebench_stability_qwen3.6-27b.md`
(Note: the RL adapter was almost lost — the original `rl_kernelsmith` never saved the policy weights;
fixed with `save_pretrained` + a hard guard that aborts rather than report a false success.)

## Step 2 — Efficacy + variety  ✅
- **2a · vs human experts** (`reports/headtohead_experts_qwen3.6-27b.md`): harvested Liger/Unsloth/
  Triton-tutorial kernels through OUR harness. **Ours faster on all 5 comparable ops** — `softmax`
  (+1.7% vs the tutorial kernel) and `swiglu` (+5.3% vs Liger) are MODEL-authored; `rmsnorm`/`relu2`/
  `layernorm` (gold seeds) beat their expert counterparts too. Our adversarial harness also exposed
  **5/11 expert kernels as non-robust** (CompilationError on odd shape 37×4097).
- **2b · KernelBench** — **intentionally skipped.** Our model is a norm/activation-fusion specialist;
  KernelBench is generalist (matmul/conv) in a foreign `ModelNew` format. An unassisted run scores
  ~0 (out of scope/format); any prompt-prefill/wrapper to lift it is a non-comparable shortcut. The
  honest evidence (32/32 vs max-autotune + beating experts) already answers "are we good?".
- **2c · grammar** expanded **12 → 44 verified fused ops** (`chains.py`): added `tanh/sigmoid/relu/
  square` then `abs/softsign/hardsigmoid/hardswish` epilogues × {rms,layer}norm × ±residual. Each new
  op gold-passes the harness AND its negative control is rejected (`verify_chains`, 32/32).

## Step 3 — Genuine new-kernel inventions  ✅
Adapter `rl_adapter_newops`. RL self-distill on the 16 new ops (resumed from `rl_adapter`) →
**16/16 new ops are robust, model-authored discoveries** vs max-autotune, incl. every `tanh` fusion
(`layernorm_tanh 1.47×`, `rmsnorm_tanh 1.45×`). → `reports/discovery_newops_qwen3.6-27b.md`

**Key finding:** RL self-distill (reward = a correct, fast kernel) taught the new ops — including
`tanh` — that a 4.5 h continue-SFT pass could NOT (it stalled + interfered with proven ops, so it was
stopped). The verifier's reward beats corpus-imitation for teaching new ops.

---

## Where everything lives
HF repo `YMRohit/ouroboros-kernelsmith-qwen3.6-27b` (and mirrored in `./volume_backup/`):
- `sft_adapter/`, `rl_adapter/` (orig 16/16), `rl_adapter_newops/` (new 16) — LoRA, 2.55 GB each
- `best_kernels/` — 32 model-authored fast Triton kernels (THE product)
- `datasets/` (corpus, also `YMRohit/ouroboros-kernel-corpus`) · `reports/` (the 4 below)
- Reports: `cold_baseline`, `rebench_stability`, `headtohead_experts`, `discovery_newops` (.md)

## Reproduce (Modal entrypoints in `modal_app.py`)
```bash
.venv-modal/bin/modal run modal_app.py::verify_fastpath   # prove tilelang GDN backward on Hopper
.venv-modal/bin/modal run modal_app.py::selftest          # harness: gold pass, wrong rejected
.venv-modal/bin/modal run --detach modal_app.py::train_all   # corpus→SFT→RL→rebench (full)
.venv-modal/bin/modal run --detach modal_app.py::rl --ops <OPS> --adapter outputs/rl_adapter \
      --save-adapter outputs/rl_adapter_newops --out reports/kernelsmith_newops.json
.venv-modal/bin/modal run modal_app.py::rebench           # 5× stability vs max-autotune
.venv-modal/bin/modal run modal_app.py::verify_chains     # gold-pass/wrong-reject the grammar
.venv-modal/bin/modal run modal_app.py::bench_experts     # head-to-head vs Liger/Unsloth/Triton
```
Op sets: `ALL_OPS` (31 SFT), `RL_OPS` (16), `DISCOVERY_OPS` (16 new), `FULL_SFT_OPS`/`FULL_RL_OPS`.

> `README_MODAL.md` is the ORIGINAL 2B setup guide (Qwen3.5-2B, torch 2.12, A100) — superseded by the
> 27B config baked into `modal_app.py` (H200, torch 2.10 + tilelang). Read it for the Modal workflow,
> but the pinned versions/model there are the old run's.

## Integrity notes (what we refused)
- Stopped the full continue-SFT when it interfered with proven ops (didn't burn epochs chasing it).
- Skipped/declined the KernelBench prompt-prefill shortcut (would inflate a non-comparable number).
- Every "discovery" is allclose-verified + beats max-autotune across 5 fresh runs; nulls reported.
_Generated 2026-06-10._
