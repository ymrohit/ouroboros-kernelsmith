# OUROBOROS — results (2026-06-04)

A tiny model (Qwen3.5-2B) that writes the GPU kernels, verified live by an immutable
harness. Two phases: **proper SFT to competence, then RL to push past PyTorch.**

## The product
- `harness.py` — immutable compile→allclose→bench referee (subprocess-isolated, CUDA-event
  median vs eager / torch.compile / **max-autotune**, deterministic adversarial stress).
- 8 ops: rmsnorm, softmax, swiglu, add_rmsnorm, rope, layernorm, add_layernorm, geglu —
  each gold-passes and its deliberately-wrong negative control is rejected.
- `outputs/sft_adapter_8op` — the SFT'd LoRA (rank 64): the competent kernel-writer.
- `outputs/best_kernels/*.py` — the 8 best kernels, all model-authored where it won.

## Phase 1 — SFT (competence)
Teacher (Opus) hand-wrote **3 structurally-distinct kernels per op** (two-pass vector/scalar,
whole-row single-block, online softmax, shift-load rope, …); harness-filtered to **320
verified kernels, ~9 distinct structures/op**, balanced per-op. LoRA rank 64 + MLP targets.
Result: **cold base 0% → 100% valid on all 8 ops** at temps 0.3/0.5/0.8, 2–5 distinct
structures each (learned to *write* Triton, not memorize one string).

Bug found & fixed: `max_new=384` truncated the long two-pass kernels at eval → falsely
reported layernorm at 0%. The model had learned it; the measurement lied. Fixed → 1024.

## Phase 2 — RL (push past PyTorch), pure-model `--explore-frac 0.0`
- **192/192 generations verified (100%)** — the model writes valid Triton every time.
- **6/8 best kernels authored by the model**; LM took the archive lead 14 times.

## Discovery — 5× stability re-bench vs max-autotune (the incumbent autotuner)
Bar: model-authored kernel beats max-autotune reproducibly (mean−spread > 1.0 over 5 runs).

| op | vs max-autotune (mean±sd) | verdict |
|---|---|---|
| rmsnorm | 1.093 ± 0.026 | **discovery** |
| add_rmsnorm | 1.083 ± 0.059 | **discovery** |
| add_layernorm | 1.076 ± 0.058 | **discovery** |
| softmax | 1.075 ± 0.041 | **discovery** |
| swiglu | 1.070 ± 0.005 | **discovery** |
| layernorm | 1.055 ± 0.031 | **discovery** |
| geglu | 1.005 ± 0.023 | ceiling (ties) |
| rope | 1.158 ± 0.167 | too noisy (max-autotune varies 0.83–1.30) |

**6/8 model-authored kernels beat the incumbent autotuner reproducibly.** Lever confirmed:
4-op suite → 3 discoveries, 8-op suite → 6 — scales with ENVIRONMENT, not optimizer/rounds
(the program-wide result, same as `sec_sqli/discovery_specialist`).

## Honest bounds (held, not inflated)
- Wins are **modest (5–9%) but reproducible** scheduling improvements the autotuner missed —
  NOT novel algorithm classes. RL tunes within the model's SFT'd repertoire.
- rope's flashy in-search 2.15× was winner's-curse + a weak default-compile baseline →
  deflated to 1.58× vs compile and **loses** to max-autotune (0.83×). Not a discovery.
- geglu ties the ceiling. Both nulls reported plainly.
- The harness is the sole arbiter; the model never touches it; every number is
  allclose==True + a CUDA-event median, winner's-curse-validated on a fresh re-bench.

## Reproduce
```bash
VENV=/home/tihor/webllm/.venv-train/bin/python
export HF_HOME=$PWD/.hf-cache HF_TOKEN=$(cat ~/.cache/huggingface/token)
$VENV harness.py                                    # moat: all 8 gold pass, wrong rejected
$VENV -u sft_train.py --lora-rank 64 --gate 0.8     # phase 1 -> outputs/sft_adapter_8op
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV -u rl_kernelsmith.py \
  --load-adapter outputs/sft_adapter_8op --no-kl --explore-frac 0.0 --max-new 1024 \
  --ops rmsnorm,softmax,swiglu,add_rmsnorm,rope,layernorm,add_layernorm,geglu --rounds 48
$VENV rebench_stability.py                          # discovery: beats max-autotune reproducibly?
```
