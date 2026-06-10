# OUROBOROS — a tiny model that writes the GPU kernels that run it

> A small, replaceable proposer writes Triton kernels. An **immutable harness** compiles
> them, checks `allclose` against a PyTorch reference, and benchmarks wall-clock on the
> 4090. **Correctness is a boolean; speed is a number.** The model is not the product —
> **the verifier-compiler harness is.**

This domain is a deliberate descendant of `../sec_sqli/discovery_specialist`, which
succeeded by grounding every claim in a real observed effect (a seeded canary exfiltrated
from a live app), never a pattern match. OUROBOROS ports that discipline to GPU kernels:
the "canary" is `allclose(out, ref) == True`, and the "exploit strength" is the measured
`t_baseline / t_kernel`.

---

## The thesis you are defending

The demo is undeniable because no cloud entrant can fake a live benchmark honestly. So the
**entire value lives in the harness being un-gameable**. Everything else (the proposer, the
search, the UI) is replaceable scaffolding around that one honest measurement.

## Doctrine (read before touching anything)

1. **The harness is the product. Build/trust it FIRST.** `harness.py` is the immutable
   referee — the analog of `discovery_specialist/dvwa_oracle.py`. A hand-written kernel
   must flow through it to a green tick + honest latency *before* any model is wired. That
   gate is **proven**: `python harness.py` → `reports/harness_selftest.json`
   (gold kernels pass; deliberately-wrong kernels are REJECTED).

2. **Benchmark honesty is the whole ballgame — this codebase has been burned.** Memory:
   the `matrix_formula_rewrites` "win" was a verifier certifying an ablation artifact. The
   GPU analog is a benchmark that times launch-async noise, JIT/inductor *compilation*, or
   work the compiler elided. The harness guards every one of these and **they are
   non-negotiable**:
   - **CUDA events + `synchronize`**, never `time.time()` around a launch (it's async).
   - **Warm up BOTH paths** (Triton JITs on first typed launch; `torch.compile` traces on
     first call). Cold = you're timing compilation.
   - **Median-of-N** (the 4090 boost-clocks drift).
   - **Honest baseline = `torch.compile`, and report LOSSES plainly.** Beating eager is the
     floor; beating `torch.compile` is the flex. Do **not** tune toward a green number.

3. **Correctness is ADVERSARIAL, not just multi-shape.** `specs.py` sweeps shape **and
   dtype (fp16/bf16/fp32) and magnitude (×1/×8/×64)**. A kernel correct only on benign
   N(0,1) inputs (softmax with no max-subtraction; RMSNorm with no `rsqrt`) **fails** — that
   is the negative-control analog of dvwa_oracle's benign / SQL-ish-but-non-exfiltrating
   payloads. Tolerances are **derived from fp accumulation, never hand-tuned to pass**.

4. **Subprocess isolation + hard timeout, from day one.** Triton kernels segfault and hang.
   `evaluate()` runs compile+run+bench in a child the parent kills on timeout. A crash
   becomes `status="crash"`, never a silent "ok".

5. **State honest bounds up front** (the `discovery_specialist` habit). Current measured
   result (`reports/harness_selftest.json`, 4090, fp16 8192×4096, median-of-100): the
   hand-written gold kernels beat **eager 4.7–8.1×** and **`torch.compile` 1.08–1.13×**.
   The compile win is real but **narrow (~8–13%)** — so it must clear the noise floor to be
   claimed: run `harness.py` twice; `speedup_compile` drifts ~±0.01×, so an 8% win is solid,
   but never report a sub-few-percent "win" without that variance check. Lead any demo with
   the large guaranteed eager win; treat beat-compile as the flex, and **report losses
   plainly when an op ties or loses**.

   ⚠️ This number was wrong once already: an earlier `_bench` cloned the inputs *inside* the
   CUDA-event window, timing a ~134 MB memcpy alongside the kernel and pulling every ratio
   toward 1.0 (faking a compile "tie"). That is the matrix-rewrite-artifact failure mode in
   GPU form. The lesson is load-bearing: **clone/setup OUTSIDE the timed window; only the
   kernel launch goes between the events.**

## The one re-map from `discovery_specialist` (do not get this wrong)

In the specialist, **diversity IS the objective** — `mech_of` powers a novelty-or-die
reward and the product is mechanism coverage (a QD archive). **Here the objective is the
single FASTEST CORRECT kernel per op.** One op, one number. So:

- The reward is the **scalar measured speedup**, optimized by UCB/GRPO — *not* novelty.
- The `mech_of` analog (`canonicalize.py`) is **dedup only**: don't pay the harness to
  re-measure a kernel that canonicalizes to one already seen. It never scores anything.

Importing `discovery_specialist/rl_discovery.py`'s novelty-only reward here would be a
category error.

## Files

| file | role | analog in `discovery_specialist` |
|---|---|---|
| `harness.py` | **immutable** compile→allclose→bench referee; subprocess-isolated | `dvwa_oracle.py` |
| `specs.py` | `OpSpec` suite (rmsnorm, softmax, swiglu, add_rmsnorm): reference fn, adversarial inputs, derived tolerances, fixed bench input | `OBJ`/`ENVS` env defs |
| `seed_kernels/` | hand-written **gold** Triton + deliberately-**wrong** negative controls | `HAND_SEEDS` + anti-cheat negatives |
| `canonicalize.py` | AST-normalize + hash for **dedup**; knob extraction | `mech_of` (re-mapped: dedup, not reward) |
| `teacher_kernels.py` | **PHASE-1 corpus**: teacher (Opus)-authored STRUCTURALLY-DISTINCT kernels per op (two-pass vec/scalar, whole-row, online softmax) × knob variants → harness-filtered | the diverse bootstrap |
| `sft_train.py` | **PHASE 1 — proper SFT**: build verified corpus → SFT LoRA to convergence → per-op/per-temp valid-rate gate (≥80%) → save adapter | bootstrap SFT (but real, not a warm-up) |
| `rl_kernelsmith.py` | **PHASE 2 — RL** on the SFT'd model (`--load-adapter`): GRPO + self-distill, reward = speedup, dedup-gated, per-emission attribution | `rl_specialist.py` |
| `reports/` | durable verdicts (`harness_selftest.json`, `sft.json`, `kernelsmith_*.json`) | `reports/` |
| `outputs/sft_adapter/` | the SFT'd LoRA (the competent kernel-writer) | trained adapter |
| `outputs/best_kernels/` | THE PRODUCT: fastest correct kernel per op | — |

## Two phases — SFT THEN RL (do not skip Phase 1)

A cold 2B writes **0%** valid Triton (it hallucinates `import triton.core`); with no valid
generations, GRPO has zero signal. So competence must come FIRST, from a **proper SFT** — not
a warm-up. The load-bearing lesson (advisor): SFT on knob-twiddled copies of N seeds teaches
"memorize N programs," not "write Triton." The corpus must be **structurally diverse** — the
teacher (you, Opus) hand-writes genuinely different correct implementations per op; the
harness filters to verified-only. Measured gate, sampled at the RL temperature: **valid-rate
≥80% per op AND outputs diverse (distinct-structure count >1, not one memorized string).**
Only then `--load-adapter` into RL. Verify the adapter reloads **trainable** (PEFT can
silently reload frozen → RL no-op) before the long run.

## Kernel contract

A candidate kernel is a Python **source string** defining `run(*inputs) -> Tensor` matching
the op's reference signature. The harness writes it to a temp `.py` (Triton's `@jit` needs a
real defining file), imports it with a `torch/triton/tl` preamble, and is the only thing
that ever executes it. See `seed_kernels/rmsnorm.py` for the canonical shape.

## Reproduce

```bash
VENV=/home/tihor/webllm/.venv-train/bin/python      # the GPU venv (torch 2.12 + triton 3.7, 4090)

# 1. Prove the moat (D1–D2 gate): gold passes, wrong rejected, honest speedups.
$VENV harness.py

# 2. Search/dedup/archive layer WITHOUT a GPU LM (cheap, fast):
$VENV rl_kernelsmith.py --no-llm --ops rmsnorm,softmax,swiglu --rounds 12 --group 6

# 3. PHASE 1 — proper SFT (model lives in a LOCAL cache; custom HF_HOME HIDES the token,
#    so pass it explicitly). Trains to a measured ≥80% valid-rate gate, saves outputs/sft_adapter:
export HF_HOME=/home/tihor/webllm/verifier-self-distillation-loop/domains/ouroboros/.hf-cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # needed if (re)downloading the model
$VENV -u sft_train.py --epochs 30 --patience 6 --eval-temps 0.4,0.7 --eval-k 6 --gate 0.8

# 4. PHASE 2 — RL on the SFT'd model. explore-frac 0.0 = MODEL writes every kernel (no crutch);
#    --no-kl keeps VRAM ~13GB (<20GB). PYTORCH_CUDA_ALLOC_CONF=expandable_segments curbs frag.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV -u rl_kernelsmith.py \
  --model Qwen/Qwen3.5-2B --load-adapter outputs/sft_adapter \
  --ops rmsnorm,softmax,swiglu,add_rmsnorm --rounds 48 --group 4 --no-kl --explore-frac 0.0

# dedup sanity:
$VENV canonicalize.py
```

VRAM note: run the LM with `python -u` (else prints block-buffer and it looks hung). Two model
copies (KL ref) + the linear-attention torch fallback + a concurrent harness subprocess can
spike to ~24GB — use `--no-kl` + `--group 4` + `--max-new` cap to hold ~13GB.

## Adding an op

Add an `OpSpec` to `specs.SPECS` (reference, adversarial `make_inputs`, fixed `bench_inputs`,
signature hint) and a gold `seed_kernels/<op>.py` + a wrong negative control. Wire both into
`harness._selftest`'s case list and confirm gold passes / wrong is rejected **before** the
op enters the search. No op joins the loop until its negative control fails.

## Hard rules

- **Never** let the proposer or any trainer touch `harness.py`, `specs.py` references, or
  the tolerances. They are the immutable referee.
- **Never** replace `torch.compile` with eager as "the baseline" to manufacture a win.
- **Never** report a speedup without the correctness boolean that earned it.
- If a kernel can't get a green tick by hand, the model never will — fix the harness/spec,
  not the bar.
