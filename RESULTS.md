# OUROBOROS @ Qwen3.6-27B on Modal — Results

A small-model-writes-GPU-kernels run **scaled to Qwen3.6-27B** on Modal H200, with an immutable
harness as the only arbiter. **Correctness is a boolean (allclose vs PyTorch, adversarial
shapes/dtypes/magnitudes); speed is a number (CUDA-event median vs `torch.compile` max-autotune).**
Everything below is verified by that harness — no self-reported numbers.

Artifacts (private): 🤗 `YMRohit/ouroboros-kernelsmith-qwen3.6-27b`.

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
HF repo `YMRohit/ouroboros-kernelsmith-qwen3.6-27b`:
- `sft_adapter/`, `rl_adapter/` (orig 16/16), `rl_adapter_newops/` (new 16) — LoRA, 2.55 GB each
- `best_kernels/` — 32 model-authored fast Triton kernels (THE product)
- `datasets/` (corpus, also `YMRohit/ouroboros-kernel-corpus`) · `reports/` (the 4 below)
- Reports: `cold_baseline`, `rebench_stability`, `headtohead_experts`, `discovery_newops` (.md)

GitHub carries the public curated kernel artifact at `artifacts/best_kernels/`: the 69
stability-gated kernels from `reports/rebench_stability_v2.json`.

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

---

# V2 (2026-06-10) — hardening, scale-out, and the claim upgrade

## V2.1 Harness anti-gaming (the moat, proven harder)
Three exploits the v1 harness would have ACCEPTED are now impossible by construction AND
shipped as rejected negative controls: bench-shape special-casing (bench inputs joined the
correctness sweep), pointer-keyed output memoization (large-value poke each timed iteration
+ verify-after-bench), input mutation (contract check). Selftest is now **30 cases — ALL
GREEN on RTX 4090 and H200** (14 gold / 13 negative / 3 anti-gaming). New `rotate` mode
re-benches cache-cold (4 rotating clone-sets).

## V2.2 Op suite: 60 → 101 verified specs
+8 chain epilogues (leaky_relu, relu6, hardtanh, elu, selu, softplus, mish, gelu_erf) ×
{rms,layer}norm × ±residual = 32 new chain ops — **64/64 gold variants pass + 5
cross-epilogue rejections on the production stack (Modal L4)**. +5 standalone real-LLM ops
with gold+wrong seeds (softcap_softmax, rmsnorm_gemma, glu, rope_interleaved,
cross_entropy). `l1norm` evaluated and EXCLUDED — outputs below fp16 atol make allclose
vacuous there (an all-zeros kernel would pass).

## V2.3 Shape-grid re-bench: the headline claim, upgraded ✅
All 32 trained kernels × (M,N)×{fp16,bf16} grid (376 cells) vs max-autotune recompiled per
cell, H200: **32/32 geomean > 1.0, overall geomean 1.494×, 32/32 survive the cache-cold
rotated check (1.10–1.83×).** 38/376 cells (10%) are LOSSES, reported per-cell — 34 of 38
concentrate at 16384×2048 (short-row regime; tells the next round which bench shape to
add). → `reports/rebench_shapes_qwen3.6-27b.md`. The v1 single-shape headline is retired in
favor of this.

## V2.4 Expert head-to-head, now TWO conditions ✅
Condition 2 (NEW): Liger via its OWN public API, as shipped. **All 5 Liger API recipes pass
our adversarial sweep — the v1 "5/11 experts non-robust" claim is RETIRED as a
wrapper-fidelity artifact** (and stated so). H200 results: ours faster on swiglu / rmsnorm /
relu2 / geglu in both conditions; softmax and layernorm are ties-within-noise vs the best
fixed-schedule expert; cross_entropy "win" (0.048ms vs 0.226ms) is FLAGGED — Liger's CE
computes grads in the forward, so forward-only comparison is unfair to them.
→ `reports/headtohead_experts.json`.

## V2.5 End-to-end composed block ✅
Qwen-style MLP sub-block (add_rmsnorm → gate/up GEMMs → swiglu → down GEMM), correctness-
gated, three ways. H200 + trained kernels: **1.148× vs eager, 1.004× vs compile-MA (a tie,
reported as a tie)**; 4090 + seeds: 1.085× / 1.301×. Non-GEMM sub-path 8.5–10.1× faster;
Amdahl ceiling (GEMMs ≈ 83–88% of block) printed in the output. → `reports/e2e_block.json`.

## V2.6 Discovery run on the 37 new ops ✅ (validated; stability + grid gates running)
RL self-distill resumed from `rl_adapter_newops` over the 32 new chain ops + 5 standalone:
111 rounds, group 8, explore-frac 0.0 (the model wrote EVERY kernel), H200, ~4.5h. Adapter
`rl_adapter_v2` saved + hard-guard-verified + pushed. → `reports/kernelsmith_v2.json`

- **37/37 ops validated correct AND beat max-autotune on the fresh exit measurement**
  (winner's-curse-corrected), range 1.16–2.09×. **35/37 winning kernels are LM-authored**
  (only `softcap_softmax` 1.79× and `rope_interleaved` stayed with their gold seeds).
- Standouts: `cross_entropy` **2.085×** MA (model beat its own 2.23×-compile gold seed),
  `rmsnorm_mish` 1.616×, `layernorm_mish` 1.497× — mish/softplus need exp/log overflow
  guards the model demonstrably LEARNED mid-run: pass-1 valid-rate on softplus ops was
  1–3/8, pass-2 was 8/8 (self-distill on pass-1's verified winners taught the idiom).
- Attribution: 617/888 LM kernels verified (69.5%), 101 archive lead-takes, explore arm
  contributed **zero** kernels.
- **Product: 69 model-era kernels** (32 v1 + 37 v2).
- **Shape-grid gate (37 new): PASSED** — 37/37 geomean > 1.0 vs max-autotune (overall
  1.480×), 37/37 cache-cold, 51/440 loss cells with 46 at the same 16384×2048 regime the
  v1 kernels lose in (two independent runs, one identical boundary).
  → `reports/rebench_shapes_v2_qwen3.6-27b.md`
- **5× stability gate over ALL 69 kernels: PASSED — 69/69 DISCOVERIES** (mean − spread
  > 1.0 vs max-autotune on 5 fresh runs each; mean of means 1.299×, max spread ±0.052;
  weakest 1.113 ± 0.015, strongest `cross_entropy` 2.038 ± 0.031). Durable per-sample JSON:
  `reports/rebench_stability_v2.json` (+ .md). The print-only first run was re-done after
  it failed our own traceability rule — recorded in the report's provenance note.

## V2.7 Invention experiment ✅ — the loss regime FLIPPED, and the referee overruled the human
`rl_adapter_v2` + 42 RL rounds on 7 never-trained problems (cumsum scan class, entropy +
kl_div double-reductions, 4 ops pinned to the characterized 16384×2048 loss regime).
**7/7 validated + beat max-autotune; 5/7 model-authored.** The published loss cells became
wins (`add_layernorm_sigmoid` 0.69×→**1.22×**, `layernorm_gelu` 0.88×→**1.45×**) — and NOT
via the split-row schedules the human diagnosis predicted: the model's simpler whole-row
kernels falsified the stated hypothesis. `cumsum`: loop-free `tl.cumsum` whole-row beats
the human carry-loop gold by 47% (invention-lite: primitive selection, labeled precisely).
`entropy`/`kl_div` resisted (capability boundary, stated). LM valid-rate 49.7% on foreign
problems vs ~69% familiar. Product → **76 kernels** (69 stability-gated + the 7 invention
kernels, which stay a **single-shot probe vs max-autotune** — deliberately NOT folded into
the reproducibility set; the paper states this protocol split explicitly).
→ `reports/invention_verdict.md`. The referee-overruled-the-human catalog is summarized
there as of 2026-06-11: 10 falsifications, 2 partial-mixed results, and 1 ratified
prediction.

## V2.8 The repair transfers ✅ (pre-registered prediction — HELD)
Resumed from the invention adapter on the **10 worst remaining loss-cell operators**, pinned
to 16384×2048: **10/10 flipped to wins** (1.20–1.44× vs max-autotune, every kernel
model-authored, explore arm zero, validity 0.713). The whole-row schedule is a carried
skill, not four lucky fixes. → `reports/f1_transfer_verdict.md`, adapter `rl_adapter_f1`.

## V2.9 Ablations: where does the value come from?
**27B, single seed per arm** (8 familiar ops, 24 rounds): all four arms 8/8 vs max-autotune,
geomeans 1.31–1.36 — ordering is seed noise (kept in the paper only to make that point).
**Multi-seed replication: MiniCPM5-1B, 3 seeds × 4 arms, RTX 4090** — arms within seed
noise (max−min 0.045 vs largest per-arm SD 0.030); frozen best-of-N ≈ full loop. **On
familiar operators, search against the referee dominates; learning matters on unseen
operators** (the 37-op run + the softplus/mish idiom acquisition are the contrast).
→ `reports/ablations.md`, `reports/ablations_minicpm_multiseed.md`,
`reports/minicpm_ablation/` (12 per-seed JSONs).

**Data correction (2026-06-11):** the no-learn 27B row originally read 1.302/0.969/18 from
a pre-clobber transcription; the **completed** JSON recovered from the Modal volume (args
confirm the arm, 24/24 rounds, per-op rows intact) gives **1.340/0.995/23** — no-learn
matches control outright, *strengthening* the search-dominates reading. Hardcoded fallback
removed from `paper/make_numbers.py`; provenance note in `reports/ablations.md`.

## V2.10 The paper
`paper/main.pdf` — *"The Referee Is the Product"* (13 pp): negative controls for reward
channels as the contribution, kernels as evidence, plus the falsification ledger (10
contradicted beliefs, 1 held prediction — the 10th is the no-learn transcription the
volume's completed artifact overruled). Every number regenerates from harness JSONs via
`paper/make_numbers.py` (20/20 cross-report consistency checks fail the build on
contradiction). Code + paper: `github.com/ymrohit/ouroboros-kernelsmith` (private).

## V2.11 Hygiene (so the integrity is inspectable)
Git history (v1 snapshot → every V2 change), top-level README (claim→evidence→bounds→repro),
MIT LICENSE, CI + 11 CPU tests, `_2068` dataset naming fixed to actual counts, pseudo-KL
relabeled as the |Δ seq-logprob| drift penalty it is, README_MODAL marked historical.

## Integrity notes (what we refused)
- Stopped the full continue-SFT when it interfered with proven ops (didn't burn epochs chasing it).
- Skipped/declined the KernelBench prompt-prefill shortcut (would inflate a non-comparable number).
- Every "discovery" is allclose-verified + beats max-autotune across 5 fresh runs; nulls reported.
_Generated 2026-06-10._
