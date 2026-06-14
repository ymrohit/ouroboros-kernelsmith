# OUROBOROS KernelSmith

OUROBOROS trains a model to write Triton GPU kernels, then scores those kernels with a
referee the model cannot edit.

The current run uses a Qwen3.6-27B model trained through verifier feedback. It produced 69
Triton fusion kernels that beat `torch.compile(mode="max-autotune-no-cudagraphs")` in the
recorded stability gate. The result is not self-reported by the model. A kernel only counts
after it compiles, matches a PyTorch reference on adversarial inputs, and is timed with CUDA
events against eager PyTorch, `torch.compile`, and max-autotune.

## Summary

- 69 of 69 final kernels beat max-autotune reproducibly across 5 fresh runs.
- 67 of 69 winning kernels are model-authored.
- The explore arm contributed zero winning kernels.
- The V2 suite covers 101 verified specs: explicit ops, generated norm chains, and V2
  standalone transformer ops.
- The harness rejects wrong kernels, shape-specialized kernels, memoized outputs, and
  in-place input mutation.

The harness is the central artifact. The model can be swapped out; the measurement path is
what makes the result inspectable.

## Verified Results

Every row below points to a recorded report in this repository.

| Check | Result | Report |
| --- | --- | --- |
| Shape-grid rebench, 376 cells, V2 on H200 | 32 of 32 ops have geomean > 1.0 vs max-autotune. Overall geomean: 1.494x. Cache-cold verified; loss cells are reported. | `reports/rebench_shapes_qwen3.6-27b.md` |
| 5-run stability rebench on H200 | 32 of 32 ops beat max-autotune reproducibly. Includes 16 original ops and 16 new ops; 30 of 32 are model-authored. | `reports/rebench_stability_qwen3.6-27b.md` |
| V2 discovery run, 37 unseen ops on H200 | 37 of 37 validated and beat max-autotune fresh, with speedups from 1.16x to 2.09x. 35 of 37 are model-authored. | `reports/kernelsmith_v2.json` |
| V2 shape-grid for the 37 new kernels, 440 cells | 37 of 37 have geomean > 1.0. Overall geomean: 1.480x. 51 loss cells are reported, with 46 in one known regime. | `reports/rebench_shapes_v2_qwen3.6-27b.md` |
| V2 final stability gate, all 69 kernels | 69 of 69 kernels beat max-autotune reproducibly. Per-sample JSON is included. | `reports/rebench_stability_v2.md` |
| Head-to-head vs expert Triton | OUROBOROS is faster on all 5 comparable ops under the fixed-schedule condition. | `reports/headtohead_experts_qwen3.6-27b.md` |
| RL vs continue-SFT on 16 unseen ops | RL self-distillation learned all 16. Continue-SFT stalled and was stopped. | `reports/discovery_newops_qwen3.6-27b.md` |
| Harness self-test, V2 | 14 gold kernels pass, 13 negative controls are rejected, and 3 anti-gaming controls are rejected. | `ouroboros/reports/harness_selftest.json` |
| End-to-end composed MLP block, V2 on 4090 | Correct block output, 1.085x vs eager and 1.301x vs compile-MA at block level. | `ouroboros/reports/e2e_block.json` |

## Scope

These are scheduling wins for bandwidth-bound fusion operators at measured shapes. They are
not claims of new algorithm classes, and they are not claims against cuBLAS or
FlashAttention-class kernels.

The shape-grid reports are part of the claim, not an appendix. They show where wins hold and
where they do not. Loss cells are reported per cell.

One comparison has a known caveat: the forward-only cross-entropy comparison is unfair to
Liger because Liger computes gradients in its forward path. The reports flag that wherever
the number appears.

## Referee Design

The referee checks both correctness and speed:

1. Import the candidate kernel in an isolated worker.
2. Check `allclose` against PyTorch across adversarial shapes, dtypes, and magnitudes.
3. Include the benchmark shape in the correctness sweep.
4. Reject kernels that mutate inputs.
5. Time with CUDA events after warmup.
6. Compare against eager PyTorch, `torch.compile`, and max-autotune.
7. Re-run winners through stability and shape-grid gates.

V2 added three explicit anti-gaming controls:

- A shape-specialized kernel that only works at the public benchmark shape.
- A memoizing kernel that caches output by input pointer.
- An in-place mutation kernel that corrupts the timing contract.

All three are rejected by the self-test. A rotate mode is also available for cache-cold
cross-checks.

## Op Suite

The suite currently contains 101 verified specs:

- 16 explicit transformer-style ops, including norms, softmax-family ops, GLU-family ops,
  RoPE, and dequant.
- 76 generated chain ops from `[optional residual] -> {RMSNorm, LayerNorm} -> epilogue`
  across 19 epilogues.
- 5 V2 standalone ops: `softcap_softmax`, `rmsnorm_gemma`, `glu`,
  `rope_interleaved`, and `cross_entropy`.

An op enters the loop only after its gold seed passes the harness and its negative control
is rejected. `l1norm` was evaluated and excluded because its outputs sit below fp16
tolerance, making `allclose` uninformative.

## Reproduce

Modal entrypoints:

```bash
.venv-modal/bin/modal token new
.venv-modal/bin/modal run modal_app.py::selftest
.venv-modal/bin/modal run modal_app.py::verify_chains
.venv-modal/bin/modal run --detach modal_app.py::train_all
.venv-modal/bin/modal run --detach modal_app.py::rebench_shapes
.venv-modal/bin/modal run --detach modal_app.py::bench_experts
.venv-modal/bin/modal run --detach modal_app.py::e2e
```

Local checks, assuming a CUDA GPU and the training environment:

```bash
python ouroboros/harness.py
python ouroboros/rebench_shapes.py --kernels ouroboros/seed_kernels
```

## Repository Layout

```text
modal_app.py
  Modal entrypoints for self-test, SFT, RL, rebenching, expert comparisons, and E2E runs.

ouroboros/
  harness.py
    Immutable compile, correctness, and benchmark referee.
  specs.py
    OpSpecs and shape-grid input builders.
  chains.py
    Generated fusion grammar.
  seed_kernels/
    Gold kernels, negative controls, and anti-gaming controls.
  teacher_kernels.py
    Harness-filtered teacher corpus.
  sft_train.py
    Phase 1 supervised fine-tuning.
  rl_kernelsmith.py
    Phase 2 GRPO and self-distillation with measured speed reward.
  rebench_stability.py
    Five-run stability gate.
  rebench_shapes.py
    Shape-grid rebench.
  external_harvest.py
    Expert-kernel comparison harness.
  e2e_block_bench.py
    Composed transformer-block benchmark.
  CLAUDE.md
    Project notes and guardrails for this experiment.

reports/
  Recorded H200, 4090, stability, shape-grid, ablation, and expert-comparison outputs.

volume_backup/
  Snapshot of relevant Modal volume artifacts.
```

## Integrity Notes

- A 4.5 hour continue-SFT run was stopped when it interfered with already proven ops. The
  negative result is reported instead of hidden.
- `kernelbench_eval` is kept for the base-model condition only. The specialist model scores
  near zero there because the format and task family are different; the prompt-prefill bridge
  was not used as a headline number.
- The V1 claim that 5 of 11 expert kernels were non-robust is retired. Through Liger's own
  public API, all 5 tested recipes pass the adversarial sweep. The earlier failures came from
  fixed-schedule extraction wrappers, so both conditions are now reported separately.
- The RL "KL" guard is a sequence-logprob drift penalty and is labeled that way in the
  relevant code and reports.

## Artifacts

- Model and reports: `YMRohit/ouroboros-kernelsmith-qwen3.6-27b`
- Verified corpus: `YMRohit/ouroboros-kernel-corpus`
- License: MIT
