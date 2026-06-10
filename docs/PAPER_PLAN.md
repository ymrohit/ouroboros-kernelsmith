# PAPER PLAN — everything needed for a fool-proof submission

*(Working title candidates, thesis, contribution framing, experiment checklist, reviewer
defense table, venue strategy, artifact checklist, timeline. Written 2026-06-10.)*

## 0. Title candidates

- *"The Referee Is the Product: Un-gameable Verification for LLM-Generated GPU Kernels"*
- *"OUROBOROS: Verifier-Grounded Self-Distillation Beats the Autotuner"*
- *"Negative Controls for Reward Functions: Hardening RL-for-Code Against Reward Hacking"*

Pick based on final framing; the first/third lead with methodology (recommended), the
second leads with the result.

## 1. Thesis (one sentence, goes in the abstract)

A mid-size open model, trained only by an **immutable, adversarially-hardened verifier**
(compile → allclose over adversarial shapes/dtypes/magnitudes → CUDA-event measurement),
learns to write Triton fusion kernels that **reproducibly beat `torch.compile`
max-autotune across a shape grid** and match-or-beat expert hand-written Triton — and the
verifier's *anti-gaming engineering*, not the model, is the load-bearing contribution.

## 2. Contributions (what is actually novel — do NOT lead with "LLM writes kernels")

The crowded prior art: Sakana AI CUDA Engineer, Cognition Kevin-32B, AutoTriton, CUDA-L1,
KernelBench leaderboard entrants. "LLM writes kernels" framing = mental desk-reject.

1. **Negative controls for the reward channel itself.** The harness ships with implemented
   exploits (bench-shape special-casing, pointer-keyed output memoization, in-place input
   mutation) as selftest cases that must be REJECTED. Historical hook: Sakana's AI CUDA
   Engineer (Feb 2025) claimed 150× and was publicly caught within days exploiting eval
   memory reuse — *exactly the cheat class this harness rejects by construction.* "Unit
   tests for reward functions" is a methodological contribution that travels beyond kernels.
2. **RL-from-verifier beats continue-SFT for extending a specialist.** Recorded result:
   verifier-rewarded self-distillation taught 16 (V2: +37) unseen fusion ops; a 4.5h
   continue-SFT pass stalled AND interfered with proven ops and was stopped. Clean,
   decision-relevant finding with receipts.
3. **An honest-measurement stack for model-generated code**, assembled and tested: CUDA
   events + dual-path warmup + median-of-N; clone/setup outside the timed window;
   anti-memoization poke + verify-after-bench; input-mutation rejection; bench-shape in the
   correctness sweep; rotating-buffer (cache-cold/L2) cross-check; winner's-curse
   revalidation of in-search maxima; shape-grid geomean with **losses reported per-cell**.
4. **Controlled scale comparison:** identical pipeline at Qwen3.5-2B and Qwen3.6-27B —
   almost nobody has this for verifier-driven code RL.
5. **Artifact:** 101-op verified suite (specs + gold seeds + negative controls), verified
   kernel corpus, trained adapters, the harness, all reproducible via Modal entrypoints.

## 3. Headline results to report (all harness-emitted, never self-reported)

- 32/32 ops beat max-autotune reproducibly on H200, 5× stability (mean − spread > 1.0).
- Shape-grid: per-op geomean vs max-autotune across (M,N)×{fp16,bf16} WITH loss cells
  (e.g. add_rmsnorm geomean 1.544×, 12/12 cells, cache-cold 1.258× — fill final table when
  the grid run lands).
- Two-condition expert head-to-head: fixed-schedule AND library-API (Liger as shipped);
  state plainly which are wins vs ties-within-noise; **retire** the v1 "5/11 experts
  non-robust" claim as a wrapper artifact; flag CE forward-only as unfair to Liger.
- E2E composed block: correct + measured block-level speedups with the Amdahl bound stated
  (4090: 1.085× eager / 1.301× compile-MA; H200: 1.148× eager / 1.004× compile-MA — the
  tie is reported as a tie).
- V2 discovery run: N/37 new ops with model-authored, stability-verified wins (fill when
  the RL run lands). LM-authorship attribution rates from the loop's own bookkeeping.
- Harness selftest: 30 cases (14 gold / 13 negative controls / 3 anti-gaming cheats) ALL
  GREEN on RTX 4090 + H200; grammar 64/64 on L4.

## 4. Required experiments BEFORE submission (the gap between preprint and acceptance)

**Ablations (all runnable on Modal with existing code):**
- [ ] Feedback ablation: loop WITHOUT harness feedback in the prompt (no fix→retry signal).
- [x] RL ablation: DONE (distill-only BEAT control on geomean; GRPO earns nothing on familiar ops).
- [x] SFT-only best-of-N: DONE (captures ~97% of control geomean on familiar ops).
- [ ] Explore-arm ablation: explore-frac 0.0 (recorded) vs 0.5 — who finds the wins?
- [ ] Group size sensitivity (4 vs 8).
- [ ] Anti-gaming OFF ablation (v1 harness) on an adversarial proposer: demonstrate a cheat
      kernel that v1 ACCEPTS and v2 rejects (the three controls already do this — present
      as a table).
- [ ] Scale study: 2B vs 27B on the same op subset (valid-rate, discovery count, mean
      speedup) — data partially exists from the v1 2B run.
- [ ] Third GPU arch for the headline subset (L4 or A100; 4090 + H200 already exist).
- [ ] KernelBench base-model condition (code exists: `kernelbench_eval`, no-adapter) — the
      honest cross-walk number, reported as out-of-scope-by-construction for the specialist.

**Statistics:**
- [ ] Report n, median, spread for every claim; geomean + win-rate for grids; describe the
      "mean − spread > 1.0" discovery bar and its rationale (small-sample honesty) or
      upgrade to bootstrap CIs (cheap: K=10 instead of 5).

## 5. Reviewer-objection defense table (write these INTO the paper)

| objection | defense |
|---|---|
| "These are just block-size/schedule tunings" | Yes — and the baseline IS the incumbent's tuner (max-autotune). Beating the tuner's own search space reproducibly, shape-grid-wide, on model-invented kernels for new ops is the claim; "novel algorithm classes" is explicitly disclaimed. |
| "Narrow op class — no matmul/attention" | Stated as the honest bound up front. The op class chosen is where the compiler under-fuses (the real long tail of practitioner work); cuBLAS/FA-class kernels explicitly out of scope. |
| "Wins are small (10–85%)" | Bandwidth-bound ops have a physical ceiling; report the roofline %% where possible. Fleet-scale inference economics argument. |
| "Single vendor / Triton only" | Acknowledged; framework is oracle-side, not language-side — port cost is a new harness, not a new method. |
| "Expert comparison unfair" | TWO conditions (fixed-schedule + library-as-shipped), both reported; the v1 unfairness was found and retired by us, in-paper. |
| "Benchmark gaming / reward hacking" | The centerpiece: implemented exploits as rejected negative controls; anti-memoization poke; verify-after-bench; bench-shape in correctness sweep; cache-cold rotation. Invite the reader to submit a cheat. |
| "Why not bigger/closed models" | The model is the replaceable part (2B→27B swap); thesis is verifier-side. Closed models can't be self-distilled. |
| "Single-shape overfit" | The shape-grid section exists precisely for this; losses reported per-cell. |
| "Is the corpus leaking the answers?" | Teacher corpus is structurally-diverse implementations of KNOWN ops; the discovery claim rests on the 37 NEW ops taught by RL with no new corpus (continue-SFT failed at the same task). |

## 6. Paper outline

1. **Intro** — the reward-hacking incident (Sakana) as the opening; thesis: the referee is
   the product. Contributions list.
2. **The harness** — design + the three exploits as negative controls (figure: selftest
   table). Measurement hygiene stack.
3. **The loop** — specs/grammar (101 ops), teacher corpus → SFT gate → RL/self-distill
   (reward = measured speedup), dedup, winner's-curse revalidation.
4. **Results** — stability rebench; shape-grid (geomean + loss heatmap figure);
   two-condition expert head-to-head; e2e block w/ Amdahl bound; discovery-on-new-ops
   (RL vs continue-SFT).
5. **Ablations + scale study** (section 4 checklist).
6. **Honest bounds & negative results** — every loss, the CE caveat, the rope 4090 loss,
   what l1norm taught about vacuous tolerances.
7. **Generalization** — the framework conditions (cheap/un-gameable/measured), transfer
   targets (1 paragraph; cite FRAMEWORK_VISION.md thinking), Goodhart antidotes.
8. **Related work** — Sakana AI CUDA Engineer, Kevin-32B, AutoTriton, CUDA-L1, KernelBench,
   AlphaDev/AlphaTensor, AlphaProof, RLVR literature, superoptimization (Souper, STOKE).

## 7. Venue strategy

- **Primary: MLSys** (systems+ML, artifact-friendly, values measurement honesty).
- Alternate: NeurIPS Datasets & Benchmarks (frame harness + 101-op suite + corpus as the
  benchmark contribution).
- Fast path NOW: arXiv preprint + public repo + workshop (ES-FoMo / MLArchSys-class) for
  feedback and timestamp; full venue after ablations.
- Artifact evaluation: the repo already reproduces via Modal entrypoints — submit for the
  artifact badge; it is a differentiator.

## 8. Artifact checklist (world-class = inspectable)

- [ ] Push git repo to public GitHub (history shows the integrity decisions).
- [ ] Open the HF repos (kernels + corpus) or publish a public mirror subset.
- [ ] Pin everything (image digests, seeds — mostly done), one-command repro per claim.
- [ ] CI badge (CPU tests) + documented GPU gate (selftest ALL GREEN).
- [ ] `pip install ouroboros-kernels` package: dispatch layer + fallback-to-torch +
      harness as its test suite (the "product surface").
- [ ] Per-figure data: every table/figure generated from a JSON in `reports/` by a script.

## 9. Timeline (aggressive but real)

| when | what |
|---|---|
| now (runs finishing) | V2 reports land → RESULTS.md V2 section; stability-gate new kernels |
| days 1–3 | ablation suite on Modal (sec. 4); third arch; KernelBench base condition |
| days 3–5 | scale-study table (2B data exists); figures from reports JSON |
| days 5–8 | draft (outline above); internal red-team pass = try to cheat the harness |
| day 8+ | arXiv + public repo + workshop submission; MLSys when the cycle opens |

## 10. The one rule

Every number in the paper must be traceable to a harness-emitted JSON in `reports/`. If a
number can't be regenerated by a command in the README, it doesn't go in the paper. This is
the discipline that makes it fool-proof — the paper inherits the property of the harness:
**nothing self-reported, everything measured.**
