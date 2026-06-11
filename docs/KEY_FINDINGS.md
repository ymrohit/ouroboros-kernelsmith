# KEY FINDINGS — every time the referee overruled a human

*(The project's most important pattern, cataloged with evidence pointers. The thesis says
"the verifier is the product"; these are the moments that PROVED it — each one is a human
belief, stated in advance, that the harness falsified. 2026-06-10.)*

## The flagship: the loss-region diagnosis was wrong — the model found the simpler truth

**Human belief (mine, written in `rebench_shapes_qwen3.6-27b.md` before the run):** the
16384×2048 loss regime needs *split-row / multi-row-per-program* schedules — "the
rmsnorm_wide lesson."
**What the verifier certified instead:** the model's winning kernels are plain
**whole-row single-block** schedules. The losses were caused by the looped template style
tuned at N=4096 — not by row-per-program at all. The published loss cells flipped:
`add_layernorm_sigmoid` 0.69× → **1.22×**, `layernorm_gelu` 0.88× → **1.45×**,
`softmax_short` **1.80×** in-regime.
**Evidence:** `reports/invention_verdict.md`, `reports/kernelsmith_invent.json`,
kernel sources in `ouroboros/outputs_invent/`.
**Why it matters:** a domain expert's confident, plausible-sounding diagnosis was cheaper
to falsify than to trust. The loop doesn't need the human to be right about *why* — only
the referee to be honest about *whether*.

## The same pattern, seven more times

1. **cumsum: the human gold lost to the model by 47%.** My carry-propagating scan loop
   (the "obviously correct" textbook structure) was beaten by the model's loop-free
   whole-row `tl.cumsum` (1.29× vs ~0.88× MA). The careful human solution was solving a
   constraint (N > BLOCK) that the bench regime doesn't have.
   → `reports/invention_verdict.md`.

2. **"5/11 expert kernels are non-robust" — our own headline claim, retired.** The v1
   head-to-head said Liger/Unsloth kernels failed odd shapes. The fair-condition re-test
   (their own public API) showed **all 5 Liger recipes pass** — the failures were OUR
   extraction wrappers. The harness scrutinized its own builders' claim and killed it.
   → `reports/headtohead_experts.json`, README integrity notes.

3. **Continue-SFT "should" teach new ops — it stalled and interfered; RL didn't.** The
   intuitive move (more supervised data on new ops) degraded proven ops and was stopped;
   verifier-rewarded self-distill taught all 16 (then 37 more). → v1 RESULTS step 3.

4. **The overflow-guard idiom was learned mid-run, measurably.** softplus/mish ops went
   1–3/8 valid (pass 1) → **8/8 (pass 2)** purely from self-distilling pass-1's verified
   winners — capability gain visible inside a single run's log. → `kernelsmith_v2.json`
   history, `evidence/run_logs/rl_v2_37ops_midrun_window.log`.

5. **The anti-memoization poke was too weak — caught by its own negative control.** My
   first poke (copy a sibling element) let the memoizing cheat PASS (perturbation below
   fp16 tolerance). The cheat-kernel control caught my guard's weakness before any model
   could exploit it; the poke was strengthened to a large in-distribution value.
   → harness git history (commit "V2 harness hardening").

6. **l1norm looked like a fine op — the tolerance analysis proved it unverifiable.** Its
   outputs sit below fp16 atol, so an all-zeros kernel would pass allclose. Excluded on
   verification grounds, not added for op-count vanity. → specs commit message; README.

7. **The bench was once measuring a memcpy, calling it a tie.** (v1 era, recorded in
   CLAUDE.md): cloning inputs INSIDE the timed window added a ~134MB copy to every
   measurement and pulled all ratios toward 1.0 — "torch.compile ties" that were artifacts.
   Clone-outside-the-window fixed it. The doctrine's founding scar.

8. **"All-green in the logs" is not evidence — the traceability rule caught the tooling.**
   The first 69-kernel stability gate passed everywhere we looked, but the script was
   print-only and Modal windows logs; our own every-number-from-a-JSON rule forced a
   re-run with durable output. The repo policed its builder. → `rebench_stability_v2.md`
   provenance note.

## The practice this implies: a falsification ledger

From now on, every diagnosis/prediction gets **written down before the run**, so the
referee can grade the human too. Open predictions to adjudicate next:

- [x] FALSIFIED (entry 9) — *Prediction (Claude):* the ablation arms will rank control > distill-only >
      no-feedback > no-learn on discoveries; feedback matters most for valid-rate,
      GRPO matters most for speedup. → adjudicated by `reports/ablation_*.json`.
- [ ] *Prediction (Claude):* entropy/kl_div golds fall to the model given a corpus of
      double-reduction structures + 6 more RL passes (i.e., the boundary is data, not
      capability).
- [ ] *Prediction (Claude):* the whole-row schedule family generalizes — re-pointing the
      ORIGINAL losing chain ops' RL at a 16384×2048 bench shape flips most of the 46
      remaining loss cells without new op definitions.
- [ ] *Prediction (Claude):* a ≤4B model (MiniCPM) reaches ≥80% of the 27B's discovery
      count on the chain family but loses badly on cross_entropy/qknorm-class fusions.

Each of these is falsifiable for a few H200-hours. When one dies, it goes in the catalog
above — that's the point.

## The paper line this catalog earns

*"Across one project we recorded eight instances where the verifier falsified a confident
human belief — including its own builders' diagnoses, claims, and guards. The referee
outranks intuition; the system is designed so that this is cheap to discover."*

## The Falsification Slate (pre-registered 2026-06-10, late)

Beliefs queued for trial, each cheap to adjudicate with existing infra:

- [x] F1 RATIFIED (10/10 flipped) — see reports/f1_transfer_verdict.md
  short-row losses (10 worst chain ops) require something beyond what the invention run
  learned. *Prediction (Claude): NO — resuming from `rl_adapter_invent`, the whole-row
  style transfers immediately (lead-takes in pass 1) and ≥8/10 ops flip to >1.0 vs MA at
  16384×2048.* Run: `rl` on 10 `_short` variants. ADJUDICATING NOW.
- [ ] **F2 — GPU folk wisdom on trial.** Beliefs: "more warps help bandwidth-bound ops,"
  "num_stages matters," "power-of-2 BLOCK is required," "bigger BLOCK is better on H200."
  Method: knob-sweep the 76 winners (no LLM — pure harness), score each folk rule by how
  often following it helps/hurts. *Prediction: num_stages ≈ no-op on these ops; the
  warps rule is shape-dependent, wrong ≥30% of the time.*
- [ ] **F3 — "max-autotune is near-roofline on bandwidth-bound fusions."** Method: compute
  achieved HBM bandwidth %% of peak for ours vs MA per op. *Prediction: MA leaves 15–40%
  of bandwidth on the table on fused chains; ours sits ≥80% of peak on the wins.*
- [ ] **F4 — textbook scheduling folklore (one-pass online softmax beats multi-pass; 
  masking is expensive; vectorized two-pass beats whole-row at large N).* Method: bench
  the teacher corpus's STRUCTURE variants head-to-head across the grid — the corpus
  already contains the competing textbook structures. *Prediction: at least one textbook
  rule inverts across the shape grid.*
- [ ] **F5 — "the adversarial sweep needs many random shapes."** Method: replay every
  rejected candidate, log WHICH case (stress / bench-shape / random-k) carried each
  rejection. *Prediction: stress + bench-shape catch >95%; random sweep is a thin tail —
  informs harness budget.*
- [ ] **F6 — "you need ~27B to write kernels."** MiniCPM-4B full pipeline (also the
  hackathon Tiny-Titan/OpenBMB play). *Prediction: ≥80% of 27B's discovery count on
  chains; clear gap on cross_entropy/qknorm-class fusions.*
- [ ] **F7 — "the prompt exemplar is load-bearing."** Arm with signature-only prompts.
  *Prediction: familiar ops barely affected; cold-op valid-rate halves.*

9. **The ablation prediction — mostly falsified (adjudicated 2026-06-10 late).** Predicted
   control > distill-only > no-feedback > no-learn; measured: all arms 8/8 discoveries,
   **distill-only BEAT control on geomean (1.361 vs 1.338)**, frozen best-of-N captured
   ~97% of control. The GRPO term earns nothing on familiar ops; self-distillation +
   verified search carry the loop. Learning's value concentrates on FOREIGN ops (the
   V2/invention runs are the contrast). Single-seed caveat stated. → `reports/ablations.md`.

## F1 adjudicated (2026-06-11) — prediction RATIFIED (the ledger cuts both ways)

F1 ("the loss-cell fix transfers"): **CONFIRMED — 10/10** worst-losing chain ops flipped to
wins at 16384×2048 (1.20–1.44× vs max-autotune, all model-authored, explore-arm zero),
resuming from `rl_adapter_invent`. The whole-row schedule family is a transferable skill,
not per-op luck; the product's one characterized weakness is closed for the chain family.
→ `reports/f1_transfer_verdict.md`. Note: this prediction was *correct* — pre-registration
grades the human in both directions, and the referee here ratified rather than overruled.
Standing falsifications remain at nine; confirmed predictions now one.

## Inference-only probe (2026-06-11) — prediction partly falsified (entry 10)

Probed the trained 27B at PURE INFERENCE (no RL) on 5 never-RL'd ops, A100-80GB, ~$1.
Result: dequant_int8 4/4 one-shot (1.32x MA), cumsum 1/4 (1.32x MA), rmsnorm_wide /
entropy / kl_div all 0/4. Cleanly separates raw model knowledge (genuine one-shot
generalization exists on some unseen ops) from what the loop adds (reliability + crossing
the coupled-double-reduction boundary + regime transfer). My pre-probe prediction was
partly wrong: too pessimistic on the speedups (guessed 1.0-1.2x, got 1.32x), too optimistic
on rmsnorm_wide (guessed coin-flip, got 0/4). The referee corrected the human in both
directions. → reports/probe_verdict.md. Ledger: 9 falsifications + 2 partial-mixed + 1
ratified.

## Multi-seed ablation on MiniCPM5-1B (2026-06-11) — single-seed caveat RETIRED

Re-ran the 4-arm ablation at 3 seeds each on OpenBMB MiniCPM5-1B (1B, free 4090). Arms
statistically TIED: between-arm spread 0.045 <= 2x max seed-std 0.030. The decisive
cross-check: distill-only was nominally BEST on the 27B single-seed run (1.361) and is
nominally WORST on the 1B 3-seed average (1.057) -- the same arm at opposite ends proves
the single-seed ordering was noise. Robust claim that survives error bars: on familiar ops,
search against the referee dominates; no learning ingredient (feedback/GRPO/learning-at-all)
separates. This RETIRES the paper's 'single-seed; suggestive not established' caveat and
STRENGTHENS the conclusion. Bonus: a 1B model writes kernels beating max-autotune 6/6
(geomean ~1.10x), SFT'd to 100% valid in 2 epochs -- the OpenBMB + Tiny Titan sponsor
result, all at $0. Note: my own seed-0 preview prediction (control on top) was also noise.
→ reports/ablations_minicpm_multiseed.md.
