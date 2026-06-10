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

- [ ] *Prediction (Claude):* the ablation arms will rank control > distill-only >
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
