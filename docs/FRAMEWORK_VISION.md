# The OUROBOROS framework — the bigger picture

*(Reference doc, written 2026-06-10. The kernel project is the worked example; this is the
general shape of the idea, where it transfers, and where it breaks.)*

## What the framework is, stripped of kernels

> A **replaceable proposer** (any LM), an **immutable oracle** that returns a boolean nobody
> can fake (correctness, checked adversarially) and a scalar nobody can argue with (a
> measurement, not a judgment), **negative controls that test the reward channel itself**,
> and a loop that **distills the model on its own verified wins** (SFT to competence →
> RL/self-distill with reward = the measurement).

Three requirements, and only three:
1. **Cheap verification** — seconds per candidate, so the loop can afford millions of attempts.
2. **Un-gameable verification** — engineered, not assumed: adversarial inputs, negative
   controls, anti-memoization, mutation checks, with known exploits implemented as test
   cases that must be REJECTED.
3. **Measured (not judged) quality** — wall-clock, cycles, bytes, area, proof length. The
   moment the reward is an LLM judge or a human preference, the anti-gaming edifice
   collapses. The framework's power comes from refusing to cross that line.

## The engine: the generation–verification gap

Everything runs on one asymmetry: **verification is cheaper than generation.** Wherever a
candidate can be checked in seconds but takes an expert hours to produce, a model can
propose at scale and reality can referee, and the model improves by training on what
survived. GPU kernels were chosen because the gap is extreme (compile + allclose +
CUDA-event timing ≈ seconds; a human kernel engineer ≈ days). The framework is a general
machine for converting that gap into capability.

## Where it transfers best (ranked)

| # | domain | correctness oracle | quality scalar | incumbent (the "max-autotune") | why it's big |
|---|--------|--------------------|----------------|--------------------------------|--------------|
| 1 | **SQL / query optimization** | result-set equivalence on adversarial data (NULLs, dupes, empty tables = the "odd shapes") | measured plan latency | the query planner | every company pays the query tax; long-tail queries leave 10–100× on the table |
| 2 | **Compilers / superoptimization** | differential testing + bounded formal checks | cycles, code size, energy | existing pass pipelines | every new accelerator without a mature compiler is an instant market (AlphaDev proved the prize; this commoditizes the method) |
| 3 | **Hardware design (RTL/EDA)** | simulation + logical-equivalence checking (industrial oracles exist) | synthesized area / timing / power | hand RTL + synthesis heuristics | slow-but-parallel oracle; augments the most expensive engineers in industry |
| 4 | **Formal proof (Lean/Coq)** | the proof checker — the only truly un-gameable oracle in existence | proof length, elaboration time, dependency weight | tactic libraries / hand proofs | framework adds the quality dimension + negative controls (proofs of the *wrong theorem* must be rejected — mis-stated theorems are how "verified" math goes wrong) |
| 5 | **Systems long tail** | differential correctness vs a reference impl | measured throughput / latency in a sandbox | hand-tuned libraries | serializers, parsers, compression, hash tables, GC/caching policies, congestion control — the "fusion ops" of general software: low prestige, enormous aggregate value |

## The three-layer bigger picture

**Layer 1 — the engine.** Verification cheaper than generation ⇒ capability compounds.
The framework converts any measurable domain into a self-improvement domain.

**Layer 2 — the inversion of what's scarce.** *The model is not the product; the verifier
is.* Models are converging commodities (the loop didn't care when 2B was swapped for 27B).
What's scarce is oracle engineering: adversarial input design, negative controls,
anti-memoization, measurement hygiene. **Verifier engineering is to the RL era what dataset
curation was to the supervised era.** Companies will build moats out of oracles, not weights.

**Layer 3 — the flywheel.** One loop emits three assets simultaneously:
- a library of **verified artifacts** (the product — `best_kernels/`),
- a **verified corpus** (training data for the next model),
- a **benchmark** (the harness + op suite).
Product, data, and evaluation from a single loop, each making the next iteration cheaper.
The SQL instance would emit a rewrite library + a verified-rewrite corpus + the industry's
first honest rewrite benchmark on day one.

## Where it breaks (the boundary is part of the map)

- **Judged, not measured:** writing, design, product sense, "agent helpfulness." LLM-judge
  rewards reopen every gaming vector this framework exists to close.
- **Expensive oracles:** wet-lab biology, physical robotics — the loop starves at one
  verification/day; simulators reopen gaming (a kernel can't fool CUDA events; a molecule
  *can* fool a docking score).
- **Adversarial worlds:** trading, security offense — the "oracle" is other adapting agents;
  the referee assumes physics, not opponents.
- **Goodhart at the metric:** even honest measurements overfit (the shape-grid lesson:
  optimize latency at one shape → overfit schedule). The antidote ships with the framework:
  *measure across the regime, geomean it, publish the losses.*

## One-sentence compression

**A blueprint for turning any domain with cheap, honest measurement into a domain where
intelligence self-improves — relocating the frontier from "who has the biggest model" to
"who has built the referee that can't be fooled."** The kernels were the proof. The referee
is the idea.
