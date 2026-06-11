# Inference-only probe — what does the trained 27B know without the RL loop?

**Setup.** `rl_adapter_v2` (the 37-op discovery adapter), loaded on an **A100-80GB**
(cheaper than the H200 used for training; 27B fits in bf16). Pure inference: 4 samples per
op at temp 0.6, no RL, no feedback, no retry. Each sample verified through the harness with
the max-autotune baseline. Five ops the adapter was **never RL-trained on**, chosen to span
different reasons for uncertainty. Cost: ~\$1. Raw: `reports/probe.json`.

## Result

| op | why uncertain | verified | best (one-shot) | read |
|---|---|---:|---|---|
| **dequant_int8** | int8, non-GEMM, unusual | **4/4** | 1.316× MA | SFT generalizes cleanly; nails it every time |
| **cumsum** | a *scan* (carry propagation) | **1/4** | 1.321× MA | *can* one-shot a correct fast scan, but unreliably (3 compile fails) |
| **rmsnorm_wide** | tall-skinny regime | 0/4 | none | all 4 compile-fail; the row-per-program style does not transfer to this regime |
| **entropy** | coupled double reduction | 0/4 | none | all 4 incorrect; cannot produce a correct kernel one-shot |
| **kl_div** | coupled double reduction | 0/4 | none | all incorrect/crash; same boundary |

## What this isolates: inference vs the loop

The probe cleanly separates raw model knowledge from what the search loop adds:

1. **Genuine one-shot generalization exists.** On `dequant_int8` (4/4) and `cumsum` (1/4),
   the model produces correct kernels that beat max-autotune (~1.32×) with no RL at all —
   for ops it was never RL-trained on. The SFT'd competence transfers.
2. **The loop's value is reliability + boundary-crossing.** `cumsum` at 1/4 one-shot is
   exactly the case the RL loop fixes: feedback-driven retry turns an occasional success
   into a consistent one (and indeed the invention run, with RL, got a robust cumsum
   discovery). The loop buys reliability where inference is a coin-flip.
3. **The capability boundary is real and sits at coupled double reductions.** `entropy` and
   `kl_div` are 0/4 at inference — the model cannot even produce a *correct* one-shot
   kernel — and the RL loop also never beat the hand-written golds there. Both methods agree
   on where the model's competence ends. That agreement is itself a finding.
4. **Regime transfer fails without search.** `rmsnorm_wide` (tall-skinny) is 0/4 compile
   fails; the model's whole-row style does not even compile in a foreign shape regime
   without the loop searching for an adapted schedule.

## Prediction graded (ledger entry)

Pre-probe prediction (written before the run): *one-shots cumsum + dequant_int8 at
~1.0–1.2× MA; fails to beat baseline on entropy/kl_div; coin-flip on rmsnorm_wide.*
Adjudication: **partly right, partly wrong.** Right that dequant_int8/cumsum are doable and
entropy/kl_div fail (in fact harder than predicted — they fail *correctness*, not just
speed). Wrong on rmsnorm_wide (predicted coin-flip; got 0/4) and I *underestimated* the
speedups (1.32× vs the 1.0–1.2× guessed). Net: the referee corrected the human again, in
both directions.

_2026-06-11 · A100-80GB · inference only (no RL) · k=4, temp 0.6 · strong harness (max-autotune)._
