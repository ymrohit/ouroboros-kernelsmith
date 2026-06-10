# F1 — does the loss-regime fix TRANSFER? (pre-registered, adjudicated)

**Pre-registered prediction** (`docs/KEY_FINDINGS.md` slate, written before the run):
resuming from `rl_adapter_invent`, the whole-row schedule style transfers immediately and
**≥8/10** of the worst remaining loss-cell ops flip to wins at 16384×2048.

**Result: 10/10 flipped.** All ten worst-losing chain ops now beat `torch.compile`
max-autotune in the regime where every prior kernel lost, all model-authored, explore-arm
contribution zero, validity 0.713, 30 archive lead-takes.

| op (pinned 16384×2048) | vs max-autotune | vs compile | author |
|---|---:|---:|---|
| add_layernorm_gelu_short  | 1.405 | 1.506 | LM |
| layernorm_tanh_short      | 1.444 | 1.660 | LM |
| add_layernorm_tanh_short  | 1.378 | 1.470 | LM |
| rmsnorm_silu_short        | 1.298 | 1.405 | LM |
| add_rmsnorm_gelu_short    | 1.315 | 1.383 | LM |
| layernorm_sigmoid_short   | 1.262 | 1.552 | LM |
| add_layernorm_silu_short  | 1.233 | 1.336 | LM |
| add_layernorm_square_short| 1.219 | 1.275 | LM |
| add_rmsnorm_silu_short    | 1.210 | 1.243 | LM |
| add_rmsnorm_sigmoid_short | 1.197 | 1.236 | LM |

## Reading

The invention run (§ invention_verdict) found a whole-row schedule family for 4 ops in the
loss regime. F1 tested whether that was four lucky per-op fixes or a **transferable skill**.
Resuming from the same adapter, the model applied the family to 10 ops it had never been
trained on at this shape and won every one. This is the strongest form of the
generalization claim: the discovery was a capability the model now carries, not a memorized
output. The product's single characterized weakness (the 16384×2048 short-row regime,
46 of the shape grid's loss cells) is, on this evidence, closed for the chain family.

## On being right (the ledger cuts both ways)

This prediction was *correct*, and the ledger records it as such alongside the nine
falsifications. The point of pre-registration is not that the human is always wrong; it is
that the referee, not the human's confidence, decides. Here it ratified the prediction; on
the loss-regime *diagnosis* it overruled it. Both verdicts came from the same un-arguable
source.

_2026-06-11 · H200 · exit validation fresh + max-autotune · 40 rounds from rl_adapter_invent._
