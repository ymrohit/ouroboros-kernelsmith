# The Verified Playground — building AI, one honest green tick at a time

*(Strategy doc, 2026-06-11. The product idea that fell out of the Kernel Mint: a
block-based, all-ages platform where kids build the real pieces of an AI system, an AI
fills in the hard part, and an un-gameable referee says whether it's real. The thesis of
the whole OUROBOROS project — **the verifier is the product, not the model** — is what
turns this from a toy into an education platform.)*

---

## 1. The core insight: it was never a kernel tool, it was a loop

The Kernel Mint is one instance of a loop that is the same thing that makes Minecraft and
Scratch addictive:

> **snap blocks together → an AI fills in the hard part → an honest referee says if it's
> real → you climb a leaderboard → you understand why.**

Build, instant honest feedback, compete, repeat. Point that loop at GPU kernels today; point
it at bigger pieces of an AI system tomorrow. The blocks become layers; the kid's "spark"
becomes a design idea; "the model builds it for you" becomes the AI completing their sketch.

Nothing about the loop is specific to kernels. What *is* special — and load-bearing — is the
referee. As long as reality grades the work (not vibes, not a human judge, not the model's
own say-so), you can let a child build anything and the green tick means something.

## 2. The Ladder: from kernels to models, getting more powerful and less un-cheatable

Every rung is a Lego piece of an AI system. Each has a referee. The referee gets **more
expensive and easier to fool** as you climb — that gradient is the whole design problem.

| rung | the kid builds… | the referee | cost | un-cheatable? |
|---|---|---|---|---|
| 0 · **Kernel** (today) | a fused GPU op | compile + `allclose` vs PyTorch + time | seconds | **perfect** (bit-exact) |
| 1 · **Math gadget** | softmax, a scan, an attention score | same: reference fn + speed | seconds | perfect |
| 2 · **A layer** | one attention head, an MLP block, a norm+residual unit | output matches a reference impl on random inputs | seconds | strong |
| 3 · **A tiny solver** | a classifier on a *fixed, small* dataset | held-out accuracy on a sealed test set | seconds–minutes | medium (can overfit) |
| 4 · **A recipe** | a data augmentation, an optimizer schedule, a tokenizer | downstream metric on a fixed task | minutes | medium |
| 5 · **A small model** | snap layers into a network, pick data, press train | accuracy / loss on held-out data | minutes–hours | **soft** (gameable) |

**The sweet spot for kids is the middle rungs (1–3).** They are still *real* (a referee can't
be sweet-talked) and still *magic* (you made a thing that works). Rung 0 is the purest and
the best demo; rung 5 is the most impressive-sounding but is really *Kaggle-for-kids* — good,
engaging, but the referee is soft and the "is it good?" question is fuzzy and overfittable.

**Design rule that protects the magic:** never climb to a rung whose referee you cannot keep
honest. The instant the green tick can be faked, the product becomes a vibes-based toy and
loses the one thing that made it trustworthy. Prefer a smaller piece with a perfect referee
over a grander piece with a soft one. (This is literally the lesson of the OUROBOROS paper:
the value is the un-gameable referee; the model is replaceable scaffolding.)

## 3. Product vision — "Build Small Academy"

A block-based playground, browser-first, all ages. Three nested loops:

- **Play:** snap blocks into a machine, add your own spark, press BUILD, watch a small AI try
  to make it real, get an honest verdict in seconds. (The Kernel Mint, generalized.)
- **Progress:** a curriculum of levels, one per rung of the Ladder. Each level teaches one
  real concept by making the kid *build* it, not read about it. Graduate from kernels →
  gadgets → layers → tiny solvers, and you have hand-built a chunk of the actual AI stack.
- **Compete & share:** a leaderboard per challenge, seeded with a big-model "champion" to
  beat (exactly as the Mint seeds the 27B's kernels). Share your machine; others remix it.

**The "spark" mechanic is the heart of it.** The kid types their own idea in plain words
("make it tiny", "use a clever trick", "what if it bent the numbers twice?"), and the AI
tries to honor it while the referee still enforces correctness. The kid sees their words
reflected in real code/architecture they could never have written themselves — and learns
from the gap between what they imagined and what actually verified. Creativity is safe
because reality is the backstop.

## 4. Why this beats the existing "kids + AI" tools

Teachable Machine, Scratch-ML blocks, drag-a-CNN tools already exist. They share one weakness:
**nothing grades the result honestly** — you train a thing, it says "done", and the kid has
no idea if they did something real or trivial. The Academy's differentiator is the property
this whole project is built on:

> An **un-gameable referee** turns building into a *sport*. The feedback is true, instant,
> and competitive. That is the difference between a worksheet and Minecraft.

Plus: **the AI builds from the kid's spark.** Other tools make kids assemble everything by
hand (boring fast) or hide all the machinery (learn nothing). Here the kid supplies the
*idea and the judgement*, the AI supplies the *implementation*, and the referee supplies the
*truth*. That division of labour is exactly right for learning: you practise the part that
matters (deciding what to build and why) and watch the rest get done, verified, in front of
you.

## 5. Technical architecture (almost all of it already exists)

The Kernel Mint already is rung 0–1 end to end. The platform is that, repeated per rung:

- **Block grammar → spec.** Each rung defines a small set of blocks that compose *both* a
  task description (for the AI) *and* a reference/scorer (for the referee). This is exactly
  what `specs.py` + `chains.py` do for kernels: the blocks ARE the spec, so anything a kid
  assembles is automatically verifiable. Generalise `get_spec("chain|…")` to
  `get_spec("layer|…")`, `get_spec("solver|…")`.
- **An AI proposer per rung.** A small open model fine-tuned to emit the artefact for that
  rung (a kernel, a layer, a config). The 1B-writes-verified-kernels result proves a *tiny*
  model is enough when a referee grounds it — which keeps hosting cheap and "build small".
- **The immutable referee.** `harness.py`, generalised: compile/instantiate → check against a
  reference on adversarial inputs → measure → reject cheats. The anti-gaming negative
  controls matter *more* here, not less: kids will (gleefully) try to cheat, and the referee
  has to win every time, visibly.
- **Serving:** Modal GPU backend, scale-to-zero (idle = free); Gradio/HF-Space front-end. The
  cheap-1B-on-a-small-GPU pattern from the Mint is the unit economics.
- **State:** per-challenge leaderboards (a volume/dataset), seeded with big-model champions;
  every verified build feeds a public corpus — the demo literally generates its own
  teaching data (the flywheel from `FRAMEWORK_VISION.md`).

## 6. The honest caveats (write these on the wall)

1. **Referee softness rises with rung height.** Rungs 0–2: perfect/strong, ship freely.
   Rungs 3–5: held-out metrics that can be overfit. Mitigate with sealed test sets, rotated
   per session, and "report the loss regions" honesty — but accept that high rungs are
   Kaggle, not bit-exact truth. Don't pretend otherwise to the kid.
2. **Compute climbs fast.** A kernel verifies in seconds; a trained tiny model is minutes.
   Keep the *default* experience on the cheap rungs; gate the expensive ones behind explicit
   "this one takes a few minutes" framing (the Mint already does this for cold starts).
3. **Safety / moderation.** Free-text "spark" goes into a model prompt. Needs the usual input
   filtering for an all-ages product.
4. **The model fails honestly, and that's a feature — until it's frustrating.** On novel
   combos the small model often can't deliver (we saw this with 2-activation "remixes").
   Frame failures as challenges ("can you find one it can build?"), but tune difficulty so
   most *intended* builds succeed — a kid who fails ten times in a row leaves.

## 7. Roadmap

- **MVP (have it):** Kernel Mint = rungs 0–1, the loop proven, cheap to host, kid-friendly UI.
- **v1 — one rung up:** "Build a Layer" — snap an attention head / MLP block; referee checks
  it matches a reference layer on random inputs. Same architecture, new block grammar. Still
  a perfect referee.
- **v2 — the curriculum:** levels, progression, badges, champion leaderboards per level.
- **v3 — the soft rungs, carefully:** "Train a tiny classifier" on a sealed dataset, framed
  honestly as a different (Kaggle-like) kind of challenge.
- **Throughout:** every verified build feeds the public corpus; the platform trains the next,
  better proposer on its own users' verified creations.

## 8. The one-paragraph pitch

*Kids don't learn how AI works by reading about it; they learn by building the pieces it's
made of and getting told, honestly and instantly, whether the piece is real. The Verified
Playground is a block-based world where you snap together the real components of an AI system
— a GPU kernel, an attention head, a tiny model — add your own idea in plain words, and a
small AI builds it while an un-gameable referee grades it against reality. It's Scratch with
a conscience: you can build anything, because the truth, not a teacher and not the AI's own
opinion, decides if it works. The scarce, valuable thing was never the giant model. It's the
honest referee — and that's exactly what we put in a child's hands.*

---

*Lineage: this is the OUROBOROS loop (`FRAMEWORK_VISION.md`) wearing an education hat, with
the Kernel Mint (`../kernel-mint`, `../ouroboros-mint`) as its working rung-0 prototype. The
falsification discipline (`KEY_FINDINGS.md`) is the same one that keeps the referee honest
here.*
