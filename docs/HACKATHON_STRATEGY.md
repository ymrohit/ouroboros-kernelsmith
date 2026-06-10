# Build Small Hackathon — strategy + demo plan

*(Reference doc, 2026-06-10. Source: https://huggingface.co/spaces/build-small-hackathon/field-guide
— rules pulled from its `details.md` / `faq.md`.)*

## The rules that matter

- **Models must be < 32B params** (each model individually; multiple models allowed). Our
  Qwen3.6-27B qualifies for general tracks.
- **Submission = Gradio Space on the hackathon HF org** (Docker OK if Gradio UI) +
  **mandatory demo video** + **social media post**, links + tag frontmatter in the Space
  README. Up to **10 ZeroGPU apps** per participant; **multiple submissions allowed**.
- **Tracks:** *Backyard AI* (practical daily-life) · *Thousand Token Wood* (creative/playful).
- **General prizes per track:** $4K / $2.5K / $1.5K / $1K + $2K community choice.
- **Sponsor prizes (STACKABLE):** Modal **$20K credits** (must use Modal) · OpenBMB **$10K**
  (MiniCPM core to the experience) · NVIDIA **2× RTX 5080** (Nemotron) · OpenAI $10K
  (Codex-attributed commits — see "not chasing").
- **Bonus awards:** Tiny Titan **≤4B** ($1.5K) · Best Demo ($1K) · Best Agent ($1K) ·
  Off-brand UI ($1.5K) · Bonus Quest ($2K) · Judges' Wildcard ($1K).

## The strategic insight

**Our pipeline is model-agnostic (proven by the 2B→27B swap), and the sponsor prizes
stack.** Every "use sponsor's model" prize = one detached `train_all` run on existing,
fully-automated infra. And those runs double as the **multi-model scale study the paper
needs** (2B / MiniCPM-4B / Nemotron / 27B on an identical verifier — nobody has that table).

| prize | play | effort |
|---|---|---|
| Modal $20K | already maximal: trained + verified + served on Modal | 0 |
| OpenBMB $10K **+** Tiny Titan $1.5K | **one MiniCPM-4B pipeline run** — sponsor model AND ≤4B; "4B writes verified kernels" is the purest build-small story | 1 train_all |
| NVIDIA 2×5080 | same play with Nemotron-Nano | 1 train_all |
| Best Demo / Best Agent | the Mint + a "watch it learn" live-RL-round tab | small |

## Submissions (two Spaces)

### Space 1 — "OUROBOROS Kernel Mint" (Backyard AI, flagship)
Visitor composes an op from grammar dropdowns (`±residual → {rms|layer}norm → epilogue`,
incl. never-trained combos) → **small model writes a Triton kernel live** → immutable
harness verifies + benches → if it beats the compiler it's **minted on a leaderboard with
their name**. Failed mints say REJECTED, plainly (on-brand).
- **Two engines:** Tiny Titan (MiniCPM-4B / Qwen-2B) on **ZeroGPU inside the Space** for
  instant mints; "Pro Mode" button calls the **27B on Modal** for hard ops.
- Latency plan: ~60–90s/mint. Warm container; **precompute every MA baseline for the
  dropdown space the night before** (finite set); live bench vs compile-default (fast),
  MA numbers from cache.
- Why constrained dropdowns, not free-text ops: arbitrary ops have no trusted reference —
  verifying against user-written references is gameable. The grammar composes the
  reference automatically. (This constraint IS the project's thesis; say so in the UI.)

### Space 2 — "Cheat the Referee" (Thousand Token Wood, playful)
The anti-gaming harness as a game: *"Our judge cannot be fooled. Prove us wrong."*
One-click classic exploits (shape-cheat / memoizer / mutator) publicly REJECTED with their
specific error; free-form box to submit your own cheat or honest kernel vs the model's;
humans-vs-machine leaderboard + wall of rejected cheats. No LLM inference needed — cheap,
fast (~30s/eval), and it demos the actual product (the un-gameable verifier). A human
honestly beating the model's kernel is GOOD theater (referee certifies it → proves it's
not rigged).

### Best Agent add-on (cheap)
"Watch it learn" tab streaming a live RL round (propose → verify → reward → distill) —
frames the loop as an autonomous agent; just log streaming.

## Demo video — 3-minute flow

1. (20s) Hook: *"Sakana claimed 150× and got caught cheating their own benchmark in 48h.
   We built the referee that makes that impossible — then made a small model smart enough
   to satisfy it 69 times."*
2. (90s) Live mint: judge picks dropdowns → kernel generated → green tick + speedup →
   named + minted. (Pre-recorded fallback on a hotkey.)
3. (30s) One-click cheat → REJECTED.
4. (30s) Evidence wall: 69/69 stability, shape-grid heatmap WITH red loss cells visible,
   ablation table, QR to repo/HF. Close: *"The model is replaceable. The referee is the
   product."*

One-line identity: **"Small models + an un-cheatable referee out-engineer the compiler —
mint your own verified GPU kernel, live."**

## NOT chasing

- **OpenAI Codex $10K** — needs Codex-attributed commits; our history is honestly
  Claude-Code-built. Retrofitting = integrity smudge. Skip.
- Free-text op definitions (reference-trust problem, above).

## Build order

1. **MiniCPM-4B SFT→RL on Modal** (long pole; dual-prize unlock + paper scale point).
   Corpus exists; `--model` swap + check LoRA target names + valid-rate gate for 4B.
2. Mint backend: Modal endpoint wrapping generate→verify (~half day) + Gradio front (~half day).
3. Cheat Space (few hours — harness does the work).
4. Nemotron-Nano run (second sponsor unlock).
5. MA-baseline precompute for dropdown space; leaderboard persistence (HF dataset repo).
6. Demo video, social post, Space READMEs with required frontmatter, join hackathon org.

## Practicalities / risks

- ZeroGPU quotas: per-call durations are short; MA compile (1–3 min) must NOT be in the
  live path — precompute. 4B/2B inference fits ZeroGPU easily; 27B stays on Modal.
- User-submitted kernels: harness subprocess isolation + container sandbox; cap timeouts;
  no secrets in demo containers; serialized queue with visible position.
- Every live path gets a pre-recorded fallback. "REJECTED" is a designed outcome, not a
  failure mode (model valid-rate ~70–95% per family; sample k=8, take best).
- Each visitor mint = a new verified kernel for the corpus — the demo feeds the flywheel
  (closing line of the pitch).
