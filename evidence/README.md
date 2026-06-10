# evidence/ — raw run logs (what could be recovered, stated honestly)

The **authoritative evidence** for every claim is the harness-emitted JSON/markdown in
`reports/` (in git + pushed to HF). This directory additionally archives the **raw run
transcripts**, with an honest caveat: Modal's CLI windows old logs, and several early
launches were piped through `tail`, so some files here are partial windows rather than
complete transcripts. File names say which:

- `*_modal_window.log` / `*_window.log` / `*_TAIL_ONLY.log` — partial (CLI window or tail).
- `selftest_*`, `verify_chains_*`, `bench_experts_*`, `e2e_*` — complete transcripts.

**Fixed going forward (V2.7):** `modal_app._run` now tees the complete stdout+stderr of
every phase to `reports/logs/<timestamp>_<script>.log` inside the container, which persists
to the Modal volume and the HF push automatically. Runs launched after this commit have
full transcripts by construction; the five in-flight runs (4 ablation arms + invention)
predate it, so their durable evidence is their harness-emitted JSON reports plus whatever
window the CLI retains.

The v1-era training runs (original SFT and the first RL) predate local capture entirely —
their durable evidence is the JSON reports + the recorded artifacts (adapters, kernels,
corpus) on the volume/HF, not transcripts. Stated here rather than papered over.
