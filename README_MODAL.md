# OUROBOROS on Modal — cloud training

Train the kernel-smith (Qwen3.5-2B writes Triton kernels, an immutable harness verifies them)
on Modal GPUs. This folder is **self-contained**: the training stack under `ouroboros/` was
copied read-only from `domains/ouroboros/` and is never edited there.

```
ouroboros-modal/
├── modal_app.py          # the Modal app: selftest → sft → rl → rebench
├── requirements.txt      # local driver only (modal)
├── .venv-modal/          # local venv with modal installed (gitignored)
└── ouroboros/            # the training stack (copied read-only)
    ├── harness.py        # immutable referee: compile → allclose → CUDA-event bench
    ├── specs.py  canonicalize.py  chains.py  teacher_kernels.py
    ├── sft_train.py      # PHASE 1 — SFT to ~100% valid Triton
    ├── rl_kernelsmith.py # PHASE 2 — RL / self-distill (reward = measured speedup)
    ├── rebench_stability.py
    ├── seed_kernels/     # gold kernels + deliberately-wrong negative controls
    └── CLAUDE.md RESULTS.md   # reference (the doctrine + the recorded run)
```

## 1. One-time setup

```bash
# from this folder; the venv already has modal installed
.venv-modal/bin/modal token new                       # auth to your Modal workspace
export HF_TOKEN=hf_xxxxxxxx                            # only needed for gated model downloads
```

The HF token is forwarded to the container as an inline secret from your local `HF_TOKEN`
(no named Modal secret required). Probes (`gpuinfo`, `selftest`) run with no token at all.
For public models you can skip it entirely.

## 2. Validate the GPU + the referee (cheap, ~2–3 min)

```bash
.venv-modal/bin/modal run modal_app.py::selftest
```
Spins up a GPU container, prints the device, and runs `harness.py` — gold kernels must pass and
the deliberately-wrong negative controls must be **REJECTED**. If this is green, the referee is
sound on the cloud GPU.

## 3. Train

Full pipeline (each phase on its own GPU container, artifacts handed off via a Modal Volume):
```bash
.venv-modal/bin/modal run modal_app.py
# or tune: .venv-modal/bin/modal run modal_app.py --epochs 30 --rounds 48 --group 4
```

Or run phases individually:
```bash
.venv-modal/bin/modal run modal_app.py::sft  --epochs 30 --lora-rank 64
.venv-modal/bin/modal run modal_app.py::rl   --rounds 48 --group 4
.venv-modal/bin/modal run modal_app.py::rebench
```

## 4. Get the trained model back

```bash
.venv-modal/bin/modal volume get ouroboros-outputs outputs/sft_adapter ./trained
.venv-modal/bin/modal volume get ouroboros-outputs reports ./reports
.venv-modal/bin/modal volume ls ouroboros-outputs        # browse what's stored
```

## Config

- **GPU**: defaults to `A100`. Override per run: `OURO_GPU=H100 .venv-modal/bin/modal run modal_app.py::rl`
  (valid: `A100`, `A100-80GB`, `H100`, `L40S`, `A10G`, `L4`). 2B + LoRA + benchmarking fits a
  24 GB card (`A10G`/`L4`); `A100` is the safe default.
- **Volumes**: `ouroboros-outputs` (adapters + reports), `ouroboros-hf-cache` (model weights —
  persists across runs so you only download Qwen once).
- **Pinned image** (`modal_app.py`): torch 2.12 · triton 3.7 · transformers 5.8.1 · peft 0.19.1 ·
  accelerate 1.13 · datasets 4.8.5 — the versions the run was developed against.

### Troubleshooting
- **Triton can't find `ptxas`**: swap the image base in `modal_app.py` for
  `modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")`.
- **Gated model / 401**: ensure the `huggingface` secret holds a token with access to `Qwen/Qwen3.5-2B`.
- **OOM**: the RL phase already passes `--no-kl`; drop `--group` to 4 or use a bigger GPU (`A100-80GB`).

> The numbers in `RESULTS.md` were measured on an RTX 4090. On a different Modal GPU the *speedup
> ratios will differ* (different memory bandwidth / SM count), but the loop, the correctness
> verification, and the "beats max-autotune" methodology are identical and re-run honestly.
