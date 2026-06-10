"""
OUROBOROS on Modal — train the kernel-smith on cloud GPUs.

Pipeline (each phase = its own GPU container; artifacts hand off via a Modal Volume):
    selftest  → prove the immutable harness on the cloud GPU (gold pass / wrong reject)
    sft       → teach Qwen3.5-2B to WRITE valid Triton (0% → ~100% valid), save LoRA adapter
    rl        → GRPO/self-distill on its own verified kernels (reward = measured speedup)
    rebench   → 5× stability re-bench vs torch.compile max-autotune

Quickstart (see README_MODAL.md):
    .venv-modal/bin/modal token new
    .venv-modal/bin/modal secret create huggingface HF_TOKEN=hf_xxx
    .venv-modal/bin/modal run modal_app.py::selftest          # validate GPU + harness
    .venv-modal/bin/modal run modal_app.py                    # full SFT → RL → rebench
    # or a single phase, e.g.:
    .venv-modal/bin/modal run modal_app.py::sft --epochs 30 --lora-rank 64
    .venv-modal/bin/modal run modal_app.py::rl --rounds 48 --group 4

Pull the trained adapter back:
    .venv-modal/bin/modal volume get ouroboros-outputs outputs/sft_adapter ./trained
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

# ----------------------------- config -----------------------------
APP_NAME = "ouroboros-kernelsmith"
GPU = os.environ.get("OURO_GPU", "H200")          # H100 | H200 | A100-80GB | A100 | L40S | A10G | L4
CODE_DIR = Path(__file__).parent / "ouroboros"     # the copied training stack
TIMEOUT = 12 * 60 * 60                             # 12h per phase (the wide run is long)
# Target model. Gemma-4-12B is GATED on HF — accept its license with your account and provide a
# token (HF_TOKEN). Confirm the exact repo id; override with OURO_MODEL=<repo> if it differs.
DEFAULT_MODEL = os.environ.get("OURO_MODEL", "Qwen/Qwen3.6-27B")
# Private HF repos to checkpoint into (incrementally): corpus → SFT adapter → RL final.
HF_USER = os.environ.get("OURO_HF_USER", "YMRohit")
MODEL_REPO = os.environ.get("OURO_MODEL_REPO", f"{HF_USER}/ouroboros-kernelsmith-qwen3.6-27b")
CORPUS_REPO = os.environ.get("OURO_CORPUS_REPO", f"{HF_USER}/ouroboros-kernel-corpus")

# The full environment (the real "dataset" — width is the discovery lever, RESULTS.md).
# 31 ops the teacher corpus covers (rmsnorm_wide excluded — seed-only, no teacher structures).
ALL_OPS = ("add_layernorm,add_layernorm_gelu,add_layernorm_relu2,add_layernorm_silu,"
           "add_rmsnorm,add_rmsnorm_gelu,add_rmsnorm_relu2,add_rmsnorm_rope,add_rmsnorm_silu,"
           "bias_gelu,dequant_int8,geglu,gelu,l2norm,layernorm,layernorm_gelu,layernorm_relu2,"
           "layernorm_silu,log_softmax,qknorm_rope,reglu,relu2,rmsnorm,rmsnorm_gelu,rmsnorm_relu2,"
           "rmsnorm_silu,rope,silu,softmax,softmax_scale,swiglu")
# Fusion-/compound-heavy subset for RL — where real scheduling slack (= discoveries) lives.
RL_OPS = ("add_rmsnorm,add_rmsnorm_gelu,add_rmsnorm_silu,add_rmsnorm_rope,rmsnorm_gelu,rmsnorm_silu,"
          "add_layernorm,add_layernorm_gelu,add_layernorm_silu,layernorm_gelu,geglu,reglu,swiglu,"
          "qknorm_rope,softmax,softmax_scale")

# The 16 NEW fused ops added to the chains.py grammar (tanh/sigmoid/relu/square epilogues ×
# {rms,layer}norm × {±residual}) — verified gold-pass/wrong-reject by verify_chains. Discovery target.
DISCOVERY_OPS = ("rmsnorm_tanh,rmsnorm_sigmoid,rmsnorm_relu,rmsnorm_square,"
                 "add_rmsnorm_tanh,add_rmsnorm_sigmoid,add_rmsnorm_relu,add_rmsnorm_square,"
                 "layernorm_tanh,layernorm_sigmoid,layernorm_relu,layernorm_square,"
                 "add_layernorm_tanh,add_layernorm_sigmoid,add_layernorm_relu,add_layernorm_square")

# V2 EXPANSION — 32 more chain ops (8 new epilogues × {rms,layer}norm × ±residual) and 5 new
# STANDALONE ops (each with its own gold seed + negative control wired into the selftest).
# Every one gold-passes the hardened harness and its negative control is rejected (verified
# on the 4090 before being listed here).
DISCOVERY_OPS_V2 = ",".join(
    f"{pre}{norm}_{act}"
    for act in ("leaky_relu", "relu6", "hardtanh", "elu", "selu", "softplus", "mish", "gelu_erf")
    for norm in ("rmsnorm", "layernorm")
    for pre in ("", "add_"))
STANDALONE_OPS_V2 = "softcap_softmax,rmsnorm_gemma,glu,rope_interleaved,cross_entropy"

# EVERYTHING: the FULL expanded suite — SFT over all original ops + 16 (v1 chains) + 32 (v2
# chains) + 5 standalone (84 total); RL/discovery over every fusion op old+new (69).
FULL_SFT_OPS = ALL_OPS + "," + DISCOVERY_OPS + "," + DISCOVERY_OPS_V2 + "," + STANDALONE_OPS_V2
FULL_RL_OPS  = RL_OPS  + "," + DISCOVERY_OPS + "," + DISCOVERY_OPS_V2 + "," + STANDALONE_OPS_V2

# Qwen3.6's linear-attention fast path needs:
#   is_fast_path_available = all((causal_conv1d_fn, causal_conv1d_update,
#                                 chunk_gated_delta_rule, fused_recurrent_gated_delta_rule))
#   - gated_delta_rule fns ← `fla` (flash-linear-attention >= 0.2.2, Triton — installs clean)
#   - causal_conv1d fns    ← `causal_conv1d` — PREBUILT wheels exist only up to torch 2.10
#     (no torch-2.12 wheel → that compile attempt was doomed). So we pin torch 2.10 → the
#     `cu12torch2.10` wheel installs with NO compiler needed (back on debian_slim).
image = (
    # CUDA-12.8 DEVEL base (matches torch 2.10's cu128): provides nvcc/CUDA_HOME so `tilelang` can
    # compile the Hopper Gated-DeltaNet backward kernel (debian_slim has only torch's CUDA runtime,
    # not the toolkit → tilelang's determine_target fails with "No CUDA available"). CUDA 12.8 uses
    # g++ (the CUDA-13 base needed clang and broke); causal_conv1d still uses its prebuilt wheel.
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.10.0",          # has a prebuilt causal_conv1d wheel (2.12 does not); brings triton
        "transformers==5.10.2", "peft==0.19.1",
        "accelerate==1.13.0", "datasets==4.8.5", "safetensors==0.7.0",
        "huggingface_hub>=1.5,<2", "hf_xet",
        "flash-linear-attention>=0.2.2",   # fla: chunk_/fused_recurrent_gated_delta_rule
        "tilelang",               # CORRECT gated-DeltaNet backward on Hopper (triton>=3.4 is buggy there)
        "packaging", "wheel", "ninja",
        "sentencepiece", "protobuf", "einops", "numpy",
    )
    # Install the EXACT prebuilt cu12torch2.10 wheel by URL — bypasses causal_conv1d's setup.py,
    # which needs nvcc to detect CUDA (absent on debian_slim → 'bare_metal_version' NameError).
    .run_commands(
        "pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/"
        "v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    # head-to-head baselines: liger-kernel (pip) + git for shallow-cloning unsloth/triton at runtime.
    .apt_install("git")
    .run_commands("pip install --no-deps liger-kernel")
    .add_local_dir(str(CODE_DIR), remote_path="/root/ouroboros")
)

app = modal.App(APP_NAME, image=image)

outputs = modal.Volume.from_name("ouroboros-outputs", create_if_missing=True)   # adapters + reports
hf_cache = modal.Volume.from_name("ouroboros-hf-cache", create_if_missing=True)  # model weights cache
VOL, CACHE = "/outputs", "/cache"
WORK = "/root/ouroboros"

# HF token (some Qwen/Gemma repos are gated). Inline secret backed by your local env, so probes
# run with no setup. For gated training:  export HF_TOKEN=hf_xxx   before `modal run`.
# (Or swap this for a named secret: modal.Secret.from_name("huggingface").)
hf_secret = modal.Secret.from_name("huggingface")   # created via `modal secret create huggingface HF_TOKEN=...`

COMMON = dict(
    gpu=GPU,
    volumes={VOL: outputs, CACHE: hf_cache},
    secrets=[hf_secret],
    timeout=TIMEOUT,
)


# ----------------------------- helpers -----------------------------
def _env() -> dict:
    return {
        **os.environ,
        "HF_HOME": CACHE,
        "HF_XET_HIGH_PERFORMANCE": "1",  # fast Xet transfer (hf_transfer is deprecated)
        "HF_HUB_OFFLINE": "0",          # allow first-time / gated download to the cache volume
        "TRANSFORMERS_OFFLINE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    }


def _prep_dirs():
    for d in ("outputs", "reports", "datasets"):
        os.makedirs(f"{WORK}/{d}", exist_ok=True)


def _run(cmd: list[str]):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=WORK, env=_env(), check=True)


def _save():
    """Persist produced artifacts to the volume."""
    for d in ("outputs", "reports", "datasets"):
        src = f"{WORK}/{d}"
        if os.path.isdir(src):
            shutil.copytree(src, f"{VOL}/{d}", dirs_exist_ok=True)
    outputs.commit()
    print(f"[saved] artifacts committed to volume 'ouroboros-outputs'", flush=True)


def _restore():
    """Bring prior artifacts (e.g. the SFT adapter) back into the working tree."""
    for d in ("outputs", "reports", "datasets"):
        src = f"{VOL}/{d}"
        if os.path.isdir(src):
            shutil.copytree(src, f"{WORK}/{d}", dirs_exist_ok=True)


def _gpu_banner():
    import torch
    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'} "
          f"torch={torch.__version__}", flush=True)


def _push_hf(local_dir: str, repo_id: str, repo_type: str, label: str):
    """Push an artifact dir to a PRIVATE HF repo. Never crashes the pipeline — a push failure
    is logged; the artifact is still safe in the Modal volume."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        print(f"[hf] no HF_TOKEN — skip pushing {label}", flush=True); return
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        print(f"[hf] nothing to push for {label} ({local_dir} empty)", flush=True); return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type=repo_type,
                          commit_message=f"OUROBOROS: {label}",
                          ignore_patterns=["**/__pycache__/**", "*.lock", "*.tmp"])
        pre = "datasets/" if repo_type == "dataset" else ""
        print(f"[hf] pushed {label} -> https://huggingface.co/{pre}{repo_id}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf] push {label} FAILED ({type(e).__name__}: {e}) — artifact remains in the volume", flush=True)


# ----------------------------- gpu probe -----------------------------
@app.function(gpu=GPU, timeout=300)
def gpuinfo():
    """Confirm live capacity for the requested GPU type. `OURO_GPU=H100 modal run modal_app.py::gpuinfo`."""
    import subprocess as sp
    import torch
    print(f"requested GPU type: {GPU}")
    print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
    try:
        print(sp.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception as e:
        print("nvidia-smi:", e)


# ----------------------------- fast-path verification -----------------------------
@app.function(gpu="H100", timeout=900)
def verify_fastpath():
    """PROVE the Qwen3.6 linear-attention path works ON HOPPER (sm_90 — the failing GPU) before any
    full run: is_fast_path_available True, tilelang finds CUDA, AND the actual GDN fwd+bwd runs."""
    import torch
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    from transformers.utils.import_utils import (
        is_flash_linear_attention_available, is_causal_conv1d_available)
    print("is_flash_linear_attention_available:", is_flash_linear_attention_available(), flush=True)
    print("is_causal_conv1d_available       :", is_causal_conv1d_available(), flush=True)
    try:
        import fla; print("fla version:", getattr(fla, "__version__", "?"), flush=True)
    except Exception as e:
        print("fla import FAILED:", repr(e), flush=True)
    try:
        import causal_conv1d as cc; print("causal_conv1d version:", getattr(cc, "__version__", "?"), flush=True)
    except Exception as e:
        print("causal_conv1d import FAILED:", repr(e), flush=True)
    # triton is load-bearing for the OUROBOROS harness — prove it imports AND compiles+runs a kernel.
    try:
        import triton, triton.language as tl
        print("triton version:", triton.__version__, flush=True)

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
            tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

        a = torch.randn(1024, device="cuda"); b = torch.randn(1024, device="cuda"); o = torch.empty_like(a)
        _add[(1,)](a, b, o, 1024, BLOCK=1024)
        print("triton kernel compiled+ran OK:", bool(torch.allclose(o, a + b)), flush=True)
    except Exception as e:
        print("triton FAILED:", repr(e), flush=True)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    ok = bool(M.is_fast_path_available)
    print(f">>> is_fast_path_available = {ok}  {'(FAST ✓)' if ok else '(STILL FALLBACK ✗)'}", flush=True)
    # tilelang must find CUDA (the prior crash was here on Hopper) ...
    try:
        import tilelang
        from tilelang.utils.target import determine_target
        print(f"tilelang {getattr(tilelang,'__version__','?')} determine_target:",
              determine_target(return_object=True), flush=True)
    except Exception as e:
        print("tilelang determine_target FAILED:", repr(e), flush=True)
    # ... and the actual Gated-DeltaNet fwd+BACKWARD must run on Hopper (the exact crashing path).
    bwd_ok = False
    try:
        from fla.layers import GatedDeltaNet
        layer = GatedDeltaNet(hidden_size=256, num_heads=4, mode="chunk").to("cuda").to(torch.bfloat16)
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = layer(x); y = out[0] if isinstance(out, (tuple, list)) else out
        y.sum().backward()
        g = sum(p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None)
        bwd_ok = g > 0
        print(f">>> GatedDeltaNet fwd+BACKWARD on Hopper: OK (grad sum={g:.2f}) ✓", flush=True)
    except Exception as e:
        print(f">>> GatedDeltaNet fwd+BACKWARD FAILED: {type(e).__name__}: {e}", flush=True)
    print(f">>> READY_TO_TRAIN = {ok and bwd_ok}", flush=True)
    return ok and bwd_ok


# ----------------------------- phases -----------------------------
@app.function(**COMMON)
def selftest():
    """Prove the immutable referee on the cloud GPU: gold kernels pass, wrong ones are REJECTED."""
    _gpu_banner(); _prep_dirs()
    _run([sys.executable, "harness.py"])
    _save()


@app.function(**COMMON)
def sft(
    epochs: int = 30,
    lora_rank: int = 128,
    gate: float = 0.8,
    ops: str = ALL_OPS,
    eval_temps: str = "0.3,0.5,0.8",
    eval_k: int = 8,
    batch: int = 8,
    accum: int = 2,
    model: str = DEFAULT_MODEL,
    load_adapter: str = "",
    out: str = "outputs/sft_adapter",
    skip_baseline: bool = False,
):
    """PHASE 1 — SFT a cold model from 0% → ~100% valid Triton. Saves to `out`.
    If load_adapter is set, CONTINUE-SFT (trainable) from it — teach new ops without losing old."""
    _gpu_banner(); _restore(); _prep_dirs()
    cmd = [sys.executable, "-u", "sft_train.py",
           "--model", model, "--ops", ops, "--epochs", str(epochs),
           "--lora-rank", str(lora_rank), "--gate", str(gate),
           "--eval-temps", eval_temps, "--eval-k", str(eval_k),
           "--batch", str(batch), "--accum", str(accum), "--out", out]
    if load_adapter:
        cmd += ["--load-adapter", load_adapter]
    if skip_baseline:
        cmd += ["--skip-baseline"]
    _run(cmd)
    _save()
    # HARD GUARD: the SFT adapter MUST be on disk before we claim success / push.
    ckpt = f"{WORK}/{out}/adapter_model.safetensors"
    if not os.path.exists(ckpt):
        raise RuntimeError(f"SFT adapter was NOT saved ({out} missing) — aborting.")
    print(f"[sft] verified SFT adapter on disk: {ckpt}", flush=True)
    _push_hf(f"{WORK}/outputs", MODEL_REPO, "model", f"SFT adapter ({out})")
    _push_hf(f"{WORK}/datasets", CORPUS_REPO, "dataset", "verified corpus (full grammar)")
    print("[sft] SFT adapter + corpus pushed ✓", flush=True)


@app.function(**COMMON)
def rl(
    rounds: int = 48,
    group: int = 8,
    ops: str = RL_OPS,
    adapter: str = "outputs/sft_adapter",
    explore_frac: float = 0.0,
    max_new: int = 1024,
    model: str = DEFAULT_MODEL,
    save_adapter: str = "outputs/rl_adapter",
    out: str = "reports/kernelsmith_rl.json",
):
    """PHASE 2 — RL/self-distill on the SFT'd model. explore_frac=0.0 → the model writes every kernel.
    SAVES the RL adapter to `save_adapter`, hard-verifies it wrote, then pushes everything."""
    _gpu_banner(); _restore(); _prep_dirs()
    if not os.path.isdir(f"{WORK}/{adapter}"):
        raise RuntimeError(f"adapter '{adapter}' not found in the volume — run `sft` first.")
    _run([sys.executable, "-u", "rl_kernelsmith.py",
          "--model", model, "--ops", ops, "--rounds", str(rounds), "--group", str(group),
          "--no-kl", "--explore-frac", str(explore_frac), "--max-new", str(max_new),
          "--load-adapter", adapter, "--save-adapter", save_adapter, "--out", out])
    _save()
    # HARD GUARD: do not claim success / push a misleading "done" if the RL adapter didn't save.
    rl_ckpt = f"{WORK}/{save_adapter}/adapter_model.safetensors"
    if not os.path.exists(rl_ckpt):
        raise RuntimeError(f"RL adapter was NOT saved ({save_adapter} missing) — aborting.")
    print(f"[rl] verified RL adapter on disk: {rl_ckpt}", flush=True)
    _push_hf(f"{WORK}/outputs", MODEL_REPO, "model", f"RL adapter ({save_adapter}) + best kernels")
    _push_hf(f"{WORK}/reports", MODEL_REPO, "model", "RL reports")
    print("[rl] RL adapter + kernels + reports pushed ✓", flush=True)


@app.function(**COMMON)
def rebench():
    """5× stability re-bench vs torch.compile max-autotune (the honest 'beats the incumbent' bar)."""
    _gpu_banner(); _restore(); _prep_dirs()
    _run([sys.executable, "-u", "rebench_stability.py"])
    _save()


@app.function(**COMMON)
def rebench_shapes(ops: str = "", dtypes: str = "fp16,bf16", n_iters: int = 50,
                   rotate: bool = True, kernels: str = "outputs/best_kernels"):
    """V2 SHAPE-GRID re-bench: every best kernel across a (M,N)×dtype grid vs max-autotune,
    geomean + win-rate + explicit loss regions, plus the rotating-buffer (cache-cold) check
    at the headline shape. Turns 'wins at one shape' into 'wins across the regime' (or not —
    losses are reported plainly)."""
    _gpu_banner(); _restore(); _prep_dirs()
    cmd = [sys.executable, "-u", "rebench_shapes.py", "--kernels", kernels,
           "--dtypes", dtypes, "--n-iters", str(n_iters)]
    if ops:
        cmd += ["--ops", ops]
    if rotate:
        cmd += ["--rotate"]
    _run(cmd)
    _save()
    _push_hf(f"{WORK}/reports", MODEL_REPO, "model", "shape-grid rebench")


@app.function(gpu="L4", timeout=1200)
def verify_chains():
    """DOCTRINE GATE for the EXPANDED fusion grammar (chains.py): every NEW chain op's template
    gold-passes the harness (allclose vs the COMPOSED reference, adversarial inputs) AND a wrong
    kernel is rejected — proven BEFORE any model is pointed at the new ops. No op joins the loop
    until its negative control fails."""
    import sys as _s; _s.path.insert(0, WORK); os.chdir(WORK)
    import specs                       # importing specs runs _register_chains()
    import chains
    from harness import evaluate
    NEW = {"tanh", "sigmoid", "relu", "square", "abs", "softsign", "hardsigmoid", "hardswish",
           "relu6", "hardtanh", "elu", "selu", "softplus", "mish"}
    new = [(n, srcs) for (n, _k, _ref, srcs) in chains.all_chains()
           if n.rsplit("_", 1)[-1] in NEW or n.endswith(("leaky_relu", "gelu_erf"))]
    print(f"[chains] verifying {len(new)} NEW fused ops (gold-pass, both template variants) ...", flush=True)
    passed = 0
    for name, srcs in new:
        verdicts = []
        for src in srcs:              # scalar-reduce + whole-row templates
            r = evaluate(src, name, correctness_only=True)
            verdicts.append(r.status == "ok" and r.correct)
        ok = any(verdicts)            # at least one gold template must verify
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'} {name:24} variants_ok={verdicts}", flush=True)
    # NEGATIVE CONTROLS: a correct kernel for epilogue A, evaluated under epilogue B's spec,
    # MUST be rejected — the cross-op rejection that proves the references discriminate.
    rejected = True
    for a, b in [("rmsnorm_tanh", "rmsnorm_sigmoid"), ("rmsnorm_mish", "rmsnorm_softplus"),
                 ("layernorm_elu", "layernorm_selu"), ("add_rmsnorm_relu6", "add_rmsnorm_hardtanh"),
                 ("rmsnorm_gelu_erf", "rmsnorm_gelu")]:
        src = next(s[0] for (n, s) in new if n == a)
        rj = evaluate(src, b, correctness_only=True)
        rj_ok = not (rj.status == "ok" and rj.correct)
        rejected &= rj_ok
        print(f"  WRONG-REJECT ({a} kernel ⇒ {b}): status={rj.status} "
              f"→ {'REJECTED ✓' if rj_ok else 'WRONGLY ACCEPTED ✗'}", flush=True)
    ok_all = (passed == len(new)) and rejected
    print(f">>> CHAINS_GRAMMAR_VERIFIED = {ok_all}  ({passed}/{len(new)} gold-pass, wrong-reject={rejected})",
          flush=True)
    return ok_all


@app.function(**COMMON)
def bench_experts():
    """HEAD-TO-HEAD vs human experts, TWO conditions through the SAME immutable harness
    (strong=True → vs torch.compile max-autotune):
      1. fixed_schedule — raw @triton.jit kernels extracted from Liger/Unsloth/the Triton
         tutorials with our wrappers (schedule pinned; no autotune laundering), and
      2. library_api  — Liger called through its OWN public Function API exactly as
         shipped (its dispatch, its settings). The condition that is FAIR to the library.
    Failures in condition 1 on odd shapes are wrapper-fidelity findings, NOT library bugs —
    condition 2 is the one that may make that claim. Both are reported side by side."""
    _gpu_banner(); _restore(); _prep_dirs()
    import subprocess
    ext = f"{WORK}/external"; os.makedirs(ext, exist_ok=True)
    for repo, url in [("unsloth", "https://github.com/unslothai/unsloth"),
                      ("triton", "https://github.com/triton-lang/triton")]:
        d = f"{ext}/{repo}"
        if not os.path.isdir(d):
            print(f"[h2h] cloning {repo} (shallow) ...", flush=True)
            subprocess.run(["git", "clone", "--depth", "1", url, d], check=False)
    sys.path.insert(0, WORK); os.chdir(WORK)
    import external_harvest, json
    print("[h2h] condition 1: fixed-schedule extraction ...", flush=True)
    manifest = external_harvest.harvest()          # extract + verify expert kernels vs OUR reference
    print("[h2h] condition 2: library public API (Liger as shipped) ...", flush=True)
    manifest_api = external_harvest.harvest_api()
    from harness import evaluate
    experts = {}
    for m in manifest:
        experts.setdefault(m["op"], []).append(
            ("fixed_schedule", m["provenance"], f"{WORK}/datasets/real_kernels/{m['file']}"))
    for m in manifest_api:
        if m.get("verified"):
            experts.setdefault(m["op"], []).append(
                ("library_api", m["provenance"], f"{WORK}/datasets/api_kernels/{m['file']}"))
    def _bench(src, op):
        v = evaluate(src, op, strong=True)
        return {"latency_ms": round(v.latency_ms, 4), "vs_maxauto": round(v.speedup_maxauto, 3),
                "vs_compile": round(v.speedup_compile, 3), "status": v.status}
    results = {"_api_unavailable": [m for m in manifest_api if not m.get("verified")]}
    for op in sorted(k for k in experts):
        row = {"experts": {}}
        model_p, seed_p = f"{WORK}/outputs/best_kernels/{op}.py", f"{WORK}/seed_kernels/{op}.py"
        if os.path.exists(model_p):
            row["ours"] = {"author": "MODEL", **_bench(open(model_p).read(), op)}
        elif os.path.exists(seed_p):
            row["ours"] = {"author": "gold_seed", **_bench(open(seed_p).read(), op)}
        for cond, prov, path in experts[op]:
            row["experts"][prov] = {"condition": cond, **_bench(open(path).read(), op)}
        results[op] = row
        print(f"[h2h] {op}: ours={row.get('ours')} | experts={row['experts']}", flush=True)
    json.dump(results, open(f"{WORK}/reports/headtohead_experts.json", "w"), indent=2)
    _save()
    _push_hf(f"{WORK}/reports", MODEL_REPO, "model", "head-to-head vs Liger/Unsloth/Triton experts (2 conditions)")
    print("[h2h] done + pushed ✓", flush=True)
    return results


@app.function(**COMMON)
def kernelbench_eval(level: int = 1, n_problems: int = 100, backend: str = "triton",
                     adapter: str = "outputs/rl_adapter_newops", max_new: int = 6144,
                     problem_ids: str = ""):
    """Run our trained model on KernelBench (the LLM-kernel leaderboard). HONEST specialist test: our
    model is a fusion-op specialist in OUR format; KernelBench is generalist (matmul/conv/full nets) —
    expect a low correctness rate that marks the specialist boundary. backend=triton, fp32.
    KernelBench installed --no-deps (path import) so it can't downgrade our torch 2.10 / triton 3.6."""
    _gpu_banner(); _restore(); _prep_dirs()
    import subprocess
    kb = "/root/kernelbench_repo"
    if not os.path.isdir(kb):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ScalingIntelligence/KernelBench", kb], check=True)
    # KernelBench helper deps (all torch-free → can't downgrade our torch 2.10 / triton 3.6)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydra_config", "python-dotenv",
                    "tomli", "tabulate", "litellm", "openai", "pydantic", "tqdm"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dill>=0.4.1"], check=False)  # multiprocess needs it
    sys.path.insert(0, f"{kb}/src"); sys.path.insert(0, WORK); os.chdir(WORK)  # so outputs/<adapter> resolves
    import torch, json
    from kernelbench.eval import eval_kernel_against_ref, fetch_ref_arch_from_level_problem_id
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend
    from kernelbench.utils import extract_first_code, set_gpu_arch
    set_gpu_arch(["Hopper"])                       # H200 = sm_90
    from rl_kernelsmith import Proposer
    # KernelBench is GENERALIST (matmul/conv) in self-contained ModelNew format. Our adapter is a
    # narrow norm/activation specialist trained to OMIT imports (our harness injects them) → wrong
    # tool here. adapter="" / "base" / "none" → BASE model (the fair entrant for the leaderboard).
    la = adapter if (adapter and adapter.lower() not in ("", "none", "base")) else None
    print(f"[KB] generator = {'BASE Qwen3.6-27B (no adapter)' if la is None else la}", flush=True)
    prop = Proposer(DEFAULT_MODEL, temp=0.0, kl=False, max_new=max_new, load_adapter=la)
    tok, model = prop.tok, prop.model
    _PREAMBLE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n\n"

    # PREFILL the assistant turn with a code fence so the model generates CODE from token 1 instead of
    # burning the budget on prose reasoning (the truncation root cause). Robust regardless of cap.
    _PREFILL = "```python\nimport torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"
    def _gen(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True) + _PREFILL
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id, use_cache=True)
        return _PREFILL + tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    pids = [int(x) for x in problem_ids.split(",") if x.strip()] or list(range(1, n_problems + 1))
    print(f"[KB] evaluating {len(pids)} problems: {pids}", flush=True)
    results = []; ncorrect = ncompiled = 0
    for i, pid in enumerate(pids, 1):
        rec = {"problem_id": pid}
        try:
            ref = fetch_ref_arch_from_level_problem_id(level, pid, with_name=False)
            prompt = get_prompt_for_backend(ref, backend=backend, option="one_shot")
            # The model rambles in prose and gets truncated before writing ModelNew. Force terse code-only.
            prompt += ("\n\nIMPORTANT: Respond with ONLY one ```python code block containing the COMPLETE, "
                       "runnable `class ModelNew(nn.Module)` plus its `@triton.jit` kernel(s) and imports. "
                       "No analysis, no planning, no explanation — output the code first and nothing else.")
            gen = _gen(prompt)
            code = extract_first_code(gen, ["python"]) or extract_first_code(gen, [""])
            if not code:
                import re as _re
                m = _re.search(r"```(?:python|cpp)?\s*\n(.*?)```", gen, _re.S)
                code = m.group(1) if m else (gen if ("class ModelNew" in gen or "def run(" in gen) else None)
            if code:
                if "import triton" not in code:                      # our kernels omit imports (harness injects)
                    code = _PREAMBLE + code
                if "class ModelNew" not in code and "def run(" in code:  # BRIDGE: our run() format -> KernelBench ModelNew
                    code += ("\n\nclass ModelNew(nn.Module):\n"
                             "    def __init__(self, *args, **kwargs):\n        super().__init__()\n"
                             "    def forward(self, *inputs):\n        return run(*inputs)\n")
            rec["gen_head"] = (code or gen)[:150]
            if not code or "class ModelNew" not in code:
                print(f"[KB DEBUG p{pid}] no ModelNew. FULL GEN ({len(gen)} chars):\n{gen[:1800]}\n===END GEN===", flush=True)
                rec.update({"compiled": False, "correct": False, "note": "no ModelNew/run() in output"})
            else:
                r = eval_kernel_against_ref(ref, code, measure_performance=True, backend=backend,
                                            precision=torch.float32, num_correct_trials=3, num_perf_trials=5)
                rec.update({"compiled": bool(r.compiled), "correct": bool(r.correctness),
                            "runtime_us": r.runtime})
                ncompiled += bool(r.compiled); ncorrect += bool(r.correctness)
        except Exception as e:
            rec.update({"compiled": False, "correct": False, "error": f"{type(e).__name__}: {str(e)[:160]}"})
        results.append(rec)
        print(f"[KB L{level} p{pid:3}] compiled={rec.get('compiled')} correct={rec.get('correct')} "
              f"rt={rec.get('runtime_us','-')} {rec.get('error', rec.get('note',''))} | head={rec.get('gen_head','')!r}",
              flush=True)
        if i % 10 == 0:   # checkpoint so a mid-run crash can't lose progress
            json.dump({"summary": {"done": i, "correct": ncorrect, "compiled": ncompiled}, "results": results},
                      open(f"{WORK}/reports/kernelbench_L{level}.json", "w"), indent=2)
            _save(); print(f"[KB] checkpoint @ {i}/{len(pids)}: correct {ncorrect}, compiled {ncompiled}", flush=True)
    summary = {"level": level, "n": len(pids), "problem_ids": pids, "backend": backend, "adapter": adapter,
               "compiled": ncompiled, "correct": ncorrect,
               "correctness_rate": round(ncorrect / max(1, len(pids)), 3)}
    print(f"\n[KB] L{level}: correct {ncorrect}/{len(pids)} ({summary['correctness_rate']:.0%}), "
          f"compiled {ncompiled}/{len(pids)}", flush=True)
    json.dump({"summary": summary, "results": results},
              open(f"{WORK}/reports/kernelbench_L{level}.json", "w"), indent=2)
    _save(); _push_hf(f"{WORK}/reports", MODEL_REPO, "model",
                      f"KernelBench L{level}: {ncorrect}/{n_problems} correct")
    return summary


# ----------------------------- one-shot pipeline (detach-safe) -----------------------------
@app.function(**COMMON)
def train_all(
    epochs: int = 30,
    lora_rank: int = 128,
    gate: float = 0.8,
    sft_ops: str = ALL_OPS,
    rl_ops: str = RL_OPS,
    rounds: int = 48,
    group: int = 8,           # RL candidates/round — generation is batched, so this uses the H100
    batch: int = 8,           # SFT micro-batch (sequences/forward) — sized for the H100
    accum: int = 2,           # grad-accum steps → effective SFT batch = batch*accum = 16
    model: str = DEFAULT_MODEL,
):
    """selftest → sft → rl → rebench, all in ONE GPU container so `modal run --detach` survives
    client disconnects. Commits artifacts to the volume after every phase."""
    _gpu_banner(); _restore(); _prep_dirs()

    print("\n===== [1/5] referee self-test =====", flush=True)
    _run([sys.executable, "harness.py"]); _save()

    print("\n===== [2/5] build verified corpus (then push privately) =====", flush=True)
    _run([sys.executable, "-u", "sft_train.py", "--model", model, "--ops", sft_ops, "--corpus-only"])
    _save(); _push_hf(f"{WORK}/datasets", CORPUS_REPO, "dataset", "verified corpus")

    print("\n===== [3/5] SFT (reuses cached corpus) → push SFT adapter =====", flush=True)
    _run([sys.executable, "-u", "sft_train.py", "--model", model, "--ops", sft_ops,
          "--epochs", str(epochs), "--lora-rank", str(lora_rank), "--gate", str(gate),
          "--batch", str(batch), "--accum", str(accum), "--skip-baseline"])
    _save(); _push_hf(f"{WORK}/outputs", MODEL_REPO, "model", "SFT adapter")

    print("\n===== [4/5] RL / self-distill (fusion-heavy ops) → push RL adapter =====", flush=True)
    _run([sys.executable, "-u", "rl_kernelsmith.py", "--model", model, "--ops", rl_ops,
          "--rounds", str(rounds), "--group", str(group), "--no-kl", "--explore-frac", "0.0",
          "--max-new", "1024", "--load-adapter", "outputs/sft_adapter",
          "--out", "reports/kernelsmith_rl.json"])
    _save(); _push_hf(f"{WORK}/outputs", MODEL_REPO, "model", "RL final adapter + best kernels")

    print("\n===== [5/5] 5x stability re-bench → push reports =====", flush=True)
    _run([sys.executable, "-u", "rebench_stability.py"]); _save()
    _push_hf(f"{WORK}/reports", MODEL_REPO, "model", "reports (selftest + rebench)")
    print("\n===== TRAINING COMPLETE — corpus + adapters + reports pushed to private HF =====", flush=True)


# ----------------------------- push to Hugging Face (private) -----------------------------
@app.function(volumes={VOL: outputs}, secrets=[hf_secret], timeout=3600)
def push_to_hf(repo_id: str, private: bool = True, base_model: str = DEFAULT_MODEL):
    """Upload the trained LoRA adapter(s) + reports to a PRIVATE HF repo. Needs a WRITE token:
    export HF_TOKEN=hf_write_xxx   before   modal run modal_app.py::push_to_hf --repo-id you/name"""
    import os as _os
    from huggingface_hub import HfApi
    tok = _os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("No HF_TOKEN. `export HF_TOKEN=hf_write_xxx` (write scope) before this call.")
    if not _os.path.isdir(f"{VOL}/outputs"):
        raise RuntimeError("No trained artifacts in the volume yet — run train_all first.")
    card = (
        "---\nlibrary_name: peft\nbase_model: " + base_model + "\ntags:\n- triton\n- gpu-kernels\n"
        "- ouroboros\n- lora\n- verifier-self-distillation\npipeline_tag: text-generation\n---\n\n"
        "# OUROBOROS kernel-smith (LoRA)\n\n"
        f"A LoRA fine-tune of `{base_model}` that writes Triton GPU kernels, trained by an immutable "
        "verifier (compile → allclose vs PyTorch → CUDA-event benchmark) via self-distillation.\n\n"
        "- **SFT**: learns to write valid Triton across the full op suite (cold 0% → ~100% valid).\n"
        "- **RL**: self-distills on its own verified, fastest kernels (reward = measured speedup).\n\n"
        "`outputs/` holds the adapter(s); `reports/` holds the verified run + 5× stability re-bench.\n"
        "Numbers are hardware-specific (trained on Modal " + GPU + "). The harness is the arbiter.\n"
    )
    with open(f"{VOL}/README.md", "w") as f:
        f.write(card)
    api = HfApi(token=tok)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=VOL, repo_id=repo_id, repo_type="model",
                      ignore_patterns=["**/__pycache__/**", "*.lock"])
    vis = "private" if private else "public"
    print(f"pushed → https://huggingface.co/{repo_id}  ({vis})", flush=True)


# ----------------------------- orchestrator -----------------------------
@app.local_entrypoint()
def main(epochs: int = 30, rounds: int = 48, group: int = 4, lora_rank: int = 128):
    """Full pipeline: selftest → sft → rl → rebench (each on its own GPU container)."""
    print("== OUROBOROS on Modal: selftest → sft → rl → rebench ==")
    selftest.remote()
    sft.remote(epochs=epochs, lora_rank=lora_rank)
    rl.remote(rounds=rounds, group=group)
    rebench.remote()
    print("\nDONE. Pull artifacts:\n"
          "  modal volume get ouroboros-outputs outputs/sft_adapter ./trained\n"
          "  modal volume get ouroboros-outputs reports ./reports")
