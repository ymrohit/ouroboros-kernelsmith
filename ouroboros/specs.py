"""OUROBOROS op suite — the SPECS the proposer writes kernels for.

Mirror of `sec_sqli/discovery_specialist`'s environment definition (`OBJ`/`PARAM`/
`ENVS`), but the "environment" here is a fusion op whose ground truth is a PyTorch
reference and whose grounded reward is a measured wall-clock speedup.

Each `OpSpec` is the immutable problem statement for one op:
  - `reference(*inputs) -> Tensor`  : the PyTorch ground truth (correctness oracle).
  - `make_inputs(rng) -> tuple`     : ADVERSARIAL randomized inputs — shape, dtype AND
                                      magnitude vary, so a kernel that is only correct on
                                      benign N(0,1) inputs (e.g. softmax with no
                                      max-subtraction) FAILS. This is the GPU analog of
                                      dvwa_oracle's benign-baseline / anti-pattern-match
                                      negative controls.
  - `tol(dtype)`                    : rtol/atol DERIVED from fp accumulation, never
                                      hand-tuned to make a kernel pass.
  - `signature_hint`                : goes into the proposer prompt.
  - The honest baselines (eager + torch.compile) are built by the harness, not stored
    here, so the same `reference` defines both ground truth and the bar.

Scope is deliberately HARD-narrowed to FUSION wins (ops where eager launches several
kernels and Triton can fuse them). We do NOT try to beat cuBLAS at dense GEMM — that is
the losing-the-window trap called out in the brief.

A candidate KERNEL is a Python source string that, when exec'd in a namespace already
holding {torch, triton, tl}, defines a callable `run(*inputs) -> Tensor` matching the
op's reference signature. The harness is the only thing that ever runs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

# ----------------------------------------------------------------------------------------
# Tolerances DERIVED from dtype accumulation behaviour (not tuned to pass).
# A reduction over D elements accumulates ~sqrt(D) * eps_machine relative error; we give a
# comfortable but principled envelope per storage dtype. Internals always accumulate in fp32.
# ----------------------------------------------------------------------------------------
_TOL = {
    torch.float32: (2e-4, 2e-5),
    torch.float16: (2e-2, 2e-3),
    torch.bfloat16: (3e-2, 8e-3),
}


def tol_for(dtype: torch.dtype) -> tuple[float, float]:
    return _TOL.get(dtype, (2e-2, 2e-3))


# ----------------------------------------------------------------------------------------
# Randomized regimes. `make_inputs` must sweep these so correctness is ADVERSARIAL: a
# kernel has to survive large magnitudes and low-precision dtypes, not just the easy case.
# ----------------------------------------------------------------------------------------
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_SCALES = [1.0, 8.0, 64.0]          # magnitude sweep — exposes overflow / missing max-sub
_ROWLEN = [128, 512, 1024, 4096, 4097, 8192]   # incl. non-power-of-2 (masking correctness)
_NROWS = [8, 64, 1000, 4096]


def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def _randn(rng, shape, dtype, scale, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(rng.randrange(2**31))
    return (torch.randn(shape, generator=g, device=dev, dtype=torch.float32) * scale).to(dtype)


# ----------------------------------------------------------------------------------------
# References (ground truth). Each upcasts to fp32 for the numerically-sensitive reduction
# exactly as a correct kernel must — so allclose tests the FUSION, not a dtype mismatch.
# ----------------------------------------------------------------------------------------
def _rmsnorm_ref(x, w, eps: float = 1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms).to(x.dtype) * w


def _softmax_ref(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _swiglu_ref(gate, up):
    # the FFN activation fusion: SiLU(gate) * up  (elementwise, multi-launch in eager)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)


def _add_rmsnorm_ref(x, residual, w, eps: float = 1e-6):
    h = (x.float() + residual.float())
    rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * rms).to(x.dtype) * w


def _rope_ref(x, cos, sin):
    # LLaMA "rotate_half" RoPE:  out = x*cos + rotate_half(x)*sin,
    # rotate_half(x) = cat(-x[D/2:], x[:D/2]).  Eager: a cat (alloc) + 2 muls + add (multi-launch).
    D = x.shape[-1]; h = D // 2
    xf = x.float()
    rot = torch.cat([-xf[..., h:], xf[..., :h]], dim=-1)
    return (xf * cos.float() + rot * sin.float()).to(x.dtype)


def _layernorm_ref(x, w, b, eps: float = 1e-5):
    # full LayerNorm: subtract mean, divide by std, affine. Two reductions (mean then var) +
    # affine — eager launches several kernels. Distinct from RMSNorm (mean-subtraction + bias).
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    var = (xc * xc).mean(-1, keepdim=True)
    return (xc * torch.rsqrt(var + eps) * w.float() + b.float()).to(x.dtype)


def _add_layernorm_ref(x, residual, w, b, eps: float = 1e-5):
    h = x.float() + residual.float()
    mu = h.mean(-1, keepdim=True)
    hc = h - mu
    var = (hc * hc).mean(-1, keepdim=True)
    return (hc * torch.rsqrt(var + eps) * w.float() + b.float()).to(x.dtype)


_GELU_C = 0.7978845608028654       # sqrt(2/pi)


def _geglu_ref(gate, up):
    # GeGLU: gelu_tanh(gate) * up  (the FFN gate with GELU instead of SiLU). tanh approximation.
    g = gate.float()
    gelu = 0.5 * g * (1.0 + torch.tanh(_GELU_C * (g + 0.044715 * g * g * g)))
    return (gelu * up.float()).to(gate.dtype)


def _qknorm_rope_ref(x, w, cos, sin, eps: float = 1e-6):
    # FUSION CHAIN (reduction -> gather): per-head RMSNorm then RoPE — the Qwen-style QK-norm.
    # The composed reference; max-autotune fuses the chain too, so beating it means the model
    # found a schedule inductor's search missed (keeping the rms scale in-register, no
    # intermediate materialized).
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    n = xf * rms * w.float()
    D = x.shape[-1]; h = D // 2
    rot = torch.cat([-n[..., h:], n[..., :h]], dim=-1)
    return (n * cos.float() + rot * sin.float()).to(x.dtype)


# ---- comprehensive suite: activations, reductions, softmax family, chains, quant ----------
def _gelu_ref(x):
    g = x.float()
    return (0.5 * g * (1.0 + torch.tanh(_GELU_C * (g + 0.044715 * g * g * g)))).to(x.dtype)


def _silu_ref(x):
    g = x.float()
    return (g * torch.sigmoid(g)).to(x.dtype)


def _relu2_ref(x):
    g = torch.relu(x.float())
    return (g * g).to(x.dtype)


def _bias_gelu_ref(x, bias):
    g = x.float() + bias.float()
    return (0.5 * g * (1.0 + torch.tanh(_GELU_C * (g + 0.044715 * g * g * g)))).to(x.dtype)


def _reglu_ref(gate, up):
    return (torch.relu(gate.float()) * up.float()).to(gate.dtype)


def _l2norm_ref(x, eps: float = 1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _log_softmax_ref(x):
    return torch.log_softmax(x.float(), dim=-1).to(x.dtype)


def _softmax_scale_ref(x, scale):
    return torch.softmax(x.float() * scale.float(), dim=-1).to(x.dtype)


def _layernorm_gelu_ref(x, w, b, eps: float = 1e-5):
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    var = (xc * xc).mean(-1, keepdim=True)
    ln = xc * torch.rsqrt(var + eps) * w.float() + b.float()
    return (0.5 * ln * (1.0 + torch.tanh(_GELU_C * (ln + 0.044715 * ln * ln * ln)))).to(x.dtype)


def _add_rmsnorm_rope_ref(x, residual, w, cos, sin, eps: float = 1e-6):
    h = x.float() + residual.float()
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    D = x.shape[-1]; hd = D // 2
    rot = torch.cat([-n[..., hd:], n[..., :hd]], dim=-1)
    return (n * cos.float() + rot * sin.float()).to(x.dtype)


def _dequant_int8_ref(q, scale):
    # per-row int8 weight dequantization: out = q * scale  (memory-bound, NON-GEMM)
    return (q.float() * scale.float()).to(torch.float16)


# ---- V2 standalone additions: real LLM ops beyond the chain grammar ----------------------
_SOFTCAP = 30.0


def _softcap_softmax_ref(x):
    # Gemma2-style logit softcapping then softmax: softmax(cap * tanh(x / cap)).
    t = _SOFTCAP * torch.tanh(x.float() / _SOFTCAP)
    return torch.softmax(t, dim=-1).to(x.dtype)


def _rmsnorm_gemma_ref(x, w, eps: float = 1e-6):
    # Gemma-style RMSNorm: scale by (1 + w), not w. The +1 is the classic silent-wrongness trap.
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms * (1.0 + w.float())).to(x.dtype)


def _glu_ref(gate, up):
    # the ORIGINAL gated linear unit: sigmoid(gate) * up
    return (torch.sigmoid(gate.float()) * up.float()).to(gate.dtype)


def _rope_interleaved_ref(x, cos, sin):
    # GPT-J / interleaved RoPE: pairs (x[2i], x[2i+1]) rotated by (cos[i], sin[i]).
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    c, s = cos.float(), sin.float()
    out = torch.empty_like(xf)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x2 * c + x1 * s
    return out.to(x.dtype)


def _cross_entropy_ref(x, tgt):
    # fused per-row cross-entropy (the Liger flagship fusion): -log_softmax(x)[tgt], no reduction.
    return torch.nn.functional.cross_entropy(x.float(), tgt, reduction="none").to(x.dtype)


# ---- INVENTION suite (V2.7): problem classes the model was never trained on --------------
def _cumsum_ref(x):
    # row-wise inclusive prefix sum — a SCAN, not a reduction: a different parallel
    # algorithm class (carry propagation across blocks) from everything in the suite.
    return torch.cumsum(x.float(), dim=-1).to(x.dtype)


def _entropy_ref(x):
    # per-row Shannon entropy of softmax(x), fused from logits: H = lse(x) - sum(x*p).
    xf = x.float()
    lse = torch.logsumexp(xf, dim=-1, keepdim=True)
    p = torch.exp(xf - lse)
    return (lse.squeeze(-1) - (xf * p).sum(-1)).to(x.dtype)


def _kl_div_ref(x, y):
    # per-row KL(softmax(x) || softmax(y)) from raw logits — the distillation op.
    # DOUBLE logsumexp fusion; both must be max-subtracted.
    xf, yf = x.float(), y.float()
    lx = xf - torch.logsumexp(xf, dim=-1, keepdim=True)
    ly = yf - torch.logsumexp(yf, dim=-1, keepdim=True)
    return (torch.exp(lx) * (lx - ly)).sum(-1).to(x.dtype)


# ----------------------------------------------------------------------------------------
@dataclass
class OpSpec:
    name: str
    reference: Callable
    make_inputs: Callable          # rng -> tuple[Tensor, ...]  (ADVERSARIAL random, for correctness)
    bench_inputs: Callable         # () -> tuple[Tensor, ...]    (FIXED large fp16, for timing)
    stress_inputs: Callable        # () -> list[tuple]  GUARANTEED killer cases (high-scale fp16/bf16,
                                   #                    odd N). Run on EVERY eval so a wrong kernel is
                                   #                    caught deterministically, never by seed luck.
    signature_hint: str
    notes: str = ""
    extra: dict = field(default_factory=dict)
    # Per-op tolerance override — DERIVED from the op's numerics (documented at the spec),
    # never tuned to make a kernel pass. Needed where the global elementwise envelope is
    # the wrong yardstick (e.g. scans: error tracks the running-path magnitude; entropy:
    # the REFERENCE itself carries catastrophic cancellation).
    tol_override: dict = field(default_factory=dict)

    def tol(self, dtype: torch.dtype) -> tuple[float, float]:
        return self.tol_override.get(dtype, tol_for(dtype))


# Guaranteed-hard cases: large magnitude (overflow trap), low precision, NON-power-of-2 row
# length (masking trap), small odd row count. These are run on every correctness check so the
# negative controls (no max-subtract / no rsqrt) fail DETERMINISTICALLY, not probabilistically.
def _stress_1tensor(N=4097, M=37):
    return [(_fixed((M, N), torch.float16, 64.0, seed=91),),
            (_fixed((M, N), torch.bfloat16, 64.0, seed=92),),
            (_fixed((3, 8192), torch.float16, 32.0, seed=93),)]


def _stress_rmsnorm(N=4097, M=37):
    return [(_fixed((M, N), torch.float16, 64.0, 91), _fixed((N,), torch.float16, 1.0, 94)),
            (_fixed((M, N), torch.bfloat16, 64.0, 92), _fixed((N,), torch.bfloat16, 1.0, 95)),
            (_fixed((3, 8192), torch.float16, 32.0, 93), _fixed((8192,), torch.float16, 1.0, 96))]


def _stress_swiglu(N=4097, M=37):
    return [(_fixed((M, N), torch.float16, 64.0, 91), _fixed((M, N), torch.float16, 8.0, 94)),
            (_fixed((M, N), torch.bfloat16, 64.0, 92), _fixed((M, N), torch.bfloat16, 8.0, 95))]


def _stress_add_rmsnorm(N=4097, M=37):
    return [(_fixed((M, N), torch.float16, 64.0, 91), _fixed((M, N), torch.float16, 64.0, 94),
             _fixed((N,), torch.float16, 1.0, 97)),
            (_fixed((M, N), torch.bfloat16, 64.0, 92), _fixed((M, N), torch.bfloat16, 64.0, 95),
             _fixed((N,), torch.bfloat16, 1.0, 98))]


# A single FIXED, realistic LLM-regime shape so a candidate's speedup is COMPARABLE across
# the whole search (apples-to-apples), and the bench input is identical for kernel + both
# baselines. fp16, deterministic seed.
_BENCH_M, _BENCH_N = 8192, 4096


def _fixed(shape, dtype=torch.float16, scale=1.0, seed=1234, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=dev, dtype=torch.float32) * scale).to(dtype)


def _bench_rmsnorm():
    return (_fixed((_BENCH_M, _BENCH_N), seed=1), _fixed((_BENCH_N,), seed=2))


def _bench_softmax():
    return (_fixed((_BENCH_M, _BENCH_N), seed=3),)


def _bench_swiglu():
    return (_fixed((_BENCH_M, _BENCH_N), seed=4), _fixed((_BENCH_M, _BENCH_N), seed=5))


def _bench_add_rmsnorm():
    return (_fixed((_BENCH_M, _BENCH_N), seed=6), _fixed((_BENCH_M, _BENCH_N), seed=7),
            _fixed((_BENCH_N,), seed=8))


def _rope_costab(M, D, seed, dtype=torch.float16, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    ang = torch.randn((M, D // 2), generator=g, device=dev, dtype=torch.float32)
    c = torch.cos(ang); s = torch.sin(ang)
    return torch.cat([c, c], -1).to(dtype), torch.cat([s, s], -1).to(dtype)


def _bench_rope():
    M, D = 32768, 128
    cos, sin = _rope_costab(M, D, 21)
    return (_fixed((M, D), seed=20), cos, sin)


def _bench_layernorm():
    return (_fixed((_BENCH_M, _BENCH_N), seed=30), _fixed((_BENCH_N,), seed=31), _fixed((_BENCH_N,), seed=32))


def _bench_add_layernorm():
    return (_fixed((_BENCH_M, _BENCH_N), seed=33), _fixed((_BENCH_M, _BENCH_N), seed=34),
            _fixed((_BENCH_N,), seed=35), _fixed((_BENCH_N,), seed=36))


def _bench_geglu():
    return (_fixed((_BENCH_M, _BENCH_N), seed=37), _fixed((_BENCH_M, _BENCH_N), seed=38))


def _bench_qknorm_rope():
    M, D = 32768, 128
    cos, sin = _rope_costab(M, D, 41)
    return (_fixed((M, D), seed=40), _fixed((D,), seed=42), cos, sin)


# ---- input generators (adversarial) ----------------------------------------------------
def _mk_rmsnorm(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    x = _randn(rng, (M, N), dt, sc)
    w = _randn(rng, (N,), dt, 1.0)
    return (x, w)


def _mk_softmax(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc),)


def _mk_swiglu(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, 1.0))


def _mk_add_rmsnorm(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, sc), _randn(rng, (N,), dt, 1.0))


_HEADDIM = [64, 128, 256]          # even head dims (RoPE)


def _mk_rope(rng):
    M, D, dt, sc = _pick(rng, _NROWS), _pick(rng, _HEADDIM), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(2**31))
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    c = torch.cos(ang); s = torch.sin(ang)
    cos = torch.cat([c, c], -1).to(dt); sin = torch.cat([s, s], -1).to(dt)
    return (_randn(rng, (M, D), dt, sc), cos, sin)


def _stress_rope():
    out = []
    for D, dt in [(128, torch.float16), (256, torch.bfloat16), (64, torch.float16)]:
        g = torch.Generator(device="cuda").manual_seed(700 + D)
        ang = torch.randn((37, D // 2), generator=g, device="cuda", dtype=torch.float32)
        c = torch.cos(ang); s = torch.sin(ang)
        out.append((_fixed((37, D), dt, 64.0, 700 + D),
                    torch.cat([c, c], -1).to(dt), torch.cat([s, s], -1).to(dt)))
    return out


def _mk_layernorm(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (N,), dt, 1.0), _randn(rng, (N,), dt, 1.0))


def _mk_add_layernorm(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, sc),
            _randn(rng, (N,), dt, 1.0), _randn(rng, (N,), dt, 1.0))


def _mk_geglu(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, 1.0))


def _stress_layernorm():
    return [(_fixed((37, 4097), torch.float16, 64.0, 91), _fixed((4097,), torch.float16, 1.0, 94),
             _fixed((4097,), torch.float16, 1.0, 95)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 92), _fixed((4097,), torch.bfloat16, 1.0, 96),
             _fixed((4097,), torch.bfloat16, 1.0, 97))]


def _stress_add_layernorm():
    return [(_fixed((37, 4097), torch.float16, 64.0, 91), _fixed((37, 4097), torch.float16, 64.0, 92),
             _fixed((4097,), torch.float16, 1.0, 94), _fixed((4097,), torch.float16, 1.0, 95)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 96), _fixed((37, 4097), torch.bfloat16, 64.0, 97),
             _fixed((4097,), torch.bfloat16, 1.0, 98), _fixed((4097,), torch.bfloat16, 1.0, 99))]


def _stress_geglu():
    return [(_fixed((37, 4097), torch.float16, 64.0, 91), _fixed((37, 4097), torch.float16, 8.0, 94)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 92), _fixed((37, 4097), torch.bfloat16, 8.0, 95))]


def _mk_qknorm_rope(rng):
    M, D, dt, sc = _pick(rng, _NROWS), _pick(rng, _HEADDIM), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(2**31))
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    c = torch.cos(ang); s = torch.sin(ang)
    cos = torch.cat([c, c], -1).to(dt); sin = torch.cat([s, s], -1).to(dt)
    return (_randn(rng, (M, D), dt, sc), _randn(rng, (D,), dt, 1.0), cos, sin)


def _stress_qknorm_rope():
    out = []
    for D, dt in [(128, torch.float16), (256, torch.bfloat16), (64, torch.float16)]:
        g = torch.Generator(device="cuda").manual_seed(800 + D)
        ang = torch.randn((37, D // 2), generator=g, device="cuda", dtype=torch.float32)
        c = torch.cos(ang); s = torch.sin(ang)
        out.append((_fixed((37, D), dt, 64.0, 800 + D), _fixed((D,), dt, 1.0, 810 + D),
                    torch.cat([c, c], -1).to(dt), torch.cat([s, s], -1).to(dt)))
    return out


# ---------- comprehensive suite: input generators ----------------------------------------
def _mk_x(rng):                       # plain elementwise (M,N)
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc),)


def _mk_x_bias(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (N,), dt, 1.0))


def _mk_gate_up(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, 1.0))


def _mk_softmax_scale(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    scl = torch.tensor([0.05 + rng.random() * 1.5], device="cuda", dtype=dt)
    return (_randn(rng, (M, N), dt, sc), scl)


def _mk_layernorm_gelu(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (N,), dt, 1.0), _randn(rng, (N,), dt, 1.0))


def _mk_add_rmsnorm_rope(rng):
    M, D, dt, sc = _pick(rng, _NROWS), _pick(rng, _HEADDIM), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(2**31))
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    c = torch.cos(ang); s = torch.sin(ang)
    return (_randn(rng, (M, D), dt, sc), _randn(rng, (M, D), dt, sc), _randn(rng, (D,), dt, 1.0),
            torch.cat([c, c], -1).to(dt), torch.cat([s, s], -1).to(dt))


def _int8(rng, shape, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(rng.randrange(2**31))
    return torch.randint(-127, 127, shape, generator=g, device=dev, dtype=torch.int8)


def _mk_dequant_int8(rng):
    M, N = _pick(rng, _NROWS), _pick(rng, _ROWLEN)
    q = _int8(rng, (M, N))
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(2**31))
    scale = (torch.rand((M, 1), generator=g, device="cuda") * 0.05 + 0.005).to(torch.float16)
    return (q, scale)


# bench inputs (fixed large)
def _bench_x(seed): return (_fixed((_BENCH_M, _BENCH_N), seed=seed),)
def _bench_gelu(): return _bench_x(50)
def _bench_silu(): return _bench_x(51)
def _bench_relu2(): return _bench_x(52)
def _bench_bias_gelu(): return (_fixed((_BENCH_M, _BENCH_N), seed=53), _fixed((_BENCH_N,), seed=54))
def _bench_reglu(): return (_fixed((_BENCH_M, _BENCH_N), seed=55), _fixed((_BENCH_M, _BENCH_N), seed=56))
def _bench_l2norm(): return _bench_x(57)
def _bench_log_softmax(): return _bench_x(58)
def _bench_softmax_scale(): return (_fixed((_BENCH_M, _BENCH_N), seed=59),
                                    torch.tensor([0.125], device="cuda", dtype=torch.float16))
def _bench_layernorm_gelu(): return (_fixed((_BENCH_M, _BENCH_N), seed=60),
                                     _fixed((_BENCH_N,), seed=61), _fixed((_BENCH_N,), seed=62))
def _bench_add_rmsnorm_rope():
    M, D = 32768, 128; cos, sin = _rope_costab(M, D, 63)
    return (_fixed((M, D), seed=64), _fixed((M, D), seed=65), _fixed((D,), seed=66), cos, sin)
_TALL_M = [4, 8, 16, 32]
_WIDE_N = [16384, 32768, 65536, 131072]
def _mk_rmsnorm_wide(rng):
    M, N, dt, sc = _pick(rng, _TALL_M), _pick(rng, _WIDE_N), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (N,), dt, 1.0))
def _bench_rmsnorm_wide(): return (_fixed((8, 131072), seed=90), _fixed((131072,), seed=91))
def _stress_rmsnorm_wide():
    return [(_fixed((8, 131072), torch.float16, 64.0, 90), _fixed((131072,), torch.float16, 1.0, 91)),
            (_fixed((5, 65537), torch.bfloat16, 32.0, 92), _fixed((65537,), torch.bfloat16, 1.0, 93)),
            (_fixed((16, 40000), torch.float16, 16.0, 94), _fixed((40000,), torch.float16, 1.0, 95))]
def _bench_dequant_int8():
    g = torch.Generator(device="cuda").manual_seed(67)
    q = torch.randint(-127, 127, (_BENCH_M, _BENCH_N), generator=g, device="cuda", dtype=torch.int8)
    g2 = torch.Generator(device="cuda").manual_seed(68)
    scale = (torch.rand((_BENCH_M, 1), generator=g2, device="cuda") * 0.05 + 0.005).to(torch.float16)
    return (q, scale)


# stress (guaranteed hard) — high magnitude / low precision / odd N
def _stress_x():
    return [(_fixed((37, 4097), torch.float16, 64.0, 71),), (_fixed((37, 4097), torch.bfloat16, 64.0, 72),),
            (_fixed((3, 8192), torch.float16, 32.0, 73),)]
def _stress_x_bias():
    return [(_fixed((37, 4097), torch.float16, 64.0, 71), _fixed((4097,), torch.float16, 4.0, 74)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 72), _fixed((4097,), torch.bfloat16, 4.0, 75))]
def _stress_gate_up():
    return [(_fixed((37, 4097), torch.float16, 64.0, 71), _fixed((37, 4097), torch.float16, 8.0, 74)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 72), _fixed((37, 4097), torch.bfloat16, 8.0, 75))]
def _stress_softmax_scale():
    return [(_fixed((37, 4097), torch.float16, 64.0, 71), torch.tensor([1.5], device="cuda", dtype=torch.float16)),
            (_fixed((37, 4097), torch.bfloat16, 32.0, 72), torch.tensor([0.5], device="cuda", dtype=torch.bfloat16))]
def _stress_layernorm_gelu():
    return [(_fixed((37, 4097), torch.float16, 64.0, 71), _fixed((4097,), torch.float16, 1.0, 74),
             _fixed((4097,), torch.float16, 1.0, 75)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 72), _fixed((4097,), torch.bfloat16, 1.0, 76),
             _fixed((4097,), torch.bfloat16, 1.0, 77))]
def _stress_add_rmsnorm_rope():
    out = []
    for D, dt in [(128, torch.float16), (256, torch.bfloat16)]:
        g = torch.Generator(device="cuda").manual_seed(820 + D)
        ang = torch.randn((37, D // 2), generator=g, device="cuda", dtype=torch.float32)
        c = torch.cos(ang); s = torch.sin(ang)
        out.append((_fixed((37, D), dt, 64.0, 820 + D), _fixed((37, D), dt, 64.0, 830 + D),
                    _fixed((D,), dt, 1.0, 840 + D), torch.cat([c, c], -1).to(dt), torch.cat([s, s], -1).to(dt)))
    return out
def _stress_dequant_int8():
    out = []
    for M, N, sd in [(37, 4097, 71), (3, 8192, 72)]:
        g = torch.Generator(device="cuda").manual_seed(sd)
        q = torch.randint(-127, 127, (M, N), generator=g, device="cuda", dtype=torch.int8)
        g2 = torch.Generator(device="cuda").manual_seed(sd + 1)
        scale = (torch.rand((M, 1), generator=g2, device="cuda") * 0.1 + 0.005).to(torch.float16)
        out.append((q, scale))
    return out


# ---- V2 standalone ops: generators / bench / stress --------------------------------------
def _mk_rope_inter(rng):
    M, D, dt, sc = _pick(rng, _NROWS), _pick(rng, _HEADDIM), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(2**31))
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    return (_randn(rng, (M, D), dt, sc), torch.cos(ang).to(dt), torch.sin(ang).to(dt))


def _bench_rope_inter():
    M, D = 32768, 128
    g = torch.Generator(device="cuda").manual_seed(140)
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    return (_fixed((M, D), seed=141), torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16))


def _stress_rope_inter():
    out = []
    for D, dt in [(128, torch.float16), (256, torch.bfloat16), (64, torch.float16)]:
        g = torch.Generator(device="cuda").manual_seed(850 + D)
        ang = torch.randn((37, D // 2), generator=g, device="cuda", dtype=torch.float32)
        out.append((_fixed((37, D), dt, 64.0, 850 + D), torch.cos(ang).to(dt), torch.sin(ang).to(dt)))
    return out


def _tgt(M, N, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, N, (M,), generator=g, device="cuda", dtype=torch.int64)


def _mk_cross_entropy(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _tgt(M, N, rng.randrange(2**31)))


def _bench_cross_entropy():
    return (_fixed((_BENCH_M, _BENCH_N), seed=150), _tgt(_BENCH_M, _BENCH_N, 151))


def _stress_cross_entropy():
    return [(_fixed((37, 4097), torch.float16, 64.0, 152), _tgt(37, 4097, 153)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 154), _tgt(37, 4097, 155)),
            (_fixed((3, 8192), torch.float16, 32.0, 156), _tgt(3, 8192, 157))]


def _bench_glu(): return (_fixed((_BENCH_M, _BENCH_N), seed=160), _fixed((_BENCH_M, _BENCH_N), seed=161))
def _bench_softcap_softmax(): return _bench_x(162)
def _bench_rmsnorm_gemma(): return (_fixed((_BENCH_M, _BENCH_N), seed=163), _fixed((_BENCH_N,), seed=164))


# ---- invention suite: generators / bench / stress ----------------------------------------
def _mk_xy(rng):
    M, N, dt, sc = _pick(rng, _NROWS), _pick(rng, _ROWLEN), _pick(rng, _DTYPES), _pick(rng, _SCALES)
    return (_randn(rng, (M, N), dt, sc), _randn(rng, (M, N), dt, sc))


def _stress_xy():
    return [(_fixed((37, 4097), torch.float16, 64.0, 171), _fixed((37, 4097), torch.float16, 64.0, 172)),
            (_fixed((37, 4097), torch.bfloat16, 64.0, 173), _fixed((37, 4097), torch.bfloat16, 64.0, 174)),
            (_fixed((3, 8192), torch.float16, 32.0, 175), _fixed((3, 8192), torch.float16, 32.0, 176))]


def _bench_cumsum(): return _bench_x(170)
def _bench_entropy(): return _bench_x(177)
def _bench_kl_div(): return (_fixed((_BENCH_M, _BENCH_N), seed=178), _fixed((_BENCH_M, _BENCH_N), seed=179))


SPECS: dict[str, OpSpec] = {
    "rmsnorm": OpSpec(
        name="rmsnorm",
        reference=_rmsnorm_ref,
        make_inputs=_mk_rmsnorm,
        bench_inputs=_bench_rmsnorm,
        stress_inputs=_stress_rmsnorm,
        signature_hint=(
            "def run(x, w):  # x:(M,N) w:(N,)  -> (M,N)\n"
            "    # y[i] = x[i] * rsqrt(mean(x[i]^2) + 1e-6) * w   (RMSNorm, accumulate in fp32)"),
        notes="row-wise normalise + per-channel scale; eager does square/mean/rsqrt/mul as separate launches.",
        extra={"eps": 1e-6},
    ),
    "softmax": OpSpec(
        name="softmax",
        reference=_softmax_ref,
        make_inputs=_mk_softmax,
        bench_inputs=_bench_softmax,
        stress_inputs=_stress_1tensor,
        signature_hint=(
            "def run(x):  # x:(M,N) -> (M,N)\n"
            "    # row-wise softmax. MUST subtract row-max before exp (numerical stability)."),
        notes="classic online/row softmax; the max-subtraction is the correctness trap large scales expose.",
    ),
    "swiglu": OpSpec(
        name="swiglu",
        reference=_swiglu_ref,
        make_inputs=_mk_swiglu,
        bench_inputs=_bench_swiglu,
        stress_inputs=_stress_swiglu,
        signature_hint=(
            "def run(gate, up):  # both (M,N) -> (M,N)\n"
            "    # SiLU(gate) * up  where SiLU(z) = z * sigmoid(z)   (FFN gate fusion)"),
        notes="elementwise activation fusion; eager launches silu + mul separately.",
    ),
    "add_rmsnorm": OpSpec(
        name="add_rmsnorm",
        reference=_add_rmsnorm_ref,
        make_inputs=_mk_add_rmsnorm,
        bench_inputs=_bench_add_rmsnorm,
        stress_inputs=_stress_add_rmsnorm,
        signature_hint=(
            "def run(x, residual, w):  # x,residual:(M,N) w:(N,) -> (M,N)\n"
            "    # h = x + residual; then RMSNorm(h) * w   (fused residual-add + norm)"),
        notes="the transformer block's add-then-norm; eager does add then a separate norm pass.",
        extra={"eps": 1e-6},
    ),
    "rope": OpSpec(
        name="rope",
        reference=_rope_ref,
        make_inputs=_mk_rope,
        bench_inputs=_bench_rope,
        stress_inputs=_stress_rope,
        signature_hint=(
            "def run(x, cos, sin):  # x,cos,sin all (M,D), D even -> (M,D)\n"
            "    # LLaMA rotate_half RoPE: out = x*cos + rotate_half(x)*sin,\n"
            "    # rotate_half(x) = concat(-x[:, D/2:], x[:, :D/2]).  Accumulate in fp32."),
        notes="fused rotary embedding; eager does a cat (alloc) + 2 muls + add as separate launches.",
    ),
    "layernorm": OpSpec(
        name="layernorm",
        reference=_layernorm_ref,
        make_inputs=_mk_layernorm,
        bench_inputs=_bench_layernorm,
        stress_inputs=_stress_layernorm,
        signature_hint=(
            "def run(x, w, b):  # x:(M,N) w,b:(N,) -> (M,N)\n"
            "    # LayerNorm: y = (x-mean)/sqrt(var+1e-5)*w + b, over the last dim. fp32 reductions."),
        notes="full layernorm (mean+var+affine); eager launches mean/var/normalize/affine separately.",
        extra={"eps": 1e-5},
    ),
    "add_layernorm": OpSpec(
        name="add_layernorm",
        reference=_add_layernorm_ref,
        make_inputs=_mk_add_layernorm,
        bench_inputs=_bench_add_layernorm,
        stress_inputs=_stress_add_layernorm,
        signature_hint=(
            "def run(x, residual, w, b):  # x,residual:(M,N) w,b:(N,) -> (M,N)\n"
            "    # h = x + residual; then LayerNorm(h)*w + b   (fused residual-add + layernorm)"),
        notes="residual-add + full layernorm; the transformer block's add-then-LN.",
        extra={"eps": 1e-5},
    ),
    "geglu": OpSpec(
        name="geglu",
        reference=_geglu_ref,
        make_inputs=_mk_geglu,
        bench_inputs=_bench_geglu,
        stress_inputs=_stress_geglu,
        signature_hint=(
            "def run(gate, up):  # both (M,N) -> (M,N)\n"
            "    # GeGLU: gelu_tanh(gate) * up,  gelu_tanh(z)=0.5*z*(1+tanh(0.7978845608*(z+0.044715*z^3)))"),
        notes="GELU-gated FFN fusion (tanh-approx GELU); eager launches gelu then mul separately.",
    ),
    "qknorm_rope": OpSpec(
        name="qknorm_rope",
        reference=_qknorm_rope_ref,
        make_inputs=_mk_qknorm_rope,
        bench_inputs=_bench_qknorm_rope,
        stress_inputs=_stress_qknorm_rope,
        signature_hint=(
            "def run(x, w, cos, sin):  # x,cos,sin:(M,D) w:(D,), D even -> (M,D)\n"
            "    # FUSED QK-norm+RoPE: n = rmsnorm(x)*w (over D); then rope(n): \n"
            "    #   out = n*cos + rotate_half(n)*sin.  Keep the rms scale in-register (no\n"
            "    #   intermediate n in DRAM). Accumulate the reduction in fp32."),
        notes="reduction->gather fusion chain (Qwen-style QK-norm then RoPE); the bigger-discovery probe.",
        extra={"eps": 1e-6},
    ),
    # ---- comprehensive suite -------------------------------------------------------------
    "gelu": OpSpec("gelu", _gelu_ref, _mk_x, _bench_gelu, _stress_x,
        "def run(x):  # (M,N)->(M,N)  gelu_tanh(x)=0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3)))",
        notes="GELU activation (tanh approx)."),
    "silu": OpSpec("silu", _silu_ref, _mk_x, _bench_silu, _stress_x,
        "def run(x):  # (M,N)->(M,N)  silu(x)=x*sigmoid(x)", notes="SiLU/Swish activation."),
    "relu2": OpSpec("relu2", _relu2_ref, _mk_x, _bench_relu2, _stress_x,
        "def run(x):  # (M,N)->(M,N)  relu(x)^2", notes="squared-ReLU activation (used in some FFNs)."),
    "bias_gelu": OpSpec("bias_gelu", _bias_gelu_ref, _mk_x_bias, _bench_bias_gelu, _stress_x_bias,
        "def run(x, bias):  # x:(M,N) bias:(N,) -> (M,N)  gelu_tanh(x + bias)",
        notes="fused bias-add + GELU (the FFN up-proj epilogue)."),
    "reglu": OpSpec("reglu", _reglu_ref, _mk_gate_up, _bench_reglu, _stress_gate_up,
        "def run(gate, up):  # both (M,N) -> (M,N)  relu(gate) * up",
        notes="ReGLU gated FFN fusion."),
    "l2norm": OpSpec("l2norm", _l2norm_ref, _mk_x, _bench_l2norm, _stress_x,
        "def run(x):  # (M,N)->(M,N)  x * rsqrt(sum(x^2) + 1e-6)  (L2 normalize over last dim)",
        notes="L2 normalization (reduction); fp32 accumulate.", extra={"eps": 1e-6}),
    "log_softmax": OpSpec("log_softmax", _log_softmax_ref, _mk_x, _bench_log_softmax, _stress_x,
        "def run(x):  # (M,N)->(M,N)  x - logsumexp(x); MUST subtract row-max for stability",
        notes="row-wise log-softmax (reduction); max-subtraction required."),
    "softmax_scale": OpSpec("softmax_scale", _softmax_scale_ref, _mk_softmax_scale, _bench_softmax_scale,
        _stress_softmax_scale,
        "def run(x, scale):  # x:(M,N) scale:(1,) -> (M,N)  softmax(x*scale[0]); subtract row-max",
        notes="fused scale + softmax (attention prelude); reduction."),
    "layernorm_gelu": OpSpec("layernorm_gelu", _layernorm_gelu_ref, _mk_layernorm_gelu, _bench_layernorm_gelu,
        _stress_layernorm_gelu,
        "def run(x, w, b):  # x:(M,N) w,b:(N,) -> (M,N)  gelu(LayerNorm(x)*w + b)",
        notes="FUSION CHAIN: layernorm (reduction) -> GELU epilogue.", extra={"eps": 1e-5}),
    "add_rmsnorm_rope": OpSpec("add_rmsnorm_rope", _add_rmsnorm_rope_ref, _mk_add_rmsnorm_rope,
        _bench_add_rmsnorm_rope, _stress_add_rmsnorm_rope,
        "def run(x, residual, w, cos, sin):  # x,res,cos,sin:(M,D) w:(D,) -> (M,D)\n"
        "    # h=x+residual; n=rmsnorm(h)*w; out=n*cos+rotate_half(n)*sin  (3-stage fusion chain)",
        notes="THREE-stage fusion chain: residual-add -> RMSNorm (reduction) -> RoPE (gather).",
        extra={"eps": 1e-6}),
    "dequant_int8": OpSpec("dequant_int8", _dequant_int8_ref, _mk_dequant_int8, _bench_dequant_int8,
        _stress_dequant_int8,
        "def run(q, scale):  # q:(M,N) int8, scale:(M,1) fp16 -> (M,N) fp16  out = q*scale  (per-row dequant)",
        notes="NON-GEMM int8 weight dequantization (memory-bound)."),
    # ---- V2 standalone ops --------------------------------------------------------------
    "softcap_softmax": OpSpec("softcap_softmax", _softcap_softmax_ref, _mk_softmax,
        _bench_softcap_softmax, _stress_1tensor,
        "def run(x):  # (M,N)->(M,N)  softmax(30*tanh(x/30)) row-wise (Gemma2 logit softcap).\n"
        "    # MUST apply the softcap BEFORE softmax and subtract the row-max before exp.",
        notes="Gemma2-style softcapped softmax; the cap is what large-scale inputs expose."),
    "rmsnorm_gemma": OpSpec("rmsnorm_gemma", _rmsnorm_gemma_ref, _mk_rmsnorm,
        _bench_rmsnorm_gemma, _stress_rmsnorm,
        "def run(x, w):  # x:(M,N) w:(N,) -> (M,N)\n"
        "    # Gemma RMSNorm: y = x * rsqrt(mean(x^2)+1e-6) * (1 + w)  — note the (1 + w)!",
        notes="Gemma-style RMSNorm: scale by (1+w); dropping the +1 is the classic silent bug.",
        extra={"eps": 1e-6}),
    "glu": OpSpec("glu", _glu_ref, _mk_gate_up, _bench_glu, _stress_gate_up,
        "def run(gate, up):  # both (M,N) -> (M,N)  sigmoid(gate) * up  (the original GLU)",
        notes="the original gated linear unit; eager launches sigmoid then mul separately."),
    "rope_interleaved": OpSpec("rope_interleaved", _rope_interleaved_ref, _mk_rope_inter,
        _bench_rope_inter, _stress_rope_inter,
        "def run(x, cos, sin):  # x:(M,D), cos,sin:(M,D/2), D even -> (M,D)\n"
        "    # GPT-J INTERLEAVED RoPE: out[2i]=x[2i]*cos[i]-x[2i+1]*sin[i];\n"
        "    #                         out[2i+1]=x[2i+1]*cos[i]+x[2i]*sin[i]. fp32 math.",
        notes="interleaved-pair rotary (GPT-J/NeoX-style); strided pair access is the fusion win."),
    "cross_entropy": OpSpec("cross_entropy", _cross_entropy_ref, _mk_cross_entropy,
        _bench_cross_entropy, _stress_cross_entropy,
        "def run(x, tgt):  # x:(M,N) fp, tgt:(M,) int64 -> (M,)\n"
        "    # per-row cross-entropy: logsumexp(x) - x[tgt]. MUST subtract row-max inside\n"
        "    # the logsumexp (stability). Accumulate in fp32; output dtype = x.dtype.",
        notes="fused cross-entropy (the Liger flagship): softmax+log+gather in one pass."),
    # ---- INVENTION suite: never-trained problem classes ----------------------------------
    "cumsum": OpSpec("cumsum", _cumsum_ref, _mk_x, _bench_cumsum, _stress_x,
        "def run(x):  # (M,N)->(M,N)  row-wise INCLUSIVE prefix sum (cumsum along the last dim).\n"
        "    # This is a SCAN: each output depends on ALL previous elements in the row —\n"
        "    # a carry must propagate across blocks. Accumulate in fp32.",
        notes="prefix-scan algorithm class (carry across blocks) — unlike every reduction op.",
        # SCAN tolerance: rounding error tracks the RUNNING-SUM magnitude (scale*sqrt(j)),
        # not the output element — fp32 envelope: eps*scale_max*sqrt(N_max) ≈ 1.2e-7*64*90
        # ≈ 7e-4, ×~constants → 5e-3 atol. Wrong-kernel errors here are 1e3+, so the
        # verification stays sharp. fp16/bf16 global atol already dominates this term.
        tol_override={torch.float32: (2e-4, 5e-3)}),
    "entropy": OpSpec("entropy", _entropy_ref, _mk_x, _bench_entropy, _stress_x,
        "def run(x):  # x:(M,N) -> (M,)  Shannon entropy of softmax(x) per row:\n"
        "    # H = logsumexp(x) - sum(x * softmax(x)). MUST subtract the row max inside\n"
        "    # both the logsumexp and the softmax (stability). fp32 accumulation.",
        notes="fused entropy-from-logits (sampling diagnostics): two coupled reductions.",
        # H = lse - Σx·p subtracts two O(scale·|x|) quantities — the REFERENCE itself
        # carries this cancellation, so elementwise agreement beyond eps*|x|_max*sqrt(N)
        # ≈ 1.2e-7*300*64 ≈ 2e-3 is unattainable for ANY correct kernel. fp32 atol 2e-2
        # against H ∈ [0, log N≈9]; the no-max-sub control still fails with nan/inf.
        tol_override={torch.float32: (2e-4, 2e-2)}),
    "kl_div": OpSpec("kl_div", _kl_div_ref, _mk_xy, _bench_kl_div, _stress_xy,
        "def run(x, y):  # both (M,N) logits -> (M,)  KL(softmax(x) || softmax(y)) per row:\n"
        "    # lx = x - lse(x); ly = y - lse(y); out = sum(exp(lx) * (lx - ly)).\n"
        "    # BOTH logsumexps must be max-subtracted. fp32 accumulation.",
        notes="fused distillation KL from raw logit pairs: double logsumexp + weighted sum."),
    # ---- algorithm-discovery experiment: rmsnorm at a TALL-SKINNY shape where split-K wins ---
    "rmsnorm_wide": OpSpec("rmsnorm_wide", _rmsnorm_ref, _mk_rmsnorm_wide, _bench_rmsnorm_wide,
        _stress_rmsnorm_wide,
        "def run(x, w):  # x:(M,N) w:(N,) -> (M,N)  RMSNorm; M is SMALL, N is HUGE (tall-skinny).\n"
        "    # one-program-per-row leaves the GPU idle; split the row's reduction across programs.",
        notes="rmsnorm at tall-skinny (M<<#SMs, huge N) — the regime where split-K reduction wins.",
        extra={"eps": 1e-6}),
}


def get_spec(name: str) -> OpSpec:
    if name not in SPECS:
        raise KeyError(f"unknown op {name!r}; have {sorted(SPECS)}")
    return SPECS[name]


# ----------------------------------------------------------------------------------------
# SHAPE-GRID inputs (V2, purely ADDITIVE — references/tolerances/bench_inputs untouched).
# Builds inputs for any op at an arbitrary (M, N[, dtype]) so the harness can re-bench the
# same kernel across a grid of shapes. For rope-family ops N is the head dim D (must be
# even). Deterministic seeds derived from the shape so every grid cell is reproducible.
# ----------------------------------------------------------------------------------------
_ROPE_FAMILY = {"rope", "rope_interleaved", "qknorm_rope", "add_rmsnorm_rope"}


def _grid_kind(name: str) -> str:
    """Input-signature kind for an op (mirrors _register_chains's _IN map + explicit ops)."""
    explicit = {
        "softmax": "x", "log_softmax": "x", "gelu": "x", "silu": "x", "relu2": "x",
        "l2norm": "x", "softcap_softmax": "x",
        "rmsnorm": "rms", "rmsnorm_wide": "rms", "rmsnorm_gemma": "rms",
        "layernorm": "ln", "layernorm_gelu": "ln",
        "add_rmsnorm": "add_rms", "add_layernorm": "add_ln",
        "swiglu": "gate_up", "geglu": "gate_up", "reglu": "gate_up", "glu": "gate_up",
        "bias_gelu": "x_bias", "softmax_scale": "x_scale", "dequant_int8": "int8",
        "cross_entropy": "ce", "rope": "rope", "rope_interleaved": "rope_inter",
        "qknorm_rope": "qkr", "add_rmsnorm_rope": "arr",
        "cumsum": "x", "entropy": "x", "kl_div": "gate_up",
    }
    if name in explicit:
        return explicit[name]
    if name.endswith("_short"):
        return _grid_kind(name[: -len("_short")])
    try:
        import chains
        for cname, kind, _ref, _s in chains.all_chains():
            if cname == name:
                return {"rms": "rms", "add_rms": "add_rms", "ln": "ln", "add_ln": "add_ln"}[kind]
    except Exception:
        pass
    raise KeyError(f"no grid input builder for op {name!r}")


def grid_inputs(name: str, M: int, N: int, dtype=torch.float16) -> tuple:
    kind = _grid_kind(name)
    sd = (hash((name, M, N, str(dtype))) & 0x7FFFFFF) + 7
    x = _fixed((M, N), dtype, 1.0, seed=sd)
    if kind == "x":
        return (x,)
    if kind == "rms":
        return (x, _fixed((N,), dtype, 1.0, seed=sd + 1))
    if kind == "ln":
        return (x, _fixed((N,), dtype, 1.0, seed=sd + 1), _fixed((N,), dtype, 1.0, seed=sd + 2))
    if kind == "add_rms":
        return (x, _fixed((M, N), dtype, 1.0, seed=sd + 3), _fixed((N,), dtype, 1.0, seed=sd + 1))
    if kind == "add_ln":
        return (x, _fixed((M, N), dtype, 1.0, seed=sd + 3),
                _fixed((N,), dtype, 1.0, seed=sd + 1), _fixed((N,), dtype, 1.0, seed=sd + 2))
    if kind == "gate_up":
        return (x, _fixed((M, N), dtype, 1.0, seed=sd + 3))
    if kind == "x_bias":
        return (x, _fixed((N,), dtype, 1.0, seed=sd + 1))
    if kind == "x_scale":
        return (x, torch.tensor([0.125], device="cuda", dtype=dtype))
    if kind == "ce":
        g = torch.Generator(device="cuda").manual_seed(sd + 4)
        return (x, torch.randint(0, N, (M,), generator=g, device="cuda", dtype=torch.int64))
    if kind == "int8":
        g = torch.Generator(device="cuda").manual_seed(sd)
        q = torch.randint(-127, 127, (M, N), generator=g, device="cuda", dtype=torch.int8)
        g2 = torch.Generator(device="cuda").manual_seed(sd + 1)
        return (q, (torch.rand((M, 1), generator=g2, device="cuda") * 0.05 + 0.005).to(torch.float16))
    # rope family: N is the head dim D (even)
    D = N
    if D % 2:
        raise ValueError(f"{name}: head dim must be even, got {D}")
    g = torch.Generator(device="cuda").manual_seed(sd + 5)
    ang = torch.randn((M, D // 2), generator=g, device="cuda", dtype=torch.float32)
    c, s = torch.cos(ang), torch.sin(ang)
    if kind == "rope_inter":
        return (x, c.to(dtype), s.to(dtype))
    cos = torch.cat([c, c], -1).to(dtype)
    sin = torch.cat([s, s], -1).to(dtype)
    if kind == "rope":
        return (x, cos, sin)
    if kind == "qkr":
        return (x, _fixed((D,), dtype, 1.0, seed=sd + 1), cos, sin)
    if kind == "arr":
        return (x, _fixed((M, D), dtype, 1.0, seed=sd + 3), _fixed((D,), dtype, 1.0, seed=sd + 1), cos, sin)
    raise KeyError(kind)


# ---- SHORT-ROW regime variants (V2.7 invention targets) -----------------------------------
# The shape-grid characterized ONE loss region for the whole product: 16384x2048 (many short
# rows), where row-per-program schedules underuse the GPU and inductor wins. These variants
# are the SAME ops with the bench (and its correctness case) pinned INSIDE that region —
# the invention question is whether RL finds a schedule family (split-row / multi-row /
# persistent) that wins where its entire current style loses. Adversarial correctness sweep
# unchanged (same make_inputs/stress).
_SHORT_M, _SHORT_N = 16384, 2048
_SHORT_BASES = ["rmsnorm", "softmax", "layernorm_gelu", "add_layernorm_sigmoid",
                # F1 falsification slate: the 10 worst remaining loss-cell ops — does the
                # whole-row style TRANSFER from rl_adapter_invent without new ideas?
                "add_layernorm_tanh", "add_layernorm_silu", "add_layernorm_gelu",
                "layernorm_sigmoid", "layernorm_tanh", "rmsnorm_silu", "add_rmsnorm_silu",
                "add_rmsnorm_sigmoid", "add_layernorm_square", "add_rmsnorm_gelu"]


def _register_short_variants():
    for base in _SHORT_BASES:
        b = SPECS[base]
        name = base + "_short"
        if name in SPECS:
            continue
        SPECS[name] = OpSpec(
            name, b.reference, b.make_inputs,
            (lambda base=base: grid_inputs(base, _SHORT_M, _SHORT_N)),
            b.stress_inputs,
            b.signature_hint + "\n    # REGIME: M=16384 rows of only N=2048. One-program-per-row "
            "underuses the GPU here;\n    # consider processing MULTIPLE rows per program or "
            "splitting work differently.",
            notes=f"{base} pinned to the characterized loss regime (16384x2048) — schedule invention target.",
            extra=dict(b.extra))


# ---- register the generative fusion-chain grammar (chains.py) ----------------------------
def _register_chains():
    import chains
    _IN = {"rms":     (_mk_rmsnorm, _bench_rmsnorm, _stress_rmsnorm),
           "add_rms": (_mk_add_rmsnorm, _bench_add_rmsnorm, _stress_add_rmsnorm),
           "ln":      (_mk_layernorm, _bench_layernorm, _stress_layernorm),
           "add_ln":  (_mk_add_layernorm, _bench_add_layernorm, _stress_add_layernorm)}
    _SIG = {"rms": "def run(x, w):", "add_rms": "def run(x, residual, w):",
            "ln": "def run(x, w, b):", "add_ln": "def run(x, residual, w, b):"}
    for name, kind, ref, _structs in chains.all_chains():
        if name in SPECS:
            continue
        mk, bench, stress = _IN[kind]
        act = name.rsplit("_", 1)[-1]
        SPECS[name] = OpSpec(name, ref, mk, bench, stress,
            f"{_SIG[kind]}  # fused {name}: {kind.replace('add_','residual+').replace('rms','RMSNorm').replace('ln','LayerNorm')} "
            f"then {act} epilogue; accumulate the reduction in fp32",
            notes=f"generative fusion chain (reduction->epilogue): {name}.")
_register_chains()
_register_short_variants()
