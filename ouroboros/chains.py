"""Generative fusion-chain grammar: [+residual] -> {rms|layer}norm -> ×w(+b) -> epilogue.

The widest sweep of the reduction->epilogue region (where the compiler under-fuses and the 2B
wins). References are COMPOSED; teacher kernels are TEMPLATE-GENERATED (scalar-reduce +
whole-row variants). Everything is harness-filtered downstream — nothing here is trusted.
"""
from __future__ import annotations
import torch

_C = 0.7978845608028654  # sqrt(2/pi)

# epilogue: (torch fn on fp32 tensor, triton expression in fp32 var `n`)
ACTS = {
    "gelu":  (lambda t: 0.5 * t * (1.0 + torch.tanh(_C * (t + 0.044715 * t * t * t))),
              "(0.5 * n * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * (n + 0.044715 * n * n * n))) - 1.0)))"),
    "silu":  (lambda t: t * torch.sigmoid(t), "(n * tl.sigmoid(n))"),
    "relu2": (lambda t: torch.relu(t) * torch.relu(t), "(tl.maximum(n, 0.0) * tl.maximum(n, 0.0))"),
    # --- expanded grammar (each torch fn EXACTLY matches its triton expr; no approximation) ---
    "tanh":    (lambda t: torch.tanh(t), "(2.0 * tl.sigmoid(2.0 * n) - 1.0)"),   # identity tanh(x)=2σ(2x)-1
    "sigmoid": (lambda t: torch.sigmoid(t), "tl.sigmoid(n)"),
    "relu":    (lambda t: torch.relu(t), "tl.maximum(n, 0.0)"),
    "square":  (lambda t: t * t, "(n * n)"),
    # --- 2c round 2: more real activations (each torch fn EXACTLY matches its triton expr) ---
    "abs":      (lambda t: torch.abs(t), "tl.abs(n)"),
    "softsign": (lambda t: t / (1.0 + torch.abs(t)), "(n / (1.0 + tl.abs(n)))"),
    "hardsigmoid": (lambda t: torch.clamp(t + 3.0, 0.0, 6.0) / 6.0,
                    "(tl.minimum(tl.maximum(n + 3.0, 0.0), 6.0) / 6.0)"),          # F.hardsigmoid
    "hardswish":   (lambda t: t * torch.clamp(t + 3.0, 0.0, 6.0) / 6.0,
                    "(n * tl.minimum(tl.maximum(n + 3.0, 0.0), 6.0) / 6.0)"),      # F.hardswish
}
NORMS = ["rms", "layer"]
RESID = [False, True]
ACTNAMES = ["gelu", "silu", "relu2", "tanh", "sigmoid", "relu", "square",
            "abs", "softsign", "hardsigmoid", "hardswish"]


def chain_name(norm, residual, act):
    return ("add_" if residual else "") + ("rmsnorm" if norm == "rms" else "layernorm") + "_" + act


def chain_kind(norm, residual):
    return ("add_" if residual else "") + ("rms" if norm == "rms" else "ln")   # -> input signature


def chain_reference(norm, residual, act, eps=None):
    eps = eps if eps is not None else (1e-6 if norm == "rms" else 1e-5)
    fn = ACTS[act][0]
    def ref(*args):
        if residual and norm == "rms":
            x, r, w = args; h = x.float() + r.float(); b = None
        elif residual:
            x, r, w, b = args; h = x.float() + r.float()
        elif norm == "rms":
            x, w = args; h = x.float(); b = None
        else:
            x, w, b = args; h = x.float()
        if norm == "rms":
            n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        else:
            mu = h.mean(-1, keepdim=True); hc = h - mu
            n = hc * torch.rsqrt((hc * hc).mean(-1, keepdim=True) + eps) * w.float() + b.float()
        return fn(n).to(args[0].dtype)
    return ref


# ---- teacher-kernel templates -----------------------------------------------------------
def _kernel(norm, residual, act_expr, eps, variant):
    """variant: 'scalar' (loop+scalar accumulator) or 'whole' (single block per row)."""
    ptrs = "x_ptr, " + ("r_ptr, " if residual else "") + "w_ptr, " + ("b_ptr, " if norm == "layer" else "") + "y_ptr"
    sig = "x, " + ("residual, " if residual else "") + "w" + (", b" if norm == "layer" else "")
    launch = "x, " + ("residual, " if residual else "") + "w" + (", b" if norm == "layer" else "") + ", y"
    radv = " r_ptr += row * stride;" if residual else ""
    hload = ("tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)"
             + (" + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)" if residual else ""))
    # bias load indent differs: scalar variant loads it INSIDE the apply for-loop (8 spaces),
    # whole-row loads it flat (4 spaces). Wrong indent -> IndentationError.
    bload8 = "        b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)\n" if norm == "layer" else ""
    bload4 = "    b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)\n" if norm == "layer" else ""
    if norm == "rms":
        normed = "h * rr * w"
    else:
        normed = "(h - mu) * rr * w + b"

    if variant == "scalar":
        if norm == "rms":
            reduce_block = f'''    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        h = {hload}
        s += tl.sum(h * h)
    rr = tl.rsqrt(s / N + eps)'''
        else:
            reduce_block = f'''    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        s += tl.sum({hload})
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        d = tl.where(MM, ({hload}) - mu, 0.0); v += tl.sum(d * d)
    rr = tl.rsqrt(v / N + eps)'''
        body = f'''@triton.jit
def _k({ptrs}, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride;{radv} y_ptr += row * stride
{reduce_block}
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        h = {hload}
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
{bload8}        n = {normed}
        tl.store(y_ptr + cols, {act_expr}, mask=MM)
def run({sig}):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)]({launch}, x.stride(0), N, {eps}, BLOCK=1024)
    return y
'''
    else:  # whole-row single block
        if norm == "rms":
            stat = "    rr = tl.rsqrt(tl.sum(h * h) / N + eps)"
        else:
            stat = ("    mu = tl.sum(h) / N\n    hc = tl.where(MM, h - mu, 0.0)\n"
                    "    rr = tl.rsqrt(tl.sum(hc * hc) / N + eps)")
            normed = "hc * rr * w + b"
        body = f'''@triton.jit
def _k({ptrs}, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride;{radv} y_ptr += row * stride
    cols = tl.arange(0, BLOCK); MM = cols < N
    h = {hload}
{stat}
    w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
{bload4}    n = {normed}
    tl.store(y_ptr + cols, {act_expr}, mask=MM)
def run({sig}):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)]({launch}, x.stride(0), N, {eps}, BLOCK=triton.next_power_of_2(N))
    return y
'''
    return body


def chain_structures(norm, residual, act):
    eps = 1e-6 if norm == "rms" else 1e-5
    expr = ACTS[act][1]
    return [_kernel(norm, residual, expr, eps, "scalar"), _kernel(norm, residual, expr, eps, "whole")]


def all_chains():
    """[(name, kind, reference_fn, [kernel_src, ...]), ...] for the full grammar."""
    out = []
    for norm in NORMS:
        for residual in RESID:
            for act in ACTNAMES:
                name = chain_name(norm, residual, act)
                out.append((name, chain_kind(norm, residual), chain_reference(norm, residual, act),
                            chain_structures(norm, residual, act)))
    return out
