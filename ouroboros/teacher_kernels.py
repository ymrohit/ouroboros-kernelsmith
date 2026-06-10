"""Teacher-authored DIVERSE Triton kernels — the SFT corpus seed.

The advisor's load-bearing point: SFT on knob-twiddled copies of 4 seeds teaches "memorize
4 programs," not "write Triton." So the strong teacher (Opus) hand-writes STRUCTURALLY
DISTINCT correct implementations per op — different reduction strategies, tilings, one-pass
vs multi-pass, scalar vs vector accumulation. Each is then expanded with launch-knob variants
(BLOCK / num_warps / num_stages) and HARNESS-FILTERED to verified-only by sft_train.py.

Every kernel here is meant to be CORRECT (accumulate in fp32, mask non-power-of-2 N, survive
large-magnitude fp16/bf16). The harness is the judge; broken ones are dropped, not trusted.
"""
from __future__ import annotations
import re

# ============================== RMSNorm — distinct structures ==============================
_RMS_TWOPASS_VEC = '''
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    r = tl.rsqrt(tl.sum(acc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * r * w, mask=m)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''

_RMS_TWOPASS_SCALAR = '''
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    r = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * r * w, mask=m)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''

_RMS_WHOLEROW = '''
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x) / N + eps)
    tl.store(y_ptr + row * stride + cols, x * r * w, mask=m)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y
'''

# ============================== Softmax — distinct structures ==============================
_SM_THREEPASS = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    mx = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        mx = tl.maximum(mx, x)
    rmax = tl.max(mx)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        d += tl.where(cols < N, tl.exp(x - rmax), 0.0)
    den = tl.sum(d)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.exp(x - rmax) / den, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
'''

_SM_WHOLEROW = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x)
    e = tl.where(m, tl.exp(x), 0.0)
    tl.store(y_ptr + row * stride + cols, e / tl.sum(e), mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
'''

_SM_ONLINE = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    m_i = -float("inf"); l_i = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(x))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.where(cols < N, tl.exp(x - m_new), 0.0))
        m_i = m_new
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.exp(x - m_i) / l_i, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
'''

# ============================== SwiGLU — distinct structures ===============================
_SW_FLAT = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, (g * (1.0 / (1.0 + tl.exp(-g)))) * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
'''

_SW_SIGMOID = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, g * tl.sigmoid(g) * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
'''

_SW_ROW = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); g_ptr += row * stride; u_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, g * tl.sigmoid(g) * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M,)](gate, up, y, gate.stride(0), N, BLOCK=1024)
    return y
'''

# ============================== add+RMSNorm — distinct structures ==========================
_AR_TWOPASS = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        acc += h * h
    rr = tl.rsqrt(tl.sum(acc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, h * rr * w, mask=m)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''

_AR_WHOLEROW = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, BLOCK); m = cols < N
    h = tl.load(x_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32) + \\
        tl.load(r_ptr + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rr = tl.rsqrt(tl.sum(h * h) / N + eps)
    tl.store(y_ptr + row * stride + cols, h * rr * w, mask=m)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y
'''

_AR_SCALAR = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        s += tl.sum(h * h)
    rr = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, h * rr * w, mask=m)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''

# ============================== RoPE — distinct structures =================================
_ROPE_HALFSPLIT = '''
@triton.jit
def _k(x_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < h
    x1 = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x1 * c1 - x2 * s1, mask=m)
    tl.store(y_ptr + h + cols, x2 * c2 + x1 * s2, mask=m)
def run(x, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, cos, sin, y, x.stride(0), D, BLOCK=triton.next_power_of_2(D // 2))
    return y
'''

_ROPE_SHIFTLOAD = '''
@triton.jit
def _k(x_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < D
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    cs = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    shifted = tl.where(cols < h, cols + h, cols - h)
    xr = tl.load(x_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    sign = tl.where(cols < h, -1.0, 1.0)
    tl.store(y_ptr + cols, x * cs + sign * xr * sn, mask=m)
def run(x, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, cos, sin, y, x.stride(0), D, BLOCK=triton.next_power_of_2(D))
    return y
'''

# ============================== LayerNorm — distinct structures ============================
_LN_SCALAR = '''
@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32))
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, x - mu, 0.0); v += tl.sum(d * d)
    r = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x - mu) * r * w + b, mask=m)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
'''

_LN_WHOLEROW = '''
@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    mu = tl.sum(x) / N
    xc = tl.where(m, x - mu, 0.0)
    r = tl.rsqrt(tl.sum(xc * xc) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, xc * r * w + b, mask=m)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=triton.next_power_of_2(N))
    return y
'''

# ============================== add+LayerNorm — distinct structures ========================
_ALN_SCALAR = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        s += tl.sum(h)
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, h - mu, 0.0); v += tl.sum(d * d)
    rr = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (h - mu) * rr * w + b, mask=m)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
'''

_ALN_WHOLEROW = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
        tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
    mu = tl.sum(h) / N
    hc = tl.where(m, h - mu, 0.0)
    rr = tl.rsqrt(tl.sum(hc * hc) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, hc * rr * w + b, mask=m)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, BLOCK=triton.next_power_of_2(N))
    return y
'''

# ============================== GeGLU — distinct structures ================================
# tl.tanh is absent in this triton build -> tanh(a) = 2*sigmoid(2a) - 1.
_GEGLU_FLAT = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    a = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    gelu = 0.5 * g * (1.0 + 2.0 * tl.sigmoid(2.0 * a) - 1.0)
    tl.store(y_ptr + cols, gelu * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
'''

_GEGLU_ROW = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); g_ptr += row * stride; u_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
        a = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        gelu = 0.5 * g * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0))
        tl.store(y_ptr + cols, gelu * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M,)](gate, up, y, gate.stride(0), N, BLOCK=1024)
    return y
'''

# --- third distinct structures (full rigor: 3 per op, matching the originals) -------------
_ROPE_TILED = '''
@triton.jit
def _k(x_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < h
    x1 = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x1 * c - x2 * s, mask=m)
    tl.store(y_ptr + h + cols, x2 * c + x1 * s, mask=m)
def run(x, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, cos, sin, y, x.stride(0), D, BLOCK=triton.next_power_of_2(D // 2))
    return y
'''

_LN_VEC = '''
@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    mu = tl.sum(acc) / N
    vacc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, x - mu, 0.0); vacc += d * d
    r = tl.rsqrt(tl.sum(vacc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x - mu) * r * w + b, mask=m)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
'''

_ALN_VEC = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        acc += tl.where(m, h, 0.0)
    mu = tl.sum(acc) / N
    vacc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, h - mu, 0.0); vacc += d * d
    rr = tl.rsqrt(tl.sum(vacc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (h - mu) * rr * w + b, mask=m)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
'''

_GEGLU_2D = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK); m = col < N
    base = row * stride
    g = tl.load(g_ptr + base + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + base + col, mask=m, other=0.0).to(tl.float32)
    a = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    gelu = 0.5 * g * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0))
    tl.store(y_ptr + base + col, gelu * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M, triton.cdiv(N, 1024))](gate, up, y, gate.stride(0), N, BLOCK=1024)
    return y
'''

# ====================== qknorm_rope (FUSION CHAIN: rmsnorm -> rope) =========================
# The rms scale `r` is a per-row scalar; rotated n = x[shifted]*r*w[shifted] — no intermediate
# normed tensor materialized in DRAM (the fusion win the autotuner leaves on the table).
_QKR_SHIFTLOAD = '''
@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < D
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x) / D + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    n = x * r * w
    shifted = tl.where(cols < h, cols + h, cols - h)
    xs = tl.load(x_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    ws = tl.load(w_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(cols < h, -1.0, 1.0) * (xs * r * ws)
    cs = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, n * cs + rot * sn, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
'''

_QKR_HALFSPLIT = '''
@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2
    full = tl.load(x_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < D, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(full * full) / D + eps)
    c = tl.arange(0, BLOCK); m = c < h
    x1 = tl.load(x_ptr + c, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    n1 = x1 * r * w1; n2 = x2 * r * w2
    c1 = tl.load(cos_ptr + c, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + c, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + c, n1 * c1 - n2 * s1, mask=m)
    tl.store(y_ptr + h + c, n2 * c2 + n1 * s2, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
'''

_QKR_TILEDCOS = '''
@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2
    full = tl.load(x_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < D, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(full * full) / D + eps)
    c = tl.arange(0, BLOCK); m = c < h
    x1 = tl.load(x_ptr + c, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    cc = tl.load(cos_ptr + c, mask=m, other=0.0).to(tl.float32)
    ss = tl.load(sin_ptr + c, mask=m, other=0.0).to(tl.float32)
    n1 = x1 * r * w1; n2 = x2 * r * w2
    tl.store(y_ptr + c, n1 * cc - n2 * ss, mask=m)
    tl.store(y_ptr + h + c, n2 * cc + n1 * ss, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
'''

# ====================== comprehensive suite: teacher structures ============================
def _flat_act(expr):   # elementwise activation, flat 1D grid; `expr` uses fp32 var `x`
    return f'''
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, {expr}, mask=m)
def run(x):
    y = torch.empty_like(x); n = x.numel()
    _k[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y
'''

def _row_act(expr):    # elementwise activation, one row per program (loop over N)
    return f'''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, {expr}, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
'''

_GELU_E = "0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * (x + 0.044715 * x * x * x))) - 1.0))"
_SILU_E = "x * tl.sigmoid(x)"
_RELU2_E = "tl.maximum(x, 0.0) * tl.maximum(x, 0.0)"

_BIASGELU_FLAT = '''
@triton.jit
def _k(x_ptr, b_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \\
            tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        a = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        tl.store(y_ptr + cols, 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0)), mask=m)
def run(x, bias):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, bias, y, x.stride(0), N, BLOCK=1024)
    return y
'''

_REGLU_FLAT = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, tl.maximum(g, 0.0) * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
'''
_REGLU_ROW = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); g_ptr += row * stride; u_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.maximum(g, 0.0) * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M,)](gate, up, y, gate.stride(0), N, BLOCK=1024)
    return y
'''

_L2_WHOLEROW = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x * tl.rsqrt(tl.sum(x * x) + eps), mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y
'''
_L2_SCALAR = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32) ** 2)
    r = tl.rsqrt(s + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * r, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''

_LOGSM_WHOLEROW = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x)
    lse = tl.log(tl.sum(tl.where(m, tl.exp(x), 0.0)))
    tl.store(y_ptr + cols, x - lse, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
'''
_LOGSM_TWOPASS = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    mx = -float("inf")
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mx = tl.maximum(mx, tl.max(tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)))
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        s += tl.sum(tl.where(cols < N, tl.exp(x - mx), 0.0))
    lse = mx + tl.log(s)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x - lse, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
'''

_SMSCALE_WHOLEROW = '''
@triton.jit
def _k(x_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    sc = tl.load(s_ptr).to(tl.float32)
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=-float("inf")).to(tl.float32) * sc
    x = x - tl.max(x)
    e = tl.where(m, tl.exp(x), 0.0)
    tl.store(y_ptr + cols, e / tl.sum(e), mask=m)
def run(x, scale):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, scale, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
'''
_SMSCALE_TWOPASS = '''
@triton.jit
def _k(x_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    sc = tl.load(s_ptr).to(tl.float32)
    mx = -float("inf")
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32) * sc
        mx = tl.maximum(mx, tl.max(v))
    d = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32) * sc
        d += tl.sum(tl.where(cols < N, tl.exp(v - mx), 0.0))
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        v = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) * sc
        tl.store(y_ptr + cols, tl.exp(v - mx) / d, mask=m)
def run(x, scale):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, scale, y, x.stride(0), N, BLOCK=1024)
    return y
'''

_LNGELU_SCALAR = '''
@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32))
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        d = tl.where(m, tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) - mu, 0.0)
        v += tl.sum(d * d)
    r = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        ln = (x - mu) * r * w + b
        a = 0.7978845608028654 * (ln + 0.044715 * ln * ln * ln)
        tl.store(y_ptr + cols, 0.5 * ln * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0)), mask=m)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
'''

_ARR_SHIFT = '''
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; r_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < D
    hh = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(hh * hh) / D + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    n = hh * rms * w
    sh = tl.where(cols < h, cols + h, cols - h)
    hs = tl.load(x_ptr + sh, mask=m, other=0.0).to(tl.float32) + tl.load(r_ptr + sh, mask=m, other=0.0).to(tl.float32)
    ws = tl.load(w_ptr + sh, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(cols < h, -1.0, 1.0) * (hs * rms * ws)
    cs = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, n * cs + rot * sn, mask=m)
def run(x, residual, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
'''

_DEQ_ROW = '''
@triton.jit
def _k(q_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); q_ptr += row * stride; y_ptr += row * stride
    sc = tl.load(s_ptr + row).to(tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        q = tl.load(q_ptr + cols, mask=m, other=0).to(tl.float32)
        tl.store(y_ptr + cols, q * sc, mask=m)
def run(q, scale):
    M, N = q.shape
    y = torch.empty((M, N), device=q.device, dtype=torch.float16)
    _k[(M,)](q, scale, y, q.stride(0), N, BLOCK=1024)
    return y
'''
_DEQ_WHOLEROW = '''
@triton.jit
def _k(q_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sc = tl.load(s_ptr + row).to(tl.float32)
    cols = tl.arange(0, BLOCK); m = cols < N
    q = tl.load(q_ptr + row * stride + cols, mask=m, other=0).to(tl.float32)
    tl.store(y_ptr + row * stride + cols, q * sc, mask=m)
def run(q, scale):
    M, N = q.shape
    y = torch.empty((M, N), device=q.device, dtype=torch.float16)
    _k[(M,)](q, scale, y, q.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
'''

# ---- V2 standalone ops: teacher structures ------------------------------------------------
_SOFTCAP_WHOLEROW = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    c = 30.0 * (2.0 * tl.sigmoid(2.0 * (x / 30.0)) - 1.0)
    c = tl.where(m, c, -float("inf"))
    e = tl.exp(c - tl.max(c, 0))
    e = tl.where(m, e, 0.0)
    tl.store(y_ptr + cols, e / tl.sum(e, 0), mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
'''
_SOFTCAP_TWOPASS = '''
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    mx = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        c = 30.0 * (2.0 * tl.sigmoid(2.0 * (x / 30.0)) - 1.0)
        mx = tl.maximum(mx, tl.where(m, c, -float("inf")))
    rmax = tl.max(mx)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        c = 30.0 * (2.0 * tl.sigmoid(2.0 * (x / 30.0)) - 1.0)
        d += tl.where(m, tl.exp(c - rmax), 0.0)
    denom = tl.sum(d)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        c = 30.0 * (2.0 * tl.sigmoid(2.0 * (x / 30.0)) - 1.0)
        tl.store(y_ptr + cols, tl.exp(c - rmax) / denom, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
'''
_RG_SCALAR = '''
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rr = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * rr * (1.0 + w), mask=m)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
'''
_RG_WHOLEROW = '''
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rr = tl.rsqrt(tl.sum(x * x) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x * rr * (1.0 + w), mask=m)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y
'''
_GLU_FLAT = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK); m = cols < n_elem
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, tl.sigmoid(g) * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
'''
_GLU_ROW = '''
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); g_ptr += row * stride; u_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.sigmoid(g) * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M,)](gate, up, y, gate.stride(0), N, BLOCK=1024)
    return y
'''
_RI_STRIDED = '''
@triton.jit
def _k(x_ptr, cos_ptr, sin_ptr, y_ptr, xs, cs, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * xs; y_ptr += row * xs; cos_ptr += row * cs; sin_ptr += row * cs
    i = tl.arange(0, BLOCK); m = i < H
    x1 = tl.load(x_ptr + 2 * i, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + i, mask=m, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + 2 * i, x1 * c - x2 * s, mask=m)
    tl.store(y_ptr + 2 * i + 1, x2 * c + x1 * s, mask=m)
def run(x, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, cos, sin, y, x.stride(0), cos.stride(0), D // 2, BLOCK=triton.next_power_of_2(D // 2))
    return y
'''
_RI_TILED = '''
@triton.jit
def _k(x_ptr, cos_ptr, sin_ptr, y_ptr, xs, cs, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * xs; y_ptr += row * xs; cos_ptr += row * cs; sin_ptr += row * cs
    for off in range(0, H, BLOCK):
        i = off + tl.arange(0, BLOCK); m = i < H
        x1 = tl.load(x_ptr + 2 * i, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + i, mask=m, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + i, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + 2 * i, x1 * c - x2 * s, mask=m)
        tl.store(y_ptr + 2 * i + 1, x2 * c + x1 * s, mask=m)
def run(x, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, cos, sin, y, x.stride(0), cos.stride(0), D // 2, BLOCK=1024)
    return y
'''
_CE_TWOPASS = '''
@triton.jit
def _k(x_ptr, t_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride
    mx = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        mx = tl.maximum(mx, x)
    rmax = tl.max(mx)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        d += tl.where(cols < N, tl.exp(x - rmax), 0.0)
    lse = rmax + tl.log(tl.sum(d))
    t = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + t).to(tl.float32)
    tl.store(y_ptr + row, lse - xt)
def run(x, tgt):
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    _k[(M,)](x, tgt, y, x.stride(0), N, BLOCK=1024)
    return y
'''
_CE_ONLINE = '''
@triton.jit
def _k(x_ptr, t_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride
    m = -float("inf"); s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        bm = tl.max(x)
        nm = tl.maximum(m, bm)
        s = s * tl.exp(m - nm) + tl.sum(tl.where(cols < N, tl.exp(x - nm), 0.0))
        m = nm
    lse = m + tl.log(s)
    t = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + t).to(tl.float32)
    tl.store(y_ptr + row, lse - xt)
def run(x, tgt):
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    _k[(M,)](x, tgt, y, x.stride(0), N, BLOCK=1024)
    return y
'''

STRUCTURES = {
    "softcap_softmax": [_SOFTCAP_WHOLEROW, _SOFTCAP_TWOPASS],
    "rmsnorm_gemma":   [_RG_SCALAR, _RG_WHOLEROW],
    "glu":             [_GLU_FLAT, _GLU_ROW],
    "rope_interleaved": [_RI_STRIDED, _RI_TILED],
    "cross_entropy":   [_CE_TWOPASS, _CE_ONLINE],
    "gelu":          [_flat_act(_GELU_E), _row_act(_GELU_E)],
    "silu":          [_flat_act(_SILU_E), _row_act(_SILU_E)],
    "relu2":         [_flat_act(_RELU2_E), _row_act(_RELU2_E)],
    "bias_gelu":     [_BIASGELU_FLAT],
    "reglu":         [_REGLU_FLAT, _REGLU_ROW],
    "l2norm":        [_L2_WHOLEROW, _L2_SCALAR],
    "log_softmax":   [_LOGSM_WHOLEROW, _LOGSM_TWOPASS],
    "softmax_scale": [_SMSCALE_WHOLEROW, _SMSCALE_TWOPASS],
    "layernorm_gelu": [_LNGELU_SCALAR],
    "add_rmsnorm_rope": [_ARR_SHIFT],
    "dequant_int8":  [_DEQ_ROW, _DEQ_WHOLEROW],
    "qknorm_rope":   [_QKR_SHIFTLOAD, _QKR_HALFSPLIT, _QKR_TILEDCOS],
    "rmsnorm":       [_RMS_TWOPASS_VEC, _RMS_TWOPASS_SCALAR, _RMS_WHOLEROW],
    "softmax":       [_SM_THREEPASS, _SM_WHOLEROW, _SM_ONLINE],
    "swiglu":        [_SW_FLAT, _SW_SIGMOID, _SW_ROW],
    "add_rmsnorm":   [_AR_TWOPASS, _AR_WHOLEROW, _AR_SCALAR],
    "rope":          [_ROPE_HALFSPLIT, _ROPE_SHIFTLOAD, _ROPE_TILED],
    "layernorm":     [_LN_SCALAR, _LN_WHOLEROW, _LN_VEC],
    "add_layernorm": [_ALN_SCALAR, _ALN_WHOLEROW, _ALN_VEC],
    "geglu":         [_GEGLU_FLAT, _GEGLU_ROW, _GEGLU_2D],
}


def knob_variants(src: str):
    """Expand one structure into launch-knob variants (BLOCK / num_warps / num_stages).
    Only touches looped kernels with a literal `BLOCK=1024` launch; whole-row kernels (which
    compute BLOCK from N) get num_warps/num_stages variants only. Returns a list incl. the
    original. The harness filters these to the ones that actually verify."""
    out = [src]
    looped = "BLOCK=1024" in src
    blocks = [256, 512, 2048, 4096] if looped else []
    for b in blocks:
        out.append(src.replace("BLOCK=1024", f"BLOCK={b}"))
    # num_warps / num_stages on the launch call (applies to any structure)
    base_variants = list(out)
    for v in base_variants:
        for nw in (4, 8):
            out.append(re.sub(r"\]\(([^\n]*?)\)\n", rf"](\1, num_warps={nw})\n", v, count=1))
        out.append(re.sub(r"\]\(([^\n]*?)\)\n", r"](\1, num_stages=2)\n", v, count=1))
    return out


def all_candidates(op: str):
    """Every teacher candidate for an op: each distinct structure × its knob variants."""
    cands = []
    for s in STRUCTURES.get(op, []):
        cands.extend(knob_variants(s))
    # dedup identical strings
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


# ---- register the generative fusion-chain grammar's teacher kernels ----------------------
def _register_chain_structures():
    import chains
    for name, _kind, _ref, structs in chains.all_chains():
        if name not in STRUCTURES:
            STRUCTURES[name] = structs
_register_chain_structures()
