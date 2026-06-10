import torch
import triton
import triton.language as tl

# GOLD seed kernel: fused SwiGLU activation  SiLU(gate) * up.
# Flat elementwise grid; sigmoid in fp32. Eager launches silu then mul separately.
@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, y_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_elem
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    silu = g * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(y_ptr + cols, silu * u, mask=mask)


def run(gate, up):
    y = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _swiglu_kernel[grid](gate, up, y, n, BLOCK=BLOCK)
    return y
