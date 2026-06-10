# NEGATIVE CONTROL: uses SiLU (x*sigmoid(x)) instead of GELU — that is SwiGLU, not GeGLU.
# Compiles, runs, but the activation is the wrong function. The harness MUST reject this.
@triton.jit
def _bad_kernel(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    act = g * (1.0 / (1.0 + tl.exp(-g)))          # BUG: SiLU, not GELU
    tl.store(y_ptr + cols, act * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _bad_kernel[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
