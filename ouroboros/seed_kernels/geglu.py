# GOLD seed kernel: fused GeGLU activation  gelu_tanh(gate) * up.
# gelu_tanh(z) = 0.5*z*(1+tanh(0.7978845608*(z + 0.044715*z^3))). Flat elementwise grid, fp32.
@triton.jit
def _geglu_kernel(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    a = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    tanh = 2.0 * tl.sigmoid(2.0 * a) - 1.0           # tl.tanh absent in this triton build
    tl.store(y_ptr + cols, (0.5 * g * (1.0 + tanh)) * u, mask=m)


def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _geglu_kernel[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
