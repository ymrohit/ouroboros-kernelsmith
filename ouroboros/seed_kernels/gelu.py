# GOLD: GELU (tanh approx). Flat elementwise grid, fp32. tanh(a)=2*sigmoid(2a)-1.
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    a = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tl.store(y_ptr + cols, 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0)), mask=m)
def run(x):
    y = torch.empty_like(x); n = x.numel()
    _k[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y
