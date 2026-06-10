# NEGATIVE CONTROL: SiLU instead of GELU. Wrong activation; harness MUST reject.
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x * tl.sigmoid(x), mask=m)   # BUG: SiLU, not GELU
def run(x):
    y = torch.empty_like(x); n = x.numel()
    _k[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y
