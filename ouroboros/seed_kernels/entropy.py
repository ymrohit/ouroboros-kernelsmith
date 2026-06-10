# GOLD seed kernel: per-row Shannon entropy of softmax(x) from logits.
# H = lse - sum(x * p):  pass 1 row max; pass 2 sum exp(x-rmax) AND sum x*exp(x-rmax).
# fp32 accumulation throughout.
@triton.jit
def _entropy_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    m = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, x)
    rmax = tl.max(m)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    xe = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.where(mask, tl.exp(x - rmax), 0.0)
        d += e
        xe += tl.where(mask, x * e, 0.0)
    denom = tl.sum(d)
    lse = rmax + tl.log(denom)
    tl.store(y_ptr + row, lse - tl.sum(xe) / denom)


def run(x):
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    _entropy_kernel[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
