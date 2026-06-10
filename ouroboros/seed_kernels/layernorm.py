# GOLD seed kernel: fused LayerNorm. Two reductions (mean, then var about that mean) + affine,
# one fused kernel. fp32 accumulation. Eager launches mean/var/normalize/affine separately.
@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s += tl.sum(x)
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, x - mu, 0.0)
        v += tl.sum(d * d)
    r = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x - mu) * r * w + b, mask=m)


def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _ln_kernel[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
