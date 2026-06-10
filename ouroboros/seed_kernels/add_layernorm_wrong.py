# NEGATIVE CONTROL: residual+LayerNorm that computes the normalization stats from x ALONE
# (forgets to include the residual in mean/var) but outputs (x+residual)-centered. Plausible,
# compiles, runs, WRONG whenever residual is non-trivial. The harness MUST reject this.
@triton.jit
def _bad_kernel(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s += tl.sum(x)                   # BUG: stats from x only, residual omitted
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, x - mu, 0.0)
        v += tl.sum(d * d)
    rr = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (h - mu) * rr * w + b, mask=m)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _bad_kernel[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
