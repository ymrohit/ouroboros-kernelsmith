# NEGATIVE CONTROL: fused add+RMSNorm that FORGETS THE RESIDUAL ADD in the normalization
# statistic — it normalizes by rms(x) but outputs (x+residual). Plausible, compiles, runs,
# and is WRONG whenever residual is non-trivial (the stress cases use a large residual).
# The harness MUST reject this.
@triton.jit
def _bad_kernel(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    r_ptr += row * stride
    y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x                       # BUG: stats from x only, residual omitted here
    rms = tl.rsqrt(tl.sum(acc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        h = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32) + \
            tl.load(r_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, h * rms * w, mask=mask)


def run(x, residual, w):
    M, N = x.shape
    y = torch.empty_like(x)
    _bad_kernel[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
