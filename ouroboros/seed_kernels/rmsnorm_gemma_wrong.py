# NEGATIVE CONTROL: drops the Gemma (1 + w) — scales by w like vanilla RMSNorm.
# The exact silent bug that shipped in early Gemma ports. MUST be REJECTED.
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rr = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * rr * w, mask=mask)


def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x)
    _rmsnorm_kernel[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
