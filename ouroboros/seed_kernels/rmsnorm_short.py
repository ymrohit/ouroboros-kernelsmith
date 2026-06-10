# GOLD seed kernel: fused RMSNorm. One pass: square -> mean -> rsqrt -> scale.
# Accumulates in fp32 (correctness), one row per program. The eager path launches
# pow/mean/rsqrt/mul separately; this fuses them into a single kernel.
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.rsqrt(tl.sum(acc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x * rms * w), mask=mask)


def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = 1024
    _rmsnorm_kernel[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=BLOCK)
    return y
