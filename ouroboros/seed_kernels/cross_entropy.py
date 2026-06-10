# GOLD seed kernel: fused per-row cross-entropy  lse(x) - x[tgt], max-subtracted logsumexp,
# fp32 accumulation. The softmax+log+gather fusion eager does in 3+ launches.
@triton.jit
def _ce_kernel(x_ptr, t_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    m = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, x)
    rmax = tl.max(m)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        d += tl.where(cols < N, tl.exp(x - rmax), 0.0)
    lse = rmax + tl.log(tl.sum(d))
    t = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + t).to(tl.float32)
    tl.store(y_ptr + row, lse - xt)


def run(x, tgt):
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    _ce_kernel[(M,)](x, tgt, y, x.stride(0), N, BLOCK=1024)
    return y
