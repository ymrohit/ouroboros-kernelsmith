# GOLD seed kernel: row-wise inclusive prefix sum (SCAN). tl.cumsum within each block,
# scalar carry propagated across blocks. fp32 accumulation.
@triton.jit
def _cumsum_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    carry = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        c = tl.cumsum(x, 0) + carry
        tl.store(y_ptr + cols, c, mask=m)
        carry += tl.sum(x)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    _cumsum_kernel[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
