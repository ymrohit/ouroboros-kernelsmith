# GOLD seed kernel: row-wise softmax with the REQUIRED max-subtraction (numerical
# stability). Single-block-per-row when N fits BLOCK; otherwise an online two-pass.
# fp32 accumulation. This is the kernel the no-max-subtract negative control violates.
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    # pass 1: row max
    m = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, x)
    row_max = tl.max(m)
    # pass 2: denom
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        d += tl.where(cols < N, tl.exp(x - row_max), 0.0)
    denom = tl.sum(d)
    # pass 3: write
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.exp(x - row_max) / denom, mask=mask)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = 1024
    _softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=BLOCK)
    return y
