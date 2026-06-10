# NEGATIVE CONTROL: row softmax with NO max-subtraction. Mathematically "equivalent" and
# passes on benign N(0,1) inputs — but OVERFLOWS to inf/nan on large magnitudes (the
# adversarial scale sweep in specs._mk_softmax). The harness MUST reject it. This is the
# exact trap the brief warns about: a kernel that passes one easy regime and is wrong.
@triton.jit
def _bad_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        d += tl.where(cols < N, tl.exp(x), 0.0)        # BUG: exp(x), not exp(x - row_max)
    denom = tl.sum(d)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.exp(x) / denom, mask=mask)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    _bad_kernel[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
