# NEGATIVE CONTROL: per-block local cumsum with NO carry across blocks — correct only
# when the whole row fits one block (N <= BLOCK). Deterministically wrong on N=4097/8192.
@triton.jit
def _cumsum_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, tl.cumsum(x, 0), mask=m)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    _cumsum_kernel[(M,)](x, y, x.stride(0), N, BLOCK=1024)
    return y
