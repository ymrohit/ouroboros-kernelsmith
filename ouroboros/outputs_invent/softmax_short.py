@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    cols = tl.arange(0, BLOCK); MM = cols < N
    x = tl.load(x_ptr + cols, mask=MM, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x)
    e = tl.exp(x)
    y = e / tl.sum(e)
    tl.store(y_ptr + cols, y, mask=MM)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y