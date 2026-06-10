# seed kernel for log_softmax (verified teacher structure)
@triton.jit
def _k(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x)
    lse = tl.log(tl.sum(tl.where(m, tl.exp(x), 0.0)))
    tl.store(y_ptr + cols, x - lse, mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
