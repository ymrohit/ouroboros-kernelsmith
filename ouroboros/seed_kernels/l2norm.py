# seed kernel for l2norm (verified teacher structure)
@triton.jit
def _k(x_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x * tl.rsqrt(tl.sum(x * x) + eps), mask=m)
def run(x):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y
