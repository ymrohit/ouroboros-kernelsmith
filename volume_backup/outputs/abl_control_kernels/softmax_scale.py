# seed kernel for softmax_scale (verified teacher structure)
@triton.jit
def _k(x_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    sc = tl.load(s_ptr).to(tl.float32)
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + cols, mask=m, other=-float("inf")).to(tl.float32) * sc
    x = x - tl.max(x)
    e = tl.where(m, tl.exp(x), 0.0)
    tl.store(y_ptr + cols, e / tl.sum(e), mask=m)
def run(x, scale):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, scale, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
