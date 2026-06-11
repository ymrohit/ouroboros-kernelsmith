@triton.jit
def _k(x_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=m, other=-float("inf")).to(tl.float32)
    sc = tl.load(s_ptr).to(tl.float32)
    xs = x * sc
    m_x = tl.max(xs)
    d = tl.sum(tl.where(m, tl.exp(xs - m_x), 0.0))
    y = tl.where(m, tl.exp(xs - m_x) / d, 0.0)
    tl.store(y_ptr + row * stride + cols, y, mask=m)
def run(x, scale):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, scale, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return y