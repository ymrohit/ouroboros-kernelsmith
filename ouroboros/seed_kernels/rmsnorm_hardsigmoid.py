@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        h = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(h * h)
    rr = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        h = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        n = h * rr * w
        tl.store(y_ptr + cols, (tl.minimum(tl.maximum(n + 3.0, 0.0), 6.0) / 6.0), mask=MM)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-06, BLOCK=1024)
    return y
