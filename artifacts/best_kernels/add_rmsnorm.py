@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        acc += h * h
    rr = tl.rsqrt(tl.sum(acc) / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        h = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \
            tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, h * rr * w, mask=m)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=2048, num_warps=8)
    return y