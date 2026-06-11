@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < h
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    n = tl.load(x_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.load(w_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x + n * n) / D + eps)
    xw = x * r * w; nw = n * r * v
    cx = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sx = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    cn = tl.load(cos_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, xw * cx - nw * sx, mask=m)
    tl.store(y_ptr + h + cols, nw * cn + xw * sn, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D // 2))
    return y