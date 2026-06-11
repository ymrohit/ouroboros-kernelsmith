@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2
    full = tl.load(x_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < D, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(full * full) / D + eps)
    c = tl.arange(0, BLOCK); m = c < h
    x1 = tl.load(x_ptr + c, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + h + c, mask=m, other=0.0).to(tl.float32)
    cc = tl.load(cos_ptr + c, mask=m, other=0.0).to(tl.float32)
    ss = tl.load(sin_ptr + c, mask=m, other=0.0).to(tl.float32)
    n1 = x1 * r * w1; n2 = x2 * r * w2
    tl.store(y_ptr + c, n1 * cc - n2 * ss, mask=m)
    tl.store(y_ptr + h + c, n2 * cc + n1 * ss, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D), num_stages=2)
    return y