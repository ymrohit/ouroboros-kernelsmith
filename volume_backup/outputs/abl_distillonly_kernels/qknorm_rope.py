@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; full = tl.arange(0, BLOCK); m = full < D
    x = tl.load(x_ptr + full, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x) / D + eps)
    w = tl.load(w_ptr + full, mask=m, other=0.0).to(tl.float32)
    n = x * r * w
    shifted = tl.where(full < h, full + h, full - h)
    xs = tl.load(x_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    ws = tl.load(w_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(full < h, -1.0, 1.0) * (xs * r * ws)
    cs = tl.load(cos_ptr + full, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + full, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + full, n * cs + rot * sn, mask=m)
def run(x, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D), num_warps=4)
    return y