# seed kernel for add_rmsnorm_rope (verified teacher structure)
@triton.jit
def _k(x_ptr, r_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; r_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2; cols = tl.arange(0, BLOCK); m = cols < D
    hh = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(hh * hh) / D + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    n = hh * rms * w
    sh = tl.where(cols < h, cols + h, cols - h)
    hs = tl.load(x_ptr + sh, mask=m, other=0.0).to(tl.float32) + tl.load(r_ptr + sh, mask=m, other=0.0).to(tl.float32)
    ws = tl.load(w_ptr + sh, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(cols < h, -1.0, 1.0) * (hs * rms * ws)
    cs = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, n * cs + rot * sn, mask=m)
def run(x, residual, w, cos, sin):
    M, D = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
