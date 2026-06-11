# GOLD fused kernel: QK-norm + RoPE in ONE pass.  n = rmsnorm(x)*w ; out = n*cos + rot_half(n)*sin.
# The rms scale `r` is a per-row scalar shared across the whole head, so the rotated n is just
# x[shifted]*r*w[shifted] — no intermediate `n` is ever written to DRAM (the fusion win).
@triton.jit
def _k(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2
    cols = tl.arange(0, BLOCK); m = cols < D
    x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x) / D + eps)            # rmsnorm scale (scalar over the head)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    n = x * r * w
    shifted = tl.where(cols < h, cols + h, cols - h)
    xs = tl.load(x_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    ws = tl.load(w_ptr + shifted, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(cols < h, -1.0, 1.0) * (xs * r * ws)
    cs = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, n * cs + rot * sn, mask=m)


def run(x, w, cos, sin):
    M, D = x.shape
    y = torch.empty_like(x)
    _k[(M,)](x, w, cos, sin, y, x.stride(0), D, 1e-6, BLOCK=triton.next_power_of_2(D))
    return y
