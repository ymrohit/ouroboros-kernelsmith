@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    x = tl.load(x_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=0.0).to(tl.float32) + \
        tl.load(r_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x) / N + eps)
    cols = tl.arange(0, BLOCK); MM = cols < N
    w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    xhat = x * rms * w
    # gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    alpha = 0.7978845608028654 * (xhat + 0.044715 * xhat * xhat * xhat)
    tanh_alpha = 2.0 * tl.sigmoid(2.0 * alpha) - 1.0
    gelu = 0.5 * xhat * (1.0 + tanh_alpha)
    tl.store(y_ptr + cols, gelu, mask=MM)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y