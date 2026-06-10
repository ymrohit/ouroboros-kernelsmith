@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); MM = cols < N
    x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    mu = tl.sum(x) / N
    x_shifted = tl.where(MM, x - mu, 0.0)
    var = tl.sum(x_shifted * x_shifted) / N
    x_hat = x_shifted * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    ln = x_hat * w + b
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    alpha = 0.7978845608028654 * (ln + 0.044715 * ln * ln * ln)
    tanh_val = 2.0 * tl.sigmoid(2.0 * alpha) - 1.0
    gelu = 0.5 * ln * (1.0 + tanh_val)
    tl.store(y_ptr + cols, gelu, mask=MM)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y