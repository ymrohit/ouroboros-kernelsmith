@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); MM = cols < N
    x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    mu = tl.sum(x) / N
    x_hat = tl.where(MM, x - mu, 0.0)
    var = tl.sum(x_hat * x_hat) / N
    x_norm = x_hat * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    ln = x_norm * w + b
    sig = tl.sigmoid(ln) # elementwise sigmoid
    tl.store(y_ptr + cols, sig, mask=MM)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-6, 1.0, 0.0, BLOCK=triton.next_power_of_2(N))
    return y