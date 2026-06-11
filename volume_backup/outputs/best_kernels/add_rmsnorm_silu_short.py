@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    cols = tl.arange(0, BLOCK); MM = cols < N
    x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x) / N + eps)
    w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
    xhat = x * rms * w
    # silu = x * sigmoid(x)
    sigmoid_x = tl.sigmoid(xhat)
    silu = xhat * sigmoid_x
    tl.store(y_ptr + cols, silu, mask=MM)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
    return y