@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rms = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = x * rms * w
        # gelu approximation using erf: 0.5 * x * (1 + erf(sqrt(0.5) * x))
        # but the prompt says "rmsnorm_gelu_erf", which likely means:
        # y = 0.5 * xhat * (1 + erf(sqrt(0.5) * xhat))
        # Let's assume standard GELU approximation via erf.
        sqrt2 = 0.70710678118654752440
        gelu = 0.5 * xhat * (1.0 + tl.sigmoid(2.0 * sqrt2 * (xhat + 0.044715 * xhat * xhat * xhat)))
        # Wait, the prompt says "erf", so let's use the erf formula directly.
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu = 0.5 * xhat * (1.0 + tl.erf(sqrt2 * xhat))
        tl.store(y_ptr + cols, gelu, mask=MM)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, 1.0, 0.0, BLOCK=4096)
    return y