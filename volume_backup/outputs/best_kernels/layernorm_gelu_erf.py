@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x)
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        x = tl.where(MM, x - mu, 0.0); v += tl.sum(x * x)
    rrms = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = (x - mu) * rrms * w + b
        # gelu approximation using erf: 0.5 * x * (1 + erf(sqrt(0.5) * x))
        # but the prompt says "layernorm_gelu_erf", which likely means:
        # apply layernorm, then apply gelu using erf.
        # gelu(x) = 0.5 * x * (1 + erf(sqrt(0.5) * x))
        sqrt2 = tl.full([BLOCK], 0.70710678118654752427048470735387, dtype=tl.float32)
        gelu_x = 0.5 * xhat * (1.0 + tl.sigmoid(2.0 * sqrt2 * (0.7978845608028654 * (xhat + 0.044715 * xhat * xhat * xhat))))
        # The prompt specifically says "erf", so let's use the erf-based formula.
        # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_x = 0.5 * xhat * (1.0 + tl.erf(sqrt2 * xhat))
        tl.store(y_ptr + cols, gelu_x, mask=MM)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, 1.0, 0.0, BLOCK=4096)
    return y