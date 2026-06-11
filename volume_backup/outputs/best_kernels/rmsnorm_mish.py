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
        # mish activation: 2 * sigmoid(2 * xhat) * xhat - xhat
        # or more commonly: x + 2 * sigmoid(2 * x) * x - 2 * x? No, mish(x) = x * tanh(softplus(x))
        # Let's use the standard definition: mish(x) = x * tanh(ln(1 + exp(x)))
        # Or the approximation: 2 * sigmoid(2 * x) * x
        # The prompt says "rmsnorm_mish", which likely means apply mish to the normalized output.
        # Let's assume mish(x) = x * tanh(ln(1 + exp(x)))
        # A common approximation is 2 * sigmoid(2 * x) * x, but let's use the exact form if possible.
        # Triton has tl.sigmoid. mish(x) = x * tanh(softplus(x)).
        # softplus(x) = log(1 + exp(x)).
        # tanh(x) = 2 * sigmoid(2 * x) - 1.
        # So mish(x) = x * (2 * sigmoid(2 * softplus(x)) - 1)
        # Let's implement mish(xhat).
        softplus = tl.log(1.0 + tl.exp(xhat))
        tanh_softplus = 2.0 * tl.sigmoid(2.0 * softplus) - 1.0
        y = xhat * tanh_softplus
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, 1.0, 0.0, BLOCK=4096)
    return y