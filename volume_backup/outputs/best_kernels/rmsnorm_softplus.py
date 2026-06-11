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
        # Softplus: alpha * ln(1 + exp(beta * xhat))
        # Standard softplus is ln(1 + exp(x)). Here we assume standard softplus on xhat
        # The problem says "rmsnorm_softplus", implying RMSNorm then Softplus.
        # Let's assume standard softplus: ln(1 + exp(xhat))
        # Or maybe alpha * softplus(beta * xhat)?
        # Given the signature def run(x, w), no extra params.
        # Usually softplus is beta * x + (1 - beta) * ln(1 + exp(20 * beta * x)) for numerical stability?
        # Or just ln(1 + exp(x)).
        # Let's assume standard softplus: ln(1 + exp(xhat))
        # Wait, the prompt says "generative fusion chain ... accumulate the reduction in fp32".
        # And "rmsnorm_softplus".
        # Let's look at the signature: def run(x, w).
        # It doesn't pass alpha, beta, eps.
        # So we must use defaults.
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * w
        # Softplus: ln(1 + exp(x_norm))
        # Let's assume eps=1e-6.
        y = tl.log(1.0 + tl.exp(xhat))
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, 1.0, 1.0, BLOCK=4096, num_warps=4)
    return y