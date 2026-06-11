@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, alpha, lambda_, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rms = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = x * rms * w
        # SELU: alpha * lambda * exp(xhat) - lambda
        # Standard SELU: lambda * (xhat if xhat > 0 else alpha * (exp(xhat) - 1))
        # Where lambda = 1.0507009873554804934193349852946, alpha = 1.6732632423543772848170429916717
        # Let's assume standard SELU params if not passed, but signature doesn't show them.
        # Wait, signature is def run(x, residual, w). No alpha/lambda args.
        # I must use standard SELU constants.
        # SELU(x) = lambda * x if x > 0 else lambda * alpha * (exp(x) - 1)
        # Actually, standard definition: selu(x) = lambda * x for x > 0, lambda * alpha * (exp(x) - 1) for x <= 0.
        # Common values: alpha = 1.67326324, lambda = 1.0507009873554804934193349852946
        # Let's use these.
        l = 1.0507009873554804934193349852946
        a = 1.6732632423543772848170429916717
        y = tl.where(xhat > 0, l * xhat, l * a * (tl.exp(xhat) - 1.0))
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, 0.0, 0.0, BLOCK=4096, num_warps=8)
    return y