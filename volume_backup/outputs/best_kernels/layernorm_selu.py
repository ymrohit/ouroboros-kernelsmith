@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, alpha, lam, BLOCK: tl.constexpr):
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
        # SELU: alpha * lambda * exp(x) - alpha * lambda for x < 0, else x
        # Standard SELU params: alpha = 1.67326324, lambda = 1.0507009873
        # But signature says layernorm_selu, usually implying standard SELU after LN
        # Let's assume standard SELU parameters or passed? Signature doesn't pass alpha/lam.
        # Usually "selu" implies standard parameters.
        # Standard SELU: y = lambda * (x if x >= 0 else alpha * (exp(x) - 1))
        # Wait, standard definition: y = lambda * (x if x >= 0 else alpha * (exp(x) - 1))
        # Or sometimes: y = alpha * (exp(x) - 1) if x < 0 else x.
        # Let's check common fusion patterns. Often it's just the activation.
        # The prompt says "layernorm_selu".
        # I will use standard SELU parameters: alpha=1.67326324, lambda=1.0507009873
        alpha = 1.67326324; lam = 1.0507009873
        selu = tl.where(xhat >= 0, xhat, alpha * (tl.exp(xhat) - 1.0))
        selu = lam * selu
        tl.store(y_ptr + cols, selu, mask=MM)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, 1.67326324, 1.0507009873, BLOCK=4096)
    return y