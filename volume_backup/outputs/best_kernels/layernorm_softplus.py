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
        # Softplus: log(1 + exp(alpha * xhat + beta))
        # Or just softplus(xhat) = log(1 + exp(xhat))
        # The prompt says "layernorm_softplus", implying softplus of the layernorm output.
        # Standard softplus is log(1 + exp(x)).
        # Let's assume alpha=1, beta=0 for standard softplus unless specified otherwise.
        # The signature is run(x, w, b). No alpha/beta passed.
        # So softplus(y_ln) = log(1 + exp(y_ln))
        ln_out = xhat
        # Softplus
        y = tl.log(1.0 + tl.exp(ln_out))
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, 1.0, 0.0, BLOCK=4096)
    return y