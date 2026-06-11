@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x)
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        x = tl.where(MM, x - mu, 0.0); v += tl.sum(x * x)
    rrms = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        x = tl.where(MM, x - mu, 0.0)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = x * rrms * w + b
        # mish activation: 2 * sigmoid(2 * xhat) - 1 * xhat ? No, standard mish is x * tanh(softplus(x))
        # Or is it add_layernorm then mish? "add_layernorm_mish" usually means:
        # y = mish(LN(x + residual))
        # Let's assume standard mish: x * tanh(ln(1 + exp(x))) or 2 * sigmoid(2x) * x - x?
        # Standard mish: x * tanh(softplus(x))
        # softplus(x) = log(1 + exp(x))
        # tanh(x) = 2 * sigmoid(2x) - 1
        # mish(x) = x * tanh(log(1 + exp(x)))
        # Let's implement mish properly.
        # tanh(softplus(xhat))
        # softplus = log(1 + exp(xhat))
        # tanh_softplus = tanh(log(1 + exp(xhat)))
        # mish = xhat * tanh_softplus
        # Alternatively, mish(x) = 2 * sigmoid(2 * x) * x - x? No.
        # mish(x) = x * tanh(ln(1 + e^x))
        # Let's use the definition: mish(x) = x * tanh(softplus(x))
        sp = tl.log(1.0 + tl.exp(xhat))
        t = tl.sigmoid(2.0 * sp) * 2.0 - 1.0 # tanh(sp)
        y = xhat * t
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, 1.0, 0.0, BLOCK=4096)
    return y