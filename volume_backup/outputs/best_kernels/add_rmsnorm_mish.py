@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, alpha, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; r_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rrms = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32) + tl.load(r_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = x * rrms * w
        y = alpha * xhat * tl.sigmoid(2.0 * alpha * 0.5 * xhat) # mish = x * sigmoid(2*0.5*x) * alpha? No, standard mish is x * sigmoid(2*0.5*x). But signature says add_rmsnorm_mish. Usually mish(x) = x * tanh(softplus(x)). Or x * sigmoid(2x) - wait.
        # Standard mish: x * tanh(softplus(x)) = x * tanh(ln(1+exp(x)))
        # Approx: x * sigmoid(2 * softplus(x))? No.
        # Let's assume standard mish: x * sigmoid(2*x) is NOT mish.
        # Mish: x * tanh(ln(1 + e^x))
        # Triton has tl.sigmoid. tanh(x) = 2*sigmoid(2x) - 1.
        # softplus(x) = ln(1+e^x).
        # mish(x) = x * tanh(softplus(x))
        # Let's implement mish properly.
        sp = tl.log(1.0 + tl.exp(xhat)) # softplus
        y = xhat * (2.0 * tl.sigmoid(2.0 * sp) - 1.0) # tanh(sp) = 2*sigmoid(2*sp) - 1
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, 1.0, BLOCK=4096)
    return y