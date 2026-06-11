@triton.jit
def _k(x_ptr, r_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, alpha, lambda_, BLOCK: tl.constexpr):
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
        # selu: alpha * lambda * exp(xhat) - lambda, but typically selu is applied after ln
        # selu(x) = lambda * (x if x > 0 else alpha * (exp(x) - 1))
        # Let's assume standard selu parameters alpha=1.6732, lambda=1.0507
        selu = lambda_ * tl.where(xhat > 0, xhat, alpha * (tl.exp(xhat) - 1.0))
        tl.store(y_ptr + cols, selu, mask=MM)
def run(x, residual, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, b, y, x.stride(0), N, 1e-5, 1.6732, 1.0507, BLOCK=4096, num_warps=8)
    return y