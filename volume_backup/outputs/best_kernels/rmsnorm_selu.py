@triton.jit
def _k(x_ptr, w_ptr, y_ptr, stride, N, eps, alpha, lmbda, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        s += tl.sum(x * x)
    rrms = tl.rsqrt(s / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=MM, other=0.0).to(tl.float32)
        xhat = x * rrms * w
        y = tl.where(xhat > 0, alpha * (tl.exp(xhat) - 1), alpha * xhat)
        y = lmbda * (y + (1 - alpha) * xhat) # Wait, standard selu is alpha*(exp(x)-1) for x>0, alpha*x for x<=0. Then lmbda * ...? 
        # Let's stick to standard Selu: lmbda * (alpha * (exp(x) - 1) if x > 0 else alpha * x)
        # Actually standard selu is: l * (a * (e^x - 1) if x > 0 else a * x)
        # The prompt says "rmsnorm_selu". Usually Selu is applied after norm.
        # Let's assume standard Selu parameters alpha=1.67326324, lambda=1.0507009873
        # Formula: y = lambda * (alpha * (exp(x) - 1) if x > 0 else alpha * x)
        # But wait, the standard definition is: y = lambda * (alpha * (exp(x) - 1) if x < 0 else x) -- No.
        # Standard: y = lambda * (x if x >= 0 else alpha * (exp(x) - 1))
        # Let's use the standard definition.
        xhat = x * rrms * w
        y = tl.where(xhat >= 0, xhat, alpha * (tl.exp(xhat) - 1))
        y = lmbda * y
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, w):
    M, N = x.shape; y = torch.empty_like(x)
    alpha = 1.6732632423543772848170429916717; lmbda = 1.0507009873554804934193349852946
    _k[(M,)](x, w, y, x.stride(0), N, 1e-6, alpha, lmbda, BLOCK=4096)
    return y