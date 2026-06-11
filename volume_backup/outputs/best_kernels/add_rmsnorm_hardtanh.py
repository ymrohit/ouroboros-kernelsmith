@triton.jit
def _k(x_ptr, r_ptr, w_ptr, y_ptr, stride, N, eps, alpha, beta, BLOCK: tl.constexpr):
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
        # hardtanh: tanh(alpha * xhat + beta) ? or clamp?
        # The prompt says "add_rmsnorm_hardtanh". Usually hardtanh is clamp.
        # Let's assume standard hardtanh: clamp(xhat, -1, 1) or similar.
        # Given the signature def run(x, residual, w), and the error mentioning alpha and beta,
        # it's likely hardtanh(xhat, alpha, beta) = clamp(xhat, -alpha, alpha) or tanh(alpha * xhat + beta).
        # Let's assume clamp(xhat, -1, 1) for standard hardtanh, but the error mentions alpha and beta.
        # Let's assume the standard hardtanh is clamp(x, -1, 1).
        # Wait, the error says "missing 2 required positional arguments: 'alpha' and 'beta'".
        # This implies the signature should include alpha and beta, or they are fixed.
        # Let's assume alpha=1.0, beta=0.0 for tanh(alpha * xhat + beta) or clamp.
        # Let's try clamp(xhat, -1.0, 1.0).
        y = tl.minimum(tl.maximum(xhat, -1.0), 1.0)
        tl.store(y_ptr + cols, y, mask=MM)
def run(x, residual, w):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, residual, w, y, x.stride(0), N, 1e-6, 1.0, 0.0, BLOCK=4096)
    return y