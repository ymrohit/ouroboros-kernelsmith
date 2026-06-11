# seed kernel for layernorm_gelu (verified teacher structure)
@triton.jit
def _k(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32))
    mu = s / N
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        d = tl.where(m, tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) - mu, 0.0)
        v += tl.sum(d * d)
    r = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        ln = (x - mu) * r * w + b
        a = 0.7978845608028654 * (ln + 0.044715 * ln * ln * ln)
        tl.store(y_ptr + cols, 0.5 * ln * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0)), mask=m)
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
