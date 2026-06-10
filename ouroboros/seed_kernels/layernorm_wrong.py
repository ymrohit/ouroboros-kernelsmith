# NEGATIVE CONTROL: "LayerNorm" that OMITS THE MEAN-SUBTRACTION (i.e. it's really RMSNorm
# with a bias) — normalizes by sqrt(mean(x^2)) and never centers. Plausible, compiles, runs,
# WRONG (LayerNorm must subtract the mean). The harness MUST reject this.
@triton.jit
def _bad_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    v = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        v += tl.sum(x * x)               # BUG: mean-square, no mean-subtraction
    r = tl.rsqrt(v / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, x * r * w + b, mask=m)     # BUG: no (x - mu) centering
def run(x, w, b):
    M, N = x.shape; y = torch.empty_like(x)
    _bad_kernel[(M,)](x, w, b, y, x.stride(0), N, 1e-5, BLOCK=1024)
    return y
