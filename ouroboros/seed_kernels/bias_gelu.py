# seed kernel for bias_gelu (verified teacher structure)
@triton.jit
def _k(x_ptr, b_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32) + \
            tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        a = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        tl.store(y_ptr + cols, 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * a) - 1.0)), mask=m)
def run(x, bias):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, bias, y, x.stride(0), N, BLOCK=1024)
    return y
