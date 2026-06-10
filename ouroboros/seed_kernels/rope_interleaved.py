# GOLD seed kernel: GPT-J interleaved RoPE. Pairs (x[2i], x[2i+1]) rotated by (cos[i], sin[i]):
#   out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
#   out[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]
# One program per row; strided pair loads; fp32 math.
@triton.jit
def _rope_inter_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, xs, cs, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * xs
    y_ptr += row * xs
    cos_ptr += row * cs
    sin_ptr += row * cs
    i = tl.arange(0, BLOCK)
    m = i < H
    x1 = tl.load(x_ptr + 2 * i, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + i, mask=m, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + 2 * i, x1 * c - x2 * s, mask=m)
    tl.store(y_ptr + 2 * i + 1, x2 * c + x1 * s, mask=m)


def run(x, cos, sin):
    M, D = x.shape
    y = torch.empty_like(x)
    _rope_inter_kernel[(M,)](x, cos, sin, y, x.stride(0), cos.stride(0), D // 2,
                             BLOCK=triton.next_power_of_2(D // 2))
    return y
