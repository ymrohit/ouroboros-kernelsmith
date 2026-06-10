# NEGATIVE CONTROL: rotary embedding that FORGETS THE NEGATION in rotate_half
# (out[:h] = x1*cos1 + x2*sin1 instead of - x2*sin1). Plausible, compiles, runs, WRONG.
# The harness MUST reject this.
@triton.jit
def _bad_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, stride, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride; cos_ptr += row * stride; sin_ptr += row * stride; y_ptr += row * stride
    h = D // 2
    cols = tl.arange(0, BLOCK); m = cols < h
    x1 = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + cols, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cols, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + h + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, x1 * c1 + x2 * s1, mask=m)        # BUG: should be - x2*s1
    tl.store(y_ptr + h + cols, x2 * c2 + x1 * s2, mask=m)


def run(x, cos, sin):
    M, D = x.shape
    y = torch.empty_like(x)
    _bad_kernel[(M,)](x, cos, sin, y, x.stride(0), D, BLOCK=triton.next_power_of_2(D // 2))
    return y
