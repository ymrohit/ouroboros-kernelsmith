# NEGATIVE CONTROL: plain softmax with NO logit softcap. Indistinguishable on small
# benign inputs; at the stress magnitudes (x64) the cap saturates and this diverges.
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, 0))
    e = tl.where(mask, e, 0.0)
    tl.store(y_ptr + cols, e / tl.sum(e, 0), mask=mask)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    _softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
