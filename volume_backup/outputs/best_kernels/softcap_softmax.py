# GOLD seed kernel: Gemma2-style softcapped softmax  softmax(30*tanh(x/30)).
# Cap FIRST, then the max-subtracted softmax, all fp32. tanh via 2*sigmoid(2z)-1.
@triton.jit
def _softcap_softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    t = x / 30.0
    c = 30.0 * (2.0 * tl.sigmoid(2.0 * t) - 1.0)
    c = tl.where(mask, c, -float("inf"))
    e = tl.exp(c - tl.max(c, 0))
    e = tl.where(mask, e, 0.0)
    tl.store(y_ptr + cols, e / tl.sum(e, 0), mask=mask)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    _softcap_softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
