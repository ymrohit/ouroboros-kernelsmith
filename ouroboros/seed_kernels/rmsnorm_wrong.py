# NEGATIVE CONTROL: RMSNorm with the rsqrt OMITTED — multiplies by the mean-square
# instead of its reciprocal-sqrt. Looks plausible, compiles, runs, is WRONG on every
# magnitude. The harness MUST reject this (status=incorrect). Anti-cheat analog of
# dvwa_oracle's SQL-ish-but-non-exfiltrating payload.
@triton.jit
def _bad_kernel(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    ms = tl.sum(acc) / N + eps          # BUG: no rsqrt — should be tl.rsqrt(ms)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x * ms * w), mask=mask)


def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x)
    _bad_kernel[(M,)](x, w, y, x.stride(0), N, 1e-6, BLOCK=1024)
    return y
