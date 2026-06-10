# NEGATIVE CONTROL: KL with the SECOND logsumexp not max-subtracted — overflows on y at
# stress magnitudes while looking plausible on benign inputs.
@triton.jit
def _kl_kernel(x_ptr, y_ptr, o_ptr, sx, sy, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * sx
    y_ptr += row * sy
    mx = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
        mx = tl.maximum(mx, x)
    rx = tl.max(mx)
    dx = tl.zeros([BLOCK], dtype=tl.float32)
    dy = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.load(y_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        dx += tl.where(mask, tl.exp(x - rx), 0.0)
        dy += tl.where(mask, tl.exp(y), 0.0)
    lse_x = rx + tl.log(tl.sum(dx))
    lse_y = tl.log(tl.sum(dy))
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.load(y_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        lx = x - lse_x
        ly = y - lse_y
        acc += tl.where(mask, tl.exp(lx) * (lx - ly), 0.0)
    tl.store(o_ptr + row, tl.sum(acc))


def run(x, y):
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    _kl_kernel[(M,)](x, y, o, x.stride(0), y.stride(0), N, BLOCK=1024)
    return o
