@triton.jit
def _k(x_ptr, tgt_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); x_ptr += row * stride
    t = tl.load(tgt_ptr + row).to(tl.int32)
    max_x = -float("inf")
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=-float("inf")).to(tl.float32)
        max_x = tl.maximum(max_x, tl.max(x))
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); MM = cols < N
        x = tl.load(x_ptr + cols, mask=MM, other=-float("inf")).to(tl.float32)
        x = x - max_x
        s += tl.sum(tl.exp(x))
    x_t = tl.load(x_ptr + t).to(tl.float32)
    ce = max_x + tl.log(s) - x_t
    tl.store(y_ptr + row, ce)
def run(x, tgt):
    M, N = x.shape; y = torch.empty((M,), dtype=x.dtype, device=x.device)
    _k[(M,)](x, tgt, y, x.stride(0), N, BLOCK=4096)
    return y