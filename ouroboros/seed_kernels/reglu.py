# seed kernel for reglu (verified teacher structure)
@triton.jit
def _k(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = cols < n
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, tl.maximum(g, 0.0) * u, mask=m)
def run(gate, up):
    y = torch.empty_like(gate); n = gate.numel()
    _k[(triton.cdiv(n, 1024),)](gate, up, y, n, BLOCK=1024)
    return y
