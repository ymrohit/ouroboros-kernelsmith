@triton.jit
def _k(g_ptr, u_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); g_ptr += row * stride; u_ptr += row * stride; y_ptr += row * stride
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, g * tl.sigmoid(g) * u, mask=m)
def run(gate, up):
    M, N = gate.shape; y = torch.empty_like(gate)
    _k[(M,)](gate, up, y, gate.stride(0), N, BLOCK=1024, num_stages=2)
    return y