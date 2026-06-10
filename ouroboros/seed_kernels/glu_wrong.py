# NEGATIVE CONTROL: SiLU(gate)*up instead of sigmoid(gate)*up (the swiglu/glu mixup).
@triton.jit
def _glu_kernel(g_ptr, u_ptr, y_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_elem
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + cols, g * tl.sigmoid(g) * u, mask=mask)


def run(gate, up):
    y = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    _glu_kernel[(triton.cdiv(n, BLOCK),)](gate, up, y, n, BLOCK=BLOCK)
    return y
