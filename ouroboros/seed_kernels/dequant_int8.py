# seed kernel for dequant_int8 (verified teacher structure)
@triton.jit
def _k(q_ptr, s_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); q_ptr += row * stride; y_ptr += row * stride
    sc = tl.load(s_ptr + row).to(tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK); m = cols < N
        q = tl.load(q_ptr + cols, mask=m, other=0).to(tl.float32)
        tl.store(y_ptr + cols, q * sc, mask=m)
def run(q, scale):
    M, N = q.shape
    y = torch.empty((M, N), device=q.device, dtype=torch.float16)
    _k[(M,)](q, scale, y, q.stride(0), N, BLOCK=1024)
    return y
