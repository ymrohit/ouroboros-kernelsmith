@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)  # .to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)


def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x); r = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _rms_layernorm_forward[(M,)](y, y.stride(0), x, x.stride(0), w, w.stride(0), r, r.stride(0), N, 1e-6, BLOCK_SIZE=BLOCK)
    return y
