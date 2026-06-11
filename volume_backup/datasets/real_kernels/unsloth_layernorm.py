@triton.jit
def layernorm_forward(
    Y,
    Y_row_stride,
    X,
    X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    # According to https://pytorch.org/torchtune/stable/_modules/torchtune/modules/layer_norm.html#Fp32LayerNorm, all modules
    # are in float32!
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)

    mean_X = tl.sum(X_row, axis = 0) / n_cols
    # (X[0] - mean) == -mean so we need to mask it out
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask = mask)


def run(x, w, b):
    M, N = x.shape
    y = torch.empty_like(x)
    r = torch.empty(M, dtype=torch.float32, device=x.device); mu = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    layernorm_forward[(M,)](y, y.stride(0), x, x.stride(0), w, b, r, mu, N, 1e-5, BLOCK_SIZE=BLOCK)
    return y
