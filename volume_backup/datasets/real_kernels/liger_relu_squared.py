@triton.jit
def _relu_squared_forward_kernel(
    Y_ptr,
    Y_stride,
    X_ptr,
    X_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_stride
    Y_ptr += row_idx * Y_stride

    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    # relu(x) = max(0, x), then square
    relu_x = tl.maximum(x_row, 0)
    y_row = relu_x * relu_x

    tl.store(Y_ptr + col_offsets, y_row, mask=mask)


def run(x):
    M, N = x.shape
    y = torch.empty_like(x); BLOCK = triton.next_power_of_2(N)
    _relu_squared_forward_kernel[(M,)](y, y.stride(0), x, x.stride(0), N, BLOCK_SIZE=BLOCK)
    return y
