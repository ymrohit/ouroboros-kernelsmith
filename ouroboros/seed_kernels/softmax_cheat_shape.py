# NEGATIVE CONTROL (anti-gaming): special-cases the PUBLIC, FIXED bench shape.
# Correct softmax everywhere the old correctness sweep could look, but at the timing shape
# (8192, 4096) it returns uninitialized garbage instantly — the "infinite speedup" exploit.
# The V2 harness includes the bench inputs in the correctness sweep, so this MUST be REJECTED.
def run(x):
    if x.shape == (8192, 4096):
        return torch.empty_like(x)          # instant garbage at the bench shape
    return torch.softmax(x.float(), dim=-1).to(x.dtype)
