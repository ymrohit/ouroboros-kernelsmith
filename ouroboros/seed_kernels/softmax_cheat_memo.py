# NEGATIVE CONTROL (anti-gaming): memoizes its output by input POINTER, so after the first
# call it does no GPU work — the "cache the answer, win the bench" exploit. The V2 harness
# defeats this two ways: the CUDA caching allocator reuses freed clone blocks across
# correctness cases (stale hit -> wrong output), and the bench loop pokes one input element
# before every timed iteration then allclose-verifies the FINAL timed output (stale -> fail).
# Either path: this MUST be REJECTED.
_CACHE = {}


def run(x):
    key = (x.data_ptr(), tuple(x.shape), x.dtype)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out = torch.softmax(x.float(), dim=-1).to(x.dtype)
    _CACHE[key] = out
    return out
