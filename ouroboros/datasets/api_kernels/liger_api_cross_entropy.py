
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
def run(x, tgt):
    out = LigerCrossEntropyFunction.apply(x, tgt, None, -100, 0.0, 0.0, "none")
    loss = out[0] if isinstance(out, tuple) else out
    return loss.to(x.dtype)
