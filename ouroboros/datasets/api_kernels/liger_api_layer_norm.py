
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
def run(x, w, b):
    return LigerLayerNormFunction.apply(x, w, b, 1e-5)
