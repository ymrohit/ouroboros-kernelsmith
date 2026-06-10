
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
def run(x, w):
    try:
        return LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", False)  # in_place=False
    except TypeError:
        return LigerRMSNormFunction.apply(x, w, 1e-6)
