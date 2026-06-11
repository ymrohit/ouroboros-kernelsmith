
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
def run(gate, up):
    return LigerSiLUMulFunction.apply(gate, up)
