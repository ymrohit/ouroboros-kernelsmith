
from liger_kernel.ops.geglu import LigerGELUMulFunction
def run(gate, up):
    return LigerGELUMulFunction.apply(gate, up)
