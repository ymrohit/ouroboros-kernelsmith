# NEGATIVE CONTROL (contract): produces the CORRECT silu values but writes them IN-PLACE
# into its input tensor. The bench loop reuses identical args across iterations, which is
# only sound for kernels that never mutate inputs — so the correctness phase must enforce
# the contract, not assume it. This MUST be REJECTED ("kernel MUTATES its input").
def run(x):
    xf = x.float()
    x.copy_((xf * torch.sigmoid(xf)).to(x.dtype))
    return x
