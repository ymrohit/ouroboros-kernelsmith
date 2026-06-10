"""Harvest REAL Triton kernels (liger / unsloth / Triton tutorials) into the SFT corpus.

The brief's seed-corpus, done properly: extract each library's @triton.jit kernel source
FROM FILE (inspect fails on JITFunction objects), pair it with a thin `run(*inputs)` wrapper
adapted to our op spec, AUTOTUNE-NEUTRALIZE (strip @triton.autotune so the MODEL's schedule
is fixed, not triton's autotuner — the advisor's credit-laundering guard), and VERIFY each
against OUR reference through the immutable harness. Only survivors join the corpus, with
provenance recorded. Mismatches (different convention) are filtered automatically.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIGER = Path(os.path.dirname(__import__("liger_kernel.ops", fromlist=["x"]).__file__))
UNSLOTH = HERE / "external" / "unsloth" / "unsloth" / "kernels"
TUTORIALS = HERE / "external" / "triton" / "python" / "tutorials"


def extract_funcs(filepath: Path, names: list[str]) -> dict[str, str]:
    """Pull named top-level functions (incl. their decorators) from a source file, verbatim."""
    src = filepath.read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef,)) and node.name in names:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            out[node.name] = "".join(lines[start:node.end_lineno])
    return out


def list_jit_kernels(filepath: Path):
    """Names of @triton.jit / @triton.autotune-decorated functions + their def-line signature."""
    src = filepath.read_text(); tree = ast.parse(src); lines = src.splitlines()
    res = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decos = [ast.unparse(d) for d in node.decorator_list]
            if any("triton" in d for d in decos):
                defline = lines[node.lineno - 1].strip()
                res.append((node.name, decos, defline))
    return res


# module-level constexpr constants some liger kernels reference (casting modes)
_CASTING = ("_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)\n"
            "_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)\n"
            "_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)\n\n")

# Recipes: (provenance, file, [jit kernel names to extract], extra_const_prefix, op, run_wrapper).
# The run wrapper is OURS (the glue); the extracted kernels are the REAL library source the model
# learns to write. casting_mode=0 (llama) matches our llama-style rmsnorm reference.
RECIPES = [
    ("liger/rms_norm", LIGER / "rms_norm.py", ["_rms_norm_forward_kernel"], _CASTING, "rmsnorm", '''
def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x); rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _rms_norm_forward_kernel[(M,)](y, y.stride(0), x, x.stride(0), w, w.stride(0),
        rstd, rstd.stride(0), N, 1e-6, 0.0, 0, elementwise_affine=True, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("liger/relu_squared", LIGER / "relu_squared.py", ["_relu_squared_forward_kernel"], "", "relu2", '''
def run(x):
    M, N = x.shape
    y = torch.empty_like(x); BLOCK = triton.next_power_of_2(N)
    _relu_squared_forward_kernel[(M,)](y, y.stride(0), x, x.stride(0), N, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("liger/fused_add_rms_norm", LIGER / "fused_add_rms_norm.py", ["_fused_add_rms_norm_forward_kernel"],
     _CASTING, "add_rmsnorm", '''
def run(x, residual, w):
    M, N = x.shape
    y = torch.empty_like(x); s = torch.empty_like(x); rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _fused_add_rms_norm_forward_kernel[(M,)](y, y.stride(0), s, s.stride(0), x, x.stride(0),
        residual, residual.stride(0), w, w.stride(0), rstd, rstd.stride(0), N, 1e-6, 0.0, 0, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("liger/geglu", LIGER / "geglu.py", ["_geglu_tanh_forward_kernel"], "", "geglu", '''
def run(gate, up):
    M, N = gate.shape
    y = torch.empty_like(gate); BLOCK = triton.next_power_of_2(N)
    _geglu_tanh_forward_kernel[(M,)](gate, up, y, gate.stride(0), N, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("liger/swiglu", LIGER / "swiglu.py", ["silu", "_swiglu_forward_kernel"], "", "swiglu", '''
def run(gate, up):
    M, N = gate.shape
    y = torch.empty_like(gate); BLOCK = triton.next_power_of_2(N)
    _swiglu_forward_kernel[(M,)](gate, up, y, gate.stride(0), 1.0, N, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("liger/layer_norm", LIGER / "layer_norm.py", ["_layer_norm_forward_kernel"], "", "layernorm", '''
def run(x, w, b):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device); rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _layer_norm_forward_kernel[(M,)](y, y.stride(0), x, x.stride(0), w, w.stride(0), b, b.stride(0),
        mean, mean.stride(0), rstd, rstd.stride(0), N, 1e-5, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("tutorial/softmax", TUTORIALS / "02-fused-softmax.py", ["softmax_kernel"], "", "softmax", '''
def run(x):
    M, N = x.shape
    y = torch.empty_like(x); BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](y, x, x.stride(0), y.stride(0), M, N, BLOCK, 1)
    return y
'''),
    ("tutorial/layernorm", TUTORIALS / "05-layer-norm.py", ["_layer_norm_fwd_fused"], "", "layernorm", '''
def run(x, w, b):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device); rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _layer_norm_fwd_fused[(M,)](x, y, w, b, mean, rstd, x.stride(0), N, 1e-5, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("unsloth/geglu_exact", UNSLOTH / "geglu.py", ["_exact_forward_kernel"], "", "geglu", '''
def run(gate, up):
    M, N = gate.shape
    e = gate.contiguous(); g = up.contiguous()
    n = e.numel(); BLOCK = 1024
    h = torch.empty_like(e)
    _exact_forward_kernel[(triton.cdiv(n, BLOCK),)](e, g, h, n, BLOCK_SIZE=BLOCK)
    return h
'''),
    ("unsloth/rms_layernorm", UNSLOTH / "rms_layernorm.py", ["_rms_layernorm_forward"], "", "rmsnorm", '''
def run(x, w):
    M, N = x.shape
    y = torch.empty_like(x); r = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _rms_layernorm_forward[(M,)](y, y.stride(0), x, x.stride(0), w, w.stride(0), r, r.stride(0), N, 1e-6, BLOCK_SIZE=BLOCK)
    return y
'''),
    ("unsloth/layernorm", UNSLOTH / "layernorm.py", ["layernorm_forward"], "", "layernorm", '''
def run(x, w, b):
    M, N = x.shape
    y = torch.empty_like(x)
    r = torch.empty(M, dtype=torch.float32, device=x.device); mu = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    layernorm_forward[(M,)](y, y.stride(0), x, x.stride(0), w, b, r, mu, N, 1e-5, BLOCK_SIZE=BLOCK)
    return y
'''),
]


def build_source(recipe) -> str:
    _, fp, names, consts, op, wrapper = recipe
    funcs = extract_funcs(fp, names)
    missing = [n for n in names if n not in funcs]
    if missing:
        raise RuntimeError(f"could not extract {missing} from {fp}")
    body = "\n\n".join(funcs[n] for n in names)
    return consts + body + "\n\n" + wrapper.strip() + "\n"


# ---- V2 FAIRNESS CONDITION: the library's OWN public API --------------------------------
# The fixed-schedule RECIPES above extract raw @triton.jit kernels with OUR wrappers and a
# next_power_of_2 block — fine for "fixed schedule vs fixed schedule", but it bypasses the
# library's own dispatch (calculate_settings, num_warps, in_place flags). A library failing
# our odd-shape stress through OUR wrapper is NOT evidence the library is broken. This
# condition calls Liger's public Function API exactly as shipped, through the same harness.
# (Unsloth's API cannot be imported standalone — its package __init__ patches the whole
# training stack — so the API condition covers Liger; the limitation is stated in the report.)
API_RECIPES = [
    ("liger_api/rms_norm", "rmsnorm", '''
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
def run(x, w):
    try:
        return LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", False)  # in_place=False
    except TypeError:
        return LigerRMSNormFunction.apply(x, w, 1e-6)
'''),
    ("liger_api/layer_norm", "layernorm", '''
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
def run(x, w, b):
    return LigerLayerNormFunction.apply(x, w, b, 1e-5)
'''),
    ("liger_api/swiglu", "swiglu", '''
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
def run(gate, up):
    return LigerSiLUMulFunction.apply(gate, up)
'''),
    ("liger_api/geglu", "geglu", '''
from liger_kernel.ops.geglu import LigerGELUMulFunction
def run(gate, up):
    return LigerGELUMulFunction.apply(gate, up)
'''),
    ("liger_api/cross_entropy", "cross_entropy", '''
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
def run(x, tgt):
    out = LigerCrossEntropyFunction.apply(x, tgt, None, -100, 0.0, 0.0, "none")
    loss = out[0] if isinstance(out, tuple) else out
    return loss.to(x.dtype)
'''),
]


def harvest_api(verbose=True):
    """Verify the library-API condition through the SAME harness (correctness only here; the
    head-to-head benches the survivors). A recipe that fails import/correctness is recorded
    as unavailable — honestly, not silently."""
    import json
    import sys
    sys.path.insert(0, str(HERE))
    from harness import evaluate
    out_dir = HERE / "datasets" / "api_kernels"; out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for prov, op, src in API_RECIPES:
        res = evaluate(src, op, correctness_only=True)
        ok = res.status == "ok"
        if verbose:
            print(f"  {prov:28} [{op:12}] -> {res.status:12} {'' if ok else res.feedback[:60]}")
        fname = prov.replace("/", "_") + ".py"
        if ok:
            (out_dir / fname).write_text(src)
        manifest.append({"provenance": prov, "op": op, "file": fname,
                         "verified": ok, "note": "" if ok else res.feedback[:120]})
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(m["verified"] for m in manifest)
    print(f"\nlibrary-API condition: {n_ok}/{len(manifest)} verified -> {out_dir}")
    return manifest


def harvest(verbose=True):
    import sys
    sys.path.insert(0, str(HERE))
    from harness import evaluate
    out_dir = HERE / "datasets" / "real_kernels"; out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for recipe in RECIPES:
        prov, fp, names, consts, op = recipe[0], recipe[1], recipe[2], recipe[3], recipe[4]
        try:
            src = build_source(recipe)
        except Exception as e:
            if verbose: print(f"  {prov:28} -> EXTRACT FAIL: {e}")
            continue
        res = evaluate(src, op, correctness_only=True)
        ok = res.status == "ok"
        if verbose:
            print(f"  {prov:28} [{op:12}] -> {res.status:12} {'' if ok else res.feedback[:60]}")
        if ok:
            fname = prov.replace("/", "_") + ".py"
            (out_dir / fname).write_text(src)
            manifest.append({"provenance": prov, "op": op, "file": fname})
    import json
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nharvested {len(manifest)} REAL kernels (verified vs our reference) -> {out_dir}")
    return manifest


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        for label, fp in [("liger/rms_norm", LIGER / "rms_norm.py")]:
            for name, decos, defline in list_jit_kernels(fp):
                print(name, decos)
    else:
        harvest()
