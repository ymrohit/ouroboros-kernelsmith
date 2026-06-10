"""END-TO-END demo (V2): a real transformer MLP block running on OUR kernels.

The single-op benches prove per-kernel wins; this proves they survive composition. A
Qwen-style MLP sub-block — add_rmsnorm -> gate/up GEMMs -> swiglu -> down GEMM — is run
three ways on identical inputs/weights:

    eager     : plain torch ops (the multi-launch baseline)
    compiled  : torch.compile(mode="max-autotune-no-cudagraphs") on the WHOLE block
    ours      : our verified kernels for the non-GEMM ops + the same torch GEMMs

Output is allclose-verified against eager BEFORE anything is timed (no exceptions). CUDA
events, warmup all paths, median-of-N — the same discipline as the harness.

HONEST FRAME, stated up front: the GEMMs dominate this block and are IDENTICAL across
paths, so the block-level win is bounded by the non-GEMM fraction (Amdahl). We report that
fraction and the bound explicitly. The point is not a big number — it is that model-written
kernels compose into a real block and the win survives.

Usage:
  python e2e_block_bench.py                          # seed kernels, 4090-friendly sizes
  python e2e_block_bench.py --kernels outputs/best_kernels --rows 16384
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_PREAMBLE = "import torch\nimport triton\nimport triton.language as tl\n\n"


def load_kernel(path: Path):
    """Materialize a kernel source as a module (Triton @jit needs a real defining file)."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(HERE), delete=False)
    tmp.write(_PREAMBLE + path.read_text())
    tmp.close()
    spec = importlib.util.spec_from_file_location(f"_e2e_{path.stem}", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(tmp.name) if False else None   # keep file: @jit reads source at launch
    return mod.run


def bench(fn, args, n_iters=100, warmup=25):
    for _ in range(warmup):
        o = fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        o = fn(*args)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=str(HERE / "seed_kernels"),
                    help="kernel dir (outputs/best_kernels for the trained product)")
    ap.add_argument("--rows", type=int, default=8192, help="tokens in the batch (M)")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=11008)
    ap.add_argument("--n-iters", type=int, default=100, dest="n_iters")
    ap.add_argument("--out", default=str(HERE / "reports" / "e2e_block.json"))
    args = ap.parse_args()

    kdir = Path(args.kernels)
    k_add_rmsnorm = load_kernel(kdir / "add_rmsnorm.py")
    k_swiglu = load_kernel(kdir / "swiglu.py")
    R, H, F = args.rows, args.hidden, args.ffn
    dt = torch.float16
    g = torch.Generator(device="cuda").manual_seed(7)

    def rnd(*shape, scale=1.0):
        return (torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * scale).to(dt)

    x, res, w = rnd(R, H), rnd(R, H), rnd(H)
    Wg, Wu, Wd = rnd(H, F, scale=H ** -0.5), rnd(H, F, scale=H ** -0.5), rnd(F, H, scale=F ** -0.5)

    def block_eager(x, res, w, Wg, Wu, Wd):
        h = x.float() + res.float()
        rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        hn = (h * rms).to(x.dtype) * w
        gate, up = hn @ Wg, hn @ Wu
        act = torch.nn.functional.silu(gate) * up
        return act @ Wd

    def block_ours(x, res, w, Wg, Wu, Wd):
        hn = k_add_rmsnorm(x, res, w)
        gate, up = hn @ Wg, hn @ Wu
        act = k_swiglu(gate, up)
        return act @ Wd

    def nongemm_eager(x, res, w):
        h = x.float() + res.float()
        rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (h * rms).to(x.dtype) * w

    def nongemm_ours(x, res, w):
        return k_add_rmsnorm(x, res, w)

    block_compiled = torch.compile(block_eager, mode="max-autotune-no-cudagraphs")
    inputs = (x, res, w, Wg, Wu, Wd)

    # ---- CORRECTNESS GATE (before any timing; doctrine) -----------------------------------
    ref = block_eager(*inputs)
    out = block_ours(*inputs)
    err = float((out.float() - ref.float()).abs().max())
    ok = torch.allclose(out.float(), ref.float(), rtol=3e-2, atol=2e-2)
    print(f"[e2e] block output vs eager: max abs err {err:.3e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[e2e] ABORT: composed block is not correct; nothing will be timed.")
        sys.exit(1)

    # ---- TIME the three paths --------------------------------------------------------------
    t_eager, _ = bench(block_eager, inputs, args.n_iters)
    t_comp, _ = bench(block_compiled, inputs, args.n_iters)
    t_ours, _ = bench(block_ours, inputs, args.n_iters)
    t_ng_eager, _ = bench(nongemm_eager, (x, res, w), args.n_iters)
    t_ng_ours, _ = bench(nongemm_ours, (x, res, w), args.n_iters)
    gemm_frac = 1.0 - (t_ng_eager / t_eager)

    toks = lambda t: R / (t / 1e3)
    print(f"\nE2E MLP block | rows={R} hidden={H} ffn={F} fp16 | {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    print(f"  eager              : {t_eager:.4f} ms   ({toks(t_eager)/1e6:.2f}M tok/s)")
    print(f"  torch.compile (MA) : {t_comp:.4f} ms   ({toks(t_comp)/1e6:.2f}M tok/s)")
    print(f"  OURS               : {t_ours:.4f} ms   ({toks(t_ours)/1e6:.2f}M tok/s)")
    print(f"  -> vs eager  {t_eager/t_ours:.3f}x | vs compile-MA {t_comp/t_ours:.3f}x")
    print(f"  non-GEMM (norm) sub-path: eager {t_ng_eager:.4f} ms -> ours {t_ng_ours:.4f} ms "
          f"({t_ng_eager/t_ng_ours:.2f}x)")
    print(f"  HONEST BOUND: GEMMs are ~{gemm_frac:.0%} of the eager block and identical across "
          f"paths;\n  the max possible block-level win from our kernels is "
          f"~{1/ (1 - (t_ng_eager - t_ng_ours)/t_eager):.3f}x (Amdahl).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # ALSO write a device-tagged copy so concurrent containers on different GPUs can never
    # alias each other's e2e result (the 4090 local file once overwrote the H200 one).
    dev_tag = "h200" if "H200" in torch.cuda.get_device_name(0) else (
        "4090" if "4090" in torch.cuda.get_device_name(0) else "other")
    tagged = str(Path(args.out).with_name(f"e2e_block_{dev_tag}.json"))
    payload = json.dumps({
        "device": torch.cuda.get_device_name(0), "rows": R, "hidden": H, "ffn": F,
        "kernels": str(kdir), "max_abs_err_vs_eager": err,
        "block_ms": {"eager": t_eager, "compile_ma": t_comp, "ours": t_ours},
        "speedup": {"vs_eager": round(t_eager / t_ours, 4), "vs_compile_ma": round(t_comp / t_ours, 4)},
        "nongemm_ms": {"eager": t_ng_eager, "ours": t_ng_ours},
        "gemm_fraction_of_eager_block": round(gemm_frac, 4)}, indent=2)
    Path(args.out).write_text(payload)
    Path(tagged).write_text(payload)
    print(f"  report -> {args.out}  (+ device-tagged {tagged})")


if __name__ == "__main__":
    main()
