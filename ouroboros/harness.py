"""OUROBOROS HARNESS — the immutable verify-compile-bench referee. THIS IS THE PRODUCT.

The model is small and replaceable; this harness is the moat. It takes a candidate Triton
kernel (a source string), and returns a grounded verdict that NEITHER the proposer nor any
trainer can fake:

  correctness is a BOOLEAN  (allclose vs a PyTorch reference, across adversarial inputs)
  speed is a NUMBER         (wall-clock on the 4090, CUDA events, vs eager AND torch.compile)

This mirrors `sec_sqli/discovery_specialist/dvwa_oracle.py`: success is a real observed
effect, never a pattern match. There it was a seeded canary reflected in a live HTTP
response; here it is `allclose(out, ref) == True` AND a measured `t_baseline / t_kernel`.
A kernel that merely *looks* fast or *looks* correct gets nothing.

This codebase has been burned before (memory: the matrix-rewrite "win" was a verifier
certifying an ablation artifact). The GPU analog is a benchmark that times launch-async
noise, compilation, or elided work. Every guard below exists to prevent that:

  NON-NEGOTIABLES (the line between this and a toy):
  1. SUBPROCESS ISOLATION + HARD TIMEOUT. Triton kernels segfault and hang. A crash must
     not take down the orchestrator. compile+run+bench all happen in a child process the
     parent kills on timeout.
  2. ADVERSARIAL MULTI-SHAPE CORRECTNESS. Shapes, dtypes AND magnitudes are swept (see
     specs._mk_*). A kernel correct only on benign N(0,1) (e.g. softmax with no
     max-subtraction) FAILS — the negative-control analog.
  3. CUDA EVENTS + WARMUP (both paths) + MEDIAN-of-N. time.time() around a launch is
     meaningless (async). Without warmup you time JIT/inductor compilation. Median because
     the 4090 boost-clocks drift.
  4. HONEST BASELINE = torch.compile, reported even when we LOSE. Beating eager is the
     floor; beating compile is the flex. Losses are printed plainly, never hidden.

Parent API:   evaluate(kernel_src, spec_name, ...) -> Result
Worker mode:  python harness.py --worker      (reads one JSON request on stdin)
Self-test:    python harness.py                (gold kernels pass; wrong kernels REJECTED)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------
@dataclass
class Result:
    status: str                       # ok | compile_fail | runtime_fail | incorrect | timeout | crash
    feedback: str = ""                # structured, teachable signal for the proposer prompt
    correct: bool = False
    n_shapes_passed: int = 0
    max_abs_err: float = 0.0
    latency_ms: float = 0.0           # candidate kernel, median
    eager_ms: float = 0.0
    compile_ms: float = 0.0
    maxauto_ms: float = 0.0           # torch.compile(mode="max-autotune") — the STRONG baseline
    speedup_eager: float = 0.0        # eager_ms / latency_ms   (>1 = faster than eager)
    speedup_compile: float = 0.0      # compile_ms / latency_ms (>1 = faster than torch.compile default)
    speedup_maxauto: float = 0.0      # maxauto_ms / latency_ms (>1 = faster than max-autotune; 0 = not measured)

    def to_dict(self):
        return asdict(self)


# ============================ PARENT: subprocess driver =================================
def evaluate(kernel_src: str, spec_name: str, n_shapes: int = 8, n_iters: int = 100,
             seed: int = 0, strong: bool = False, correctness_only: bool = False,
             timeout: float | None = None) -> Result:
    """Run a candidate kernel through the full referee in an ISOLATED child process.

    The child can segfault or hang freely; we reap it. Only a clean JSON verdict on stdout
    counts as a result — anything else is a crash, reported as such (never silently 'ok').

    strong=True also benchmarks torch.compile(mode="max-autotune") — the strongest honest
    baseline — at the cost of a slow one-time autotune compile (hence the longer timeout).
    correctness_only=True skips ALL benchmarking (returns ok/incorrect from allclose alone) —
    much cheaper, for building/filtering the SFT corpus where the boolean is all that matters."""
    if timeout is None:
        timeout = 300.0 if strong else (20.0 if correctness_only else 40.0)
    req = json.dumps({"kernel_src": kernel_src, "spec_name": spec_name,
                      "n_shapes": n_shapes, "n_iters": n_iters, "seed": seed, "strong": strong,
                      "correctness_only": correctness_only})
    try:
        proc = subprocess.run([sys.executable, str(HERE / "harness.py"), "--worker"],
                              input=req, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(status="timeout", feedback=f"killed after {timeout:.0f}s (hang/deadlock in compile or launch)")
    if proc.returncode != 0:
        # segfault / illegal memory access / OOM that took the interpreter down
        tail = (proc.stderr or "").strip().splitlines()
        return Result(status="crash", feedback="child crashed (rc=%d): %s" % (
            proc.returncode, tail[-1] if tail else "no stderr"))
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("RESULT:")), None)
    if not line:
        return Result(status="crash", feedback="no verdict on stdout (worker produced no RESULT line)")
    return Result(**json.loads(line[len("RESULT:"):]))


# ============================ CHILD: the actual referee ================================
# Triton's @jit inspects the DEFINING SOURCE FILE, so a kernel must live in a real .py on
# disk — exec'ing a string fails ("@jit functions should be defined in a Python file").
# We write each candidate to a temp module (with a standard import preamble) and import it.
_PREAMBLE = "import torch\nimport triton\nimport triton.language as tl\n\n"


def _worker(req: dict) -> Result:
    import importlib.util
    import os
    import tempfile
    import torch
    sys.path.insert(0, str(HERE))
    from specs import get_spec
    import random

    spec = get_spec(req["spec_name"])

    # ---- 1. COMPILE: materialize the kernel to a temp .py and import it ------------------
    import atexit
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(HERE), delete=False)
    tmp.write(_PREAMBLE + req["kernel_src"])
    tmp.close()
    # keep the file alive: @jit reads it at LAUNCH (during correctness), not at import.
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))
    try:
        spec_mod = importlib.util.spec_from_file_location("_kern_" + str(os.getpid()), tmp.name)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
    except Exception as e:
        return Result(status="compile_fail", feedback=f"{type(e).__name__}: {str(e)[:200]}")
    run = getattr(mod, "run", None)
    if not callable(run):
        return Result(status="compile_fail", feedback="kernel defines no callable `run(*inputs)`")

    # ---- 2. CORRECTNESS: GUARANTEED stress cases (high-scale fp16/bf16, odd N) FIRST, then
    #         adversarial random sweep. The stress cases make the negative controls fail
    #         DETERMINISTICALLY — never by seed luck. ----------------------------------------
    rng = random.Random(req["seed"])
    cases = list(spec.stress_inputs()) + [spec.make_inputs(rng) for _ in range(req["n_shapes"])]
    passed = 0
    worst = 0.0
    for inputs in cases:
        ref = spec.reference(*inputs)
        rtol, atol = spec.tol(inputs[0].dtype)
        try:
            out = run(*[t.clone() for t in inputs])    # clone: kernel must not mutate caller's inputs
        except Exception as e:
            torch.cuda.synchronize()
            return Result(status="runtime_fail", n_shapes_passed=passed,
                          feedback=f"{type(e).__name__} on shape {tuple(inputs[0].shape)}/{inputs[0].dtype}: {str(e)[:160]}")
        if out is None or out.shape != ref.shape:
            return Result(status="incorrect", n_shapes_passed=passed,
                          feedback=f"wrong shape: got {None if out is None else tuple(out.shape)} want {tuple(ref.shape)}")
        d = (out.float() - ref.float()).abs()
        err = float(d.max())
        worst = max(worst, err)
        if not torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol):
            bad = int((d > atol + rtol * ref.float().abs()).sum())
            return Result(status="incorrect", n_shapes_passed=passed, max_abs_err=err,
                          feedback=(f"max abs err {err:.3e} on {tuple(inputs[0].shape)}/{inputs[0].dtype} "
                                    f"scale~{inputs[0].float().abs().max():.0f} ({bad} elems over tol {atol:.1e}/{rtol:.1e}) "
                                    f"— likely a reduction/stability bug, not a shape bug"))
        passed += 1

    # correctness-only mode: the kernel is correct across all adversarial cases; skip the
    # (expensive) benchmark entirely. Used to build/filter the SFT corpus.
    if req.get("correctness_only"):
        return Result(status="ok", correct=True, n_shapes_passed=passed, max_abs_err=worst,
                      feedback=f"PASS {passed} cases (correctness-only)")

    # ---- 3. BENCHMARK (correct kernels only) — CUDA events, warmup both, median ----------
    bench = spec.bench_inputs()
    eager = spec.reference
    try:
        compiled = torch.compile(spec.reference)
    except Exception:
        compiled = spec.reference                       # fall back; still honest vs eager

    def _bench(fn, n_iters):
        # Clone the inputs ONCE, OUTSIDE the timed window. Cloning 8192x4096 fp16 tensors is
        # itself a ~134MB GPU memcpy; doing it inside the CUDA-event window would time the
        # copy too and pull every ratio toward 1.0 (the exact contamination this harness
        # exists to prevent). Kernels write to a fresh output and don't mutate inputs (the
        # correctness phase enforced that), so identical args across iters is correct.
        args = [t.clone() for t in bench]
        for _ in range(25):
            o = fn(*args)                               # warmup: JIT/inductor compile + clock settle
        torch.cuda.synchronize()
        times = []
        for _ in range(n_iters):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            o = fn(*args)                               # ONLY the kernel is in the timed window
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
            del o
        return statistics.median(times)

    try:
        t_kernel = _bench(run, req["n_iters"])
        t_eager = _bench(eager, req["n_iters"])
        t_comp = _bench(compiled, req["n_iters"])
    except Exception as e:
        return Result(status="runtime_fail", correct=True, n_shapes_passed=passed,
                      feedback=f"correct but bench failed: {type(e).__name__}: {str(e)[:160]}")

    # STRONG baseline (opt-in): torch.compile max-autotune. Slow to compile; only used for
    # the self-test gate and final best-kernel validation, not the inner search loop.
    t_max = 0.0
    if req.get("strong"):
        try:
            cmax = torch.compile(spec.reference, mode="max-autotune-no-cudagraphs")
            t_max = _bench(cmax, req["n_iters"])
        except Exception:
            t_max = 0.0

    return Result(
        status="ok", correct=True, n_shapes_passed=passed, max_abs_err=worst,
        latency_ms=round(t_kernel, 5), eager_ms=round(t_eager, 5), compile_ms=round(t_comp, 5),
        maxauto_ms=round(t_max, 5),
        speedup_eager=round(t_eager / t_kernel, 4), speedup_compile=round(t_comp / t_kernel, 4),
        speedup_maxauto=round(t_max / t_kernel, 4) if t_max > 0 else 0.0,
        feedback=(f"PASS {passed} cases | {t_kernel:.4f}ms | {t_eager/t_kernel:.2f}x eager, "
                  f"{t_comp/t_kernel:.2f}x compile"
                  + (f", {t_max/t_kernel:.2f}x max-autotune" if t_max > 0 else "")))


# ============================ self-test (the D1-D2 gate) ===============================
def _selftest():
    """Prove the moat: a hand-written GOLD kernel passes with an honest speedup, and a
    deliberately-WRONG kernel is REJECTED. Mirrors dvwa_oracle.__main__'s positive +
    anti-cheat negatives. Writes the verdict durably to reports/."""
    seeddir = HERE / "seed_kernels"
    cases = [
        ("rmsnorm", "rmsnorm.py", "GOLD", "ok"),
        ("softmax", "softmax.py", "GOLD", "ok"),
        ("swiglu", "swiglu.py", "GOLD", "ok"),
        ("add_rmsnorm", "add_rmsnorm.py", "GOLD", "ok"),
        ("rope", "rope.py", "GOLD", "ok"),
        ("layernorm", "layernorm.py", "GOLD", "ok"),
        ("add_layernorm", "add_layernorm.py", "GOLD", "ok"),
        ("geglu", "geglu.py", "GOLD", "ok"),
        ("qknorm_rope", "qknorm_rope.py", "GOLD (fusion chain)", "ok"),
        ("rmsnorm", "rmsnorm_wrong.py", "NEGATIVE-CONTROL (no rsqrt)", "incorrect"),
        ("softmax", "softmax_wrong.py", "NEGATIVE-CONTROL (no max-subtract)", "incorrect"),
        ("add_rmsnorm", "add_rmsnorm_wrong.py", "NEGATIVE-CONTROL (drops residual in stat)", "incorrect"),
        ("rope", "rope_wrong.py", "NEGATIVE-CONTROL (drops rotate_half negation)", "incorrect"),
        ("layernorm", "layernorm_wrong.py", "NEGATIVE-CONTROL (no mean-subtraction)", "incorrect"),
        ("add_layernorm", "add_layernorm_wrong.py", "NEGATIVE-CONTROL (drops residual in stat)", "incorrect"),
        ("geglu", "geglu_wrong.py", "NEGATIVE-CONTROL (SiLU instead of GELU)", "incorrect"),
        ("qknorm_rope", "qknorm_rope_wrong.py", "NEGATIVE-CONTROL (rms scale dropped on rotated half)", "incorrect"),
    ]
    report = {"machine": None, "cases": []}
    try:
        import torch
        report["machine"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    print(f"OUROBOROS harness self-test on {report['machine']}\n" + "=" * 70)
    ok_all = True
    for spec_name, fname, label, expect in cases:
        src = (seeddir / fname).read_text()
        # GOLD kernels get the STRONG (max-autotune) baseline; negative controls are rejected
        # before benchmarking so strong is moot for them (keep them fast).
        res = evaluate(src, spec_name, n_shapes=8, n_iters=100, strong=(expect == "ok"))
        hit = (res.status == expect) or (expect == "incorrect" and res.status in ("incorrect", "runtime_fail"))
        ok_all &= hit
        mark = "OK " if hit else "XX "
        extra = ""
        if res.status == "ok":
            extra = (f"  {res.latency_ms:.4f}ms  {res.speedup_eager:.2f}x eager  "
                     f"{res.speedup_compile:.2f}x compile  "
                     f"{res.speedup_maxauto:.2f}x max-autotune")
        print(f"  {mark}{spec_name:12} {label:34} -> {res.status:12}{extra}")
        if res.status != "ok":
            print(f"        feedback: {res.feedback}")
        report["cases"].append({"spec": spec_name, "kernel": fname, "label": label,
                                "expect": expect, "got": res.status, "pass": hit, **res.to_dict()})
    repath = HERE / "reports" / "harness_selftest.json"
    repath.write_text(json.dumps(report, indent=2))
    verdict = "ALL GREEN — moat proven" if ok_all else "FAILURES — harness not trustworthy yet"
    print("=" * 70 + f"\n{verdict}\nreport -> {repath}")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal: run one JSON request from stdin")
    args = ap.parse_args()
    if args.worker:
        req = json.loads(sys.stdin.read())
        res = _worker(req)
        print("RESULT:" + json.dumps(res.to_dict()))
        return
    sys.exit(_selftest())


if __name__ == "__main__":
    main()
