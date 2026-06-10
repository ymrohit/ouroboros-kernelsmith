"""Stability re-bench: is any model-authored best kernel a REAL discovery (beats max-autotune
reproducibly), or winner's-curse noise? Re-runs the immutable harness (strong=True) K times
per op and reports speedup-vs-max-autotune mean±spread against the noise floor.

The advisor's gate: a discovery = a model-authored kernel that beats torch.compile
max-autotune by a margin clearing run-to-run noise, REPRODUCIBLY. Speed is the only arbiter;
"novel structure" is a descriptor, never the claim. Reuses the validated harness — no new
bench code to get subtly wrong.
"""
from __future__ import annotations
import statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import evaluate

K = 5
BEST = HERE / "outputs" / "best_kernels"
ops = [p.stem for p in sorted(BEST.glob("*.py"))]
print(f"stability re-bench: {K}x per op vs max-autotune (the incumbent autotuner)\n" + "=" * 72)
for op in ops:
    src = (BEST / f"{op}.py").read_text()
    maxauto, comp = [], []
    for _ in range(K):
        r = evaluate(src, op, strong=True)          # full isolated harness run incl. max-autotune
        if r.status != "ok":
            print(f"  {op:12} -> {r.status}: {r.feedback[:60]}"); break
        maxauto.append(r.speedup_maxauto); comp.append(r.speedup_compile)
    if len(maxauto) < K:
        continue
    m, s = statistics.mean(maxauto), statistics.pstdev(maxauto)
    cm = statistics.mean(comp)
    robust = (m - s) > 1.0                           # win clears the spread
    verdict = "DISCOVERY (beats max-autotune, robust)" if robust else (
              "ties/loses max-autotune (no discovery)" if m < 1.0 + 1e-9 else "marginal (within noise)")
    print(f"  {op:12} vs max-autotune: mean {m:.3f} ± {s:.3f}  (min {min(maxauto):.3f}, "
          f"max {max(maxauto):.3f}) | vs compile {cm:.3f} | {verdict}")
    print(f"               raw: {[round(x,3) for x in maxauto]}")
print("=" * 72)
print("Discovery bar: mean - spread > 1.0 vs max-autotune, i.e. the model beat the incumbent\n"
      "scheduling search reproducibly. Anything ~1.0 = ceiling on these bandwidth-bound ops.")
