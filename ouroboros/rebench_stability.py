"""Stability re-bench: is any model-authored best kernel a REAL discovery (beats max-autotune
reproducibly), or winner's-curse noise? Re-runs the immutable harness (strong=True) K times
per op and reports speedup-vs-max-autotune mean±spread against the noise floor.

The advisor's gate: a discovery = a model-authored kernel that beats torch.compile
max-autotune by a margin clearing run-to-run noise, REPRODUCIBLY. Speed is the only arbiter;
"novel structure" is a descriptor, never the claim. Reuses the validated harness — no new
bench code to get subtly wrong.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import evaluate

K = 5
BEST = HERE / "outputs" / "best_kernels"
OUT = HERE / "reports" / "rebench_stability.json"
ops = [o for o in (sys.argv[sys.argv.index("--ops") + 1].split(",")
                   if "--ops" in sys.argv else [p.stem for p in sorted(BEST.glob("*.py"))])]
print(f"stability re-bench: {K}x per op vs max-autotune (the incumbent autotuner)\n" + "=" * 72)
report = {}
for op in ops:
    src = (BEST / f"{op}.py").read_text()
    maxauto, comp = [], []
    fail = None
    for _ in range(K):
        r = evaluate(src, op, strong=True)          # full isolated harness run incl. max-autotune
        if r.status != "ok":
            fail = f"{r.status}: {r.feedback[:80]}"
            print(f"  {op:12} -> {fail}"); break
        maxauto.append(r.speedup_maxauto); comp.append(r.speedup_compile)
    if len(maxauto) < K:
        report[op] = {"verdict": "FAILED", "error": fail, "raw_vs_maxauto": maxauto}
        continue
    m, s = statistics.mean(maxauto), statistics.pstdev(maxauto)
    cm = statistics.mean(comp)
    robust = (m - s) > 1.0                           # win clears the spread
    verdict = "DISCOVERY (beats max-autotune, robust)" if robust else (
              "ties/loses max-autotune (no discovery)" if m < 1.0 + 1e-9 else "marginal (within noise)")
    print(f"  {op:12} vs max-autotune: mean {m:.3f} ± {s:.3f}  (min {min(maxauto):.3f}, "
          f"max {max(maxauto):.3f}) | vs compile {cm:.3f} | {verdict}")
    print(f"               raw: {[round(x,3) for x in maxauto]}")
    report[op] = {"verdict": verdict, "mean_vs_maxauto": round(m, 4), "spread": round(s, 4),
                  "min": round(min(maxauto), 4), "max": round(max(maxauto), 4),
                  "mean_vs_compile": round(cm, 4), "raw_vs_maxauto": [round(x, 4) for x in maxauto],
                  "robust": robust}
n_disc = sum(1 for r in report.values() if r.get("robust"))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"k": K, "discovery_bar": "mean - pstdev > 1.0 vs max-autotune",
                           "n_ops": len(report), "n_discoveries": n_disc,
                           "per_op": report}, indent=2))
print("=" * 72)
print(f"{n_disc}/{len(report)} DISCOVERIES (mean - spread > 1.0 vs max-autotune) | report -> {OUT}")
print("Discovery bar: the model beat the incumbent scheduling search reproducibly.\n"
      "Anything ~1.0 = ceiling on these bandwidth-bound ops. Failures recorded, never hidden.")
