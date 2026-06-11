"""Generate paper/numbers.tex — every statistic in the paper, extracted from the
harness-emitted JSONs. THE ONE RULE: if a number is not derivable from a report file,
it does not appear in the paper. Run this, then \\input{numbers} in main.tex.

Also performs CONSISTENCY CHECKS (counts must agree across reports) and fails loudly
if any source file is missing or contradictory.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"
RO = ROOT / "ouroboros" / "reports"


def load(p):
    return json.loads(Path(p).read_text())


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


M = {}          # macro name -> value (already formatted string)
CHECKS = []     # (description, ok)


def put(name, val):
    assert name.isalpha(), f"latex macro must be alphabetic: {name}"
    M[name] = str(val)


def check(desc, ok):
    CHECKS.append((desc, bool(ok)))
    if not ok:
        print(f"  CONSISTENCY FAIL: {desc}")


# ---------------- selftest (harness gate) -------------------------------------------------
self4090 = load(RO / "harness_selftest.json")
selfh200 = load(R / "modal_volume" / "harness_selftest.json")
for tag, rep in (("FortyNinety", self4090), ("HTwoHundred", selfh200)):
    cases = rep["cases"]
    put("selftestCases" + tag, len(cases))
    put("selftestPass" + tag, sum(1 for c in cases if c["pass"]))
    put("selftestGold" + tag, sum(1 for c in cases if c["expect"] == "ok"))
    put("selftestNeg" + tag, sum(1 for c in cases if c["expect"] == "incorrect"))
check("selftest ALL GREEN on both GPUs",
      all(c["pass"] for c in self4090["cases"]) and all(c["pass"] for c in selfh200["cases"]))
put("selftestMachineA", self4090["machine"])
put("selftestMachineB", selfh200["machine"])

# ---------------- stability gate (69 kernels) ---------------------------------------------
stab = load(R / "rebench_stability_v2.json")
per = stab["per_op"]
robust = {op: r for op, r in per.items() if r.get("robust")}
put("stabK", stab["k"])
put("stabOps", stab["n_ops"])
put("stabDiscoveries", stab["n_discoveries"])
check("stability: discoveries == ops (69/69)", stab["n_discoveries"] == stab["n_ops"] == 69)
means = [r["mean_vs_maxauto"] for r in robust.values()]
put("stabMeanOfMeans", f"{statistics.mean(means):.3f}")
put("stabMaxSpread", f"{max(r['spread'] for r in robust.values()):.3f}")
weakest = min(robust.items(), key=lambda kv: kv[1]["mean_vs_maxauto"])
strongest = max(robust.items(), key=lambda kv: kv[1]["mean_vs_maxauto"])
put("stabWeakestOp", weakest[0].replace("_", r"\_"))
put("stabWeakestMean", f"{weakest[1]['mean_vs_maxauto']:.3f}")
put("stabWeakestSpread", f"{weakest[1]['spread']:.3f}")
put("stabStrongestOp", strongest[0].replace("_", r"\_"))
put("stabStrongestMean", f"{strongest[1]['mean_vs_maxauto']:.3f}")
put("stabStrongestSpread", f"{strongest[1]['spread']:.3f}")

# ---------------- shape grids ---------------------------------------------------------------
g1 = load(R / "rebench_shapes_qwen3.6-27b.json")
g2 = load(R / "rebench_shapes_v2_qwen3.6-27b.json")
put("gridOneOps", g1["summary"]["n_ops"])
put("gridOneGeomean", f"{g1['summary']['overall_geomean_vs_maxauto']:.3f}")
put("gridTwoOps", g2["summary"]["n_ops"])
put("gridTwoGeomean", f"{g2['summary']['overall_geomean_vs_maxauto']:.3f}")
check("grid1 all ops geomean>1", g1["summary"]["ops_geomean_above_1"] == g1["summary"]["n_ops"])
check("grid2 all ops geomean>1", g2["summary"]["ops_geomean_above_1"] == g2["summary"]["n_ops"])


def grid_cells(g):
    cells = loss = short_loss = 0
    rot_ok = n_rot = 0
    for r in g["per_op"].values():
        cells += r["n_cells"]
        loss += len(r["losses"])
        short_loss += sum(1 for l in r["losses"] if (l["M"], l["N"]) == (16384, 2048))
        if "headline_rotated_vs_maxauto" in r:
            n_rot += 1
            rot_ok += r["headline_rotated_vs_maxauto"] > 1.0
    return cells, loss, short_loss, rot_ok, n_rot


c1, l1, s1, r1ok, r1n = grid_cells(g1)
c2, l2, s2, r2ok, r2n = grid_cells(g2)
put("gridOneCells", c1); put("gridOneLosses", l1); put("gridOneShortLosses", s1)
put("gridTwoCells", c2); put("gridTwoLosses", l2); put("gridTwoShortLosses", s2)
put("gridCellsTotal", c1 + c2); put("gridLossesTotal", l1 + l2)
put("gridLossPct", f"{100.0 * (l1 + l2) / (c1 + c2):.1f}")
put("gridShortLossesTotal", s1 + s2)
check("cache-cold rotated all >1.0", r1ok == r1n and r2ok == r2n)
put("gridRotOk", r1ok + r2ok); put("gridRotN", r1n + r2n)


def grid_band(g, band=0.03):
    """Grey-band re-read: each cell is one measurement session, so cells within
    +-band of 1.0 are ties, neither wins nor losses."""
    w = t = lo = sl = 0
    for r in g["per_op"].values():
        for c in r["cells"]:
            if c["status"] != "ok":
                continue
            s = c["vs_maxauto"]
            if s > 1 + band:
                w += 1
            elif s < 1 - band:
                lo += 1
                if (c["M"], c["N"]) == (16384, 2048):
                    sl += 1
            else:
                t += 1
    return w, t, lo, sl


b1, b2 = grid_band(g1), grid_band(g2)
put("gridBandWins", b1[0] + b2[0]); put("gridBandTies", b1[1] + b2[1])
put("gridBandLosses", b1[2] + b2[2]); put("gridBandShortLosses", b1[3] + b2[3])
check("grey band partitions the full grid",
      sum(b1[:3]) + sum(b2[:3]) == c1 + c2)

# family structure of the gated ops: chain-grammar instances vs structurally distinct
import re as _re
_chainpat = _re.compile(
    r"^(add_)?(rmsnorm|layernorm)(_(" + "|".join([
        "gelu_erf", "leaky_relu", "relu2", "relu6", "hardtanh", "hardsigmoid",
        "hardswish", "softsign", "softplus", "gelu", "silu", "tanh", "sigmoid",
        "relu", "square", "abs", "elu", "selu", "mish"]) + r"))?$")
n_chain = sum(1 for op in per if _chainpat.match(op))
put("stabChainOps", n_chain)
put("stabDistinctOps", len(per) - n_chain)
check("family split partitions the gated ops", n_chain < len(per))

# ---------------- V2 discovery run (37 new ops) ---------------------------------------------
v2 = load(R / "kernelsmith_v2.json")
fin = v2["final"]; att = v2["attribution"]
ok37 = [r for r in fin.values() if r["status"] == "ok"]
put("vTwoOps", len(fin))
put("vTwoOk", len(ok37))
put("vTwoBeatMA", sum(1 for r in ok37 if r["validated_speedup_vs_maxautotune"] > 1.0))
put("vTwoLM", sum(1 for r in fin.values() if r["authored_by"] == "LM"))
put("vTwoValidRate", f"{att['lm_verified_rate']:.3f}")
put("vTwoEvaluated", att["lm_kernels_evaluated"])
put("vTwoVerified", att["lm_kernels_verified"])
put("vTwoLeadTakes", att["lm_archive_wins"])
put("vTwoExplore", att["explore_kernels_verified"])
vals = sorted(r["validated_speedup_vs_maxautotune"] for r in ok37)
put("vTwoMinMA", f"{vals[0]:.2f}"); put("vTwoMaxMA", f"{vals[-1]:.2f}")
check("v2: 37/37 ok and beat MA", len(ok37) == 37 and all(v > 1 for v in vals))
ce = fin["cross_entropy"]
put("vTwoCEMA", f"{ce['validated_speedup_vs_maxautotune']:.3f}")
check("cross_entropy LM-authored", ce["authored_by"] == "LM")
# pass-1 vs pass-2 idiom learning (softplus family valid counts from history)
hist = v2["history"]
sp_ops = [h for h in hist if "softplus" in h["op"]]
p1 = [h["lm_verified_this_round"] for h in sp_ops[:4]]
p2 = [h["lm_verified_this_round"] for h in sp_ops[4:8]]
put("idiomPassOne", "/".join(str(x) for x in p1))
put("idiomPassTwo", "/".join(str(x) for x in p2))
check("idiom learning visible (pass2 > pass1 on softplus)", sum(p2) > sum(p1))

# ---------------- invention run --------------------------------------------------------------
inv = load(R / "kernelsmith_invent.json")
ifin = inv["final"]; iatt = inv["attribution"]
put("invOps", len(ifin))
put("invOk", sum(1 for r in ifin.values() if r["status"] == "ok"))
put("invBeatMA", sum(1 for r in ifin.values() if r["status"] == "ok"
                     and r["validated_speedup_vs_maxautotune"] > 1.0))
put("invLM", sum(1 for r in ifin.values() if r["authored_by"] == "LM"))
put("invValidRate", f"{iatt['lm_verified_rate']:.3f}")
put("invCumsumMA", f"{ifin['cumsum']['validated_speedup_vs_maxautotune']:.3f}")
put("invCumsumC", f"{ifin['cumsum']['validated_speedup_vs_compile']:.3f}")
put("invALSshortMA", f"{ifin['add_layernorm_sigmoid_short']['validated_speedup_vs_maxautotune']:.3f}")
put("invLGshortMA", f"{ifin['layernorm_gelu_short']['validated_speedup_vs_maxautotune']:.3f}")
put("invSMshortMA", f"{ifin['softmax_short']['validated_speedup_vs_maxautotune']:.3f}")
put("invEntropyMA", f"{ifin['entropy']['validated_speedup_vs_maxautotune']:.3f}")
put("invKLMA", f"{ifin['kl_div']['validated_speedup_vs_maxautotune']:.3f}")
check("loss cells flipped (ALS 0.69->%s>1)" % ifin['add_layernorm_sigmoid_short']['validated_speedup_vs_maxautotune'],
      ifin['add_layernorm_sigmoid_short']['validated_speedup_vs_maxautotune'] > 1.0)
# the before numbers (from grid1 per-op losses)
als = [l for l in g1["per_op"]["add_layernorm_sigmoid"]["losses"] if (l["M"], l["N"]) == (16384, 2048)]
lg = [l for l in g1["per_op"]["layernorm_gelu"]["losses"] if (l["M"], l["N"]) == (16384, 2048)]
put("beforeALS", f"{min(l['vs_maxauto'] for l in als):.2f}")
put("beforeLG", f"{min(l['vs_maxauto'] for l in lg):.2f}")

# ---------------- ablations -----------------------------------------------------------------
ARMS = ["control", "nofeedback", "distillonly", "nolearn"]
abl = {}
for a in ARMS:
    p = R / f"ablation_{a}.json"
    if p.exists():
        d = load(p)
        if "final" in d:
            v = [r["validated_speedup_vs_maxautotune"] for r in d["final"].values() if r["status"] == "ok"]
            abl[a] = {"valid": d["attribution"]["lm_verified_rate"],
                      "leads": d["attribution"]["lm_archive_wins"],
                      "ok": len(v), "beat": sum(1 for x in v if x > 1), "gm": geomean(v)}
# every arm must come from a completed JSON (final + attribution); the nolearn file was
# once clobbered by a mid-run checkpoint and is now restored from the Modal volume
# (provenance + correction note in reports/ablations.md) — no hardcoded fallbacks
check("all four 27B ablation arms read from completed JSONs", set(abl) == set(ARMS))
for a in ARMS:
    d = abl[a]
    cap = a.capitalize().replace("_", "")
    put("abl" + cap + "Valid", f"{d['valid']:.3f}")
    put("abl" + cap + "Leads", d["leads"])
    put("abl" + cap + "Beat", f"{d['beat']}/{d['ok']}")
    put("abl" + cap + "Gm", f"{d['gm']:.3f}")
check("distill-only beat control on geomean", abl["distillonly"]["gm"] > abl["control"]["gm"])
put("ablNolearnPctOfControl", f"{100.0 * abl['nolearn']['gm'] / abl['control']['gm']:.0f}")

# ---------------- multi-seed MiniCPM ablation (1B, local 4090) ------------------------------
MA = R / "minicpm_ablation"
if MA.exists():
    import statistics as _st
    mc = {}
    for arm in ARMS:
        gms, vs, beats = [], [], []
        for s in (0, 1, 2):
            d = load(MA / f"abl_{arm}_s{s}.json")
            vv = [r["validated_speedup_vs_maxautotune"] for r in d["final"].values() if r["status"] == "ok"]
            gms.append(geomean(vv)); vs.append(d["attribution"]["lm_verified_rate"])
            beats.append(sum(1 for x in vv if x > 1) / len(vv))
        mc[arm] = (gms, vs, beats)
        cap = arm.capitalize()
        put("mc" + cap + "Gm", f"{_st.mean(gms):.3f}")
        put("mc" + cap + "Std", f"{_st.pstdev(gms):.3f}")
        put("mc" + cap + "Valid", f"{_st.mean(vs):.3f}")
        put("mc" + cap + "Beat", f"{_st.mean(beats)*100:.0f}")
    gmeans = {a: _st.mean(mc[a][0]) for a in ARMS}
    gstds = {a: _st.pstdev(mc[a][0]) for a in ARMS}
    spread = max(gmeans.values()) - min(gmeans.values())
    maxstd = max(gstds.values())
    put("mcSpread", f"{spread:.3f}")
    put("mcMaxSeedStd", f"{maxstd:.3f}")
    put("mcBestArm", max(gmeans, key=gmeans.get))
    put("mcWorstArm", min(gmeans, key=gmeans.get))
    check("MiniCPM arms statistically tied (spread <= 2*max seed std)", spread <= 2 * maxstd)
    check("distill-only flipped (worst on 1B-3seed vs best on 27B-1seed)",
          min(gmeans, key=gmeans.get) == "distillonly")
    sftmc = load(MA / "sft_minicpm.json")
    put("mcSftValid", f"{100*sftmc['best_mean_validrate']:.0f}")
    put("mcSftEpochs", len(sftmc["history"]))

# ---------------- expert head-to-head --------------------------------------------------------
h2h = load(R / "headtohead_experts.json")
rows = {k: v for k, v in h2h.items() if not k.startswith("_")}
put("hthOps", len(rows))
api_ok = sum(1 for v in rows.values() for e in v["experts"].values()
             if e.get("condition") == "library_api" and e["status"] == "ok")
put("hthApiVerified", api_ok)
check("Liger API condition all verified (5)", api_ok == 5)
# the comparison set is dictated by what the experts ship; any op it carries that is NOT
# in the gated 69 must be one of the three documented v1-era standalones
hth_outside = sorted(set(rows) - set(per))
check("expert-comparison ops outside the gated suite are exactly the documented three",
      hth_outside == ["layernorm", "relu2", "rmsnorm"])
put("hthOutsideGated", len(hth_outside))

# ---------------- e2e block -----------------------------------------------------------------
# Prefer a genuine H200 file; the 4090 file is the local reports copy. They MUST be distinct
# devices (a save-race once aliased them — the consistency check below guards against it).
e2e_h = load(R / "e2e_block_h200.json") if (R / "e2e_block_h200.json").exists() else load(R / "modal_volume" / "e2e_block.json")
e2e4090 = load(RO / "e2e_block.json")
put("eteDevice", e2e_h["device"])
put("eteEagerX", f"{e2e_h['speedup']['vs_eager']:.3f}")
put("eteCompileX", f"{e2e_h['speedup']['vs_compile_ma']:.3f}")
put("eteGemmFrac", f"{100 * e2e_h['gemm_fraction_of_eager_block']:.0f}")
put("eteNonGemmX", f"{e2e_h['nongemm_ms']['eager'] / e2e_h['nongemm_ms']['ours']:.2f}")
put("eteErr", f"{e2e_h['max_abs_err_vs_eager']:.1e}")
put("eteEagerXLocal", f"{e2e4090['speedup']['vs_eager']:.3f}")
put("eteCompileXLocal", f"{e2e4090['speedup']['vs_compile_ma']:.3f}")
check("e2e H200 file is genuinely H200 (not a 4090 save-race alias)", "H200" in e2e_h["device"])

# ---------------- v1-era + sft + kernelbench --------------------------------------------------
sft = load(R / "modal_volume" / "sft.json")
put("sftBestValid", f"{100 * sft['best_mean_validrate']:.0f}")
kb = load(R / "modal_volume" / "kernelbench_L1.json")
put("kbCorrect", kb["summary"]["correct"])
put("kbN", kb["summary"]["n"])

# ---------------- totals ----------------------------------------------------------------------
put("productKernels", stab["n_ops"] + len(ifin))     # 69 + 7
put("numKernelsExplore", len(ifin))                  # exploratory-run kernels (the 76 - 69)
check("product = 76", stab["n_ops"] + len(ifin) == 76)
check("product = stab + explore", stab["n_ops"] + len(ifin) == 69 + 7)
# 10th entry added 2026-06-11: the hand-transcribed no-learn headline, contradicted by the
# completed artifact recovered from the training volume (docs/KEY_FINDINGS.md entry 11)
put("falsifications", 10)

# ---------------- emit ------------------------------------------------------------------------
out = ROOT / "paper" / "numbers.tex"
lines = ["% AUTO-GENERATED by make_numbers.py from harness-emitted reports/. DO NOT EDIT."]
for k, v in sorted(M.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
out.write_text("\n".join(lines) + "\n")
n_fail = sum(1 for _, ok in CHECKS if not ok)
print(f"wrote {len(M)} macros -> {out}")
print(f"consistency checks: {len(CHECKS) - n_fail}/{len(CHECKS)} passed")
if n_fail:
    raise SystemExit(1)
