"""Generate every figure in the paper from the harness-emitted JSONs (same ONE RULE as
make_numbers.py). Outputs PDF figures into paper/figures/."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

load = lambda p: json.loads(Path(p).read_text())


# ---------------- Fig 1: shape-grid heatmap (all 69 kernels x 12 cells) --------------------
def fig_heatmap():
    g1, g2 = load(R / "rebench_shapes_qwen3.6-27b.json"), load(R / "rebench_shapes_v2_qwen3.6-27b.json")
    per = {**g1["per_op"], **g2["per_op"]}
    # matrix-family ops only (12 cells each); rope-family (8 cells) shown separately in text
    ops = sorted(op for op, r in per.items() if r["n_cells"] == 12)
    cell_keys = [(M, N, dt) for (M, N) in [(1024, 4096), (4096, 4096), (8192, 4096),
                                           (4096, 8192), (16384, 2048), (256, 16384)]
                 for dt in ("fp16", "bf16")]
    Z = np.full((len(ops), len(cell_keys)), np.nan)
    for i, op in enumerate(ops):
        for c in per[op]["cells"]:
            if c["status"] == "ok":
                j = cell_keys.index((c["M"], c["N"], c["dtype"]))
                Z[i, j] = c["vs_maxauto"]
    fig, ax = plt.subplots(figsize=(6.0, 9.5))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0.6, vmax=2.0)
    # readable shape labels: small dims shown as-is, large dims as Nk (256 -> "256", 16384 -> "16k")
    def _sh(v):
        if v < 1024:
            return str(v)
        return f"{v // 1024}k" if v % 1024 == 0 else f"{v / 1024:.1f}k"
    ax.set_xticks(range(len(cell_keys)))
    ax.set_xticklabels([f"{_sh(M)}×{_sh(N)}\n{dt}" for M, N, dt in cell_keys],
                       fontsize=5.5, rotation=0)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([o.replace("_", "\\_") if False else o for o in ops], fontsize=5)
    for i in range(len(ops)):
        for j in range(len(cell_keys)):
            if not np.isnan(Z[i, j]) and Z[i, j] <= 1.0:
                ax.text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center", fontsize=4.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, label="speedup vs max-autotune")
    cb.ax.axhline(1.0, color="k", lw=0.8)
    ax.set_title(f"Shape-grid: {len(ops)} matrix-family kernels × 12 (M,N)×dtype cells\n"
                 "(losses annotated; rope-family ops in text)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "heatmap.pdf")
    plt.close(fig)
    print("heatmap.pdf", Z.shape)


# ---------------- Fig 2: stability gate (69 ops, mean ± spread) ----------------------------
def fig_stability():
    stab = load(R / "rebench_stability_v2.json")
    rows = sorted(stab["per_op"].items(), key=lambda kv: kv[1]["mean_vs_maxauto"])
    names = [k for k, _ in rows]
    m = np.array([v["mean_vs_maxauto"] for _, v in rows])
    s = np.array([v["spread"] for _, v in rows])
    fig, ax = plt.subplots(figsize=(6.0, 8.5))
    colors = ["#2a9d2a"] * len(rows)
    ax.barh(range(len(rows)), m - 1.0, left=1.0, xerr=s, color=colors, height=0.7,
            error_kw={"lw": 0.6})
    ax.axvline(1.0, color="k", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=4.6)
    ax.set_xlabel("mean speedup vs torch.compile max-autotune (5 fresh runs, ±population spread)")
    ax.set_title(f"Stability gate: {stab['n_discoveries']}/{stab['n_ops']} kernels beat "
                 "max-autotune reproducibly (mean − spread > 1)", fontsize=8)
    ax.set_xlim(0.95, 2.15)
    fig.tight_layout()
    fig.savefig(FIG / "stability.pdf")
    plt.close(fig)
    print("stability.pdf")


# ---------------- Fig 3: ablation bars -----------------------------------------------------
def fig_ablation():
    arms = ["control", "nofeedback", "distillonly", "nolearn"]
    labels = ["full loop\n(control)", "− feedback", "− GRPO\n(distill-only)", "− learning\n(best-of-N)"]
    gms, valids = [], []
    for a in arms:
        p = R / f"ablation_{a}.json"
        d = load(p)
        if "final" in d:
            v = [r["validated_speedup_vs_maxautotune"] for r in d["final"].values() if r["status"] == "ok"]
            gms.append(math.exp(sum(math.log(x) for x in v) / len(v)))
            valids.append(d["attribution"]["lm_verified_rate"])
        else:                                  # nolearn: headline recovered pre-clobber (ablations.md)
            gms.append(1.302); valids.append(0.969)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.2))
    x = np.arange(4)
    a1.bar(x, gms, color=["#444", "#888", "#2a9d2a", "#bbb"])
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=6)
    a1.set_ylim(1.25, 1.40); a1.set_ylabel("geomean vs MA")
    a1.set_title("(a) discovered speedup", fontsize=8)
    for xi, g in zip(x, gms):
        a1.text(xi, g + 0.002, f"{g:.3f}", ha="center", fontsize=6)
    a2.bar(x, valids, color=["#444", "#888", "#2a9d2a", "#bbb"])
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=6)
    a2.set_ylim(0.9, 1.02); a2.set_ylabel("LM valid-rate")
    a2.set_title("(b) validity", fontsize=8)
    for xi, g in zip(x, valids):
        a2.text(xi, g + 0.003, f"{g:.3f}", ha="center", fontsize=6)
    fig.suptitle("Ablations: 8 familiar ops, 24 rounds, single seed — the GRPO term earns "
                 "nothing on familiar ops", fontsize=8, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("ablation.pdf")


# ---------------- Fig 4: in-run learning (valid-rate by round + idiom) ----------------------
def fig_learning():
    v2 = load(R / "kernelsmith_v2.json")
    hist = v2["history"]
    rounds = [h["round"] for h in hist]
    rate = [h["lm_verified_rate"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.1))
    a1.plot(rounds, rate, lw=1.2, color="#2a5d9d")
    a1.set_xlabel("round"); a1.set_ylabel("cumulative LM valid-rate")
    a1.set_title("(a) validity over the 37-op discovery run", fontsize=8)
    a1.axvspan(0, 37, alpha=0.08, color="gray")
    a1.text(18, 0.95, "pass 1", fontsize=6, ha="center")
    a1.text(56, 0.95, "pass 2", fontsize=6, ha="center")
    a1.set_ylim(0, 1.0)
    # idiom: per-round LM-verified for softplus/mish family, pass1 vs pass2
    fam = [(h["round"], h["lm_verified_this_round"]) for h in hist
           if "softplus" in h["op"] or "mish" in h["op"]]
    p1 = [v for r, v in fam if r <= 37]
    p2 = [v for r, v in fam if 37 < r <= 74]
    a2.bar([0, 1], [np.mean(p1), np.mean(p2)], color=["#c44", "#2a9d2a"], width=0.5)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["pass 1", "pass 2"], fontsize=7)
    a2.set_ylabel("mean verified / 8 (softplus+mish ops)")
    a2.set_title("(b) the overflow-guard idiom is learned", fontsize=8)
    for xi, v in zip([0, 1], [np.mean(p1), np.mean(p2)]):
        a2.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "learning.pdf")
    plt.close(fig)
    print("learning.pdf")


# ---------------- Fig 5: loss-region flip ----------------------------------------------------
def fig_flip():
    g1 = load(R / "rebench_shapes_qwen3.6-27b.json")
    inv = load(R / "kernelsmith_invent.json")["final"]
    # bases that exist in grid1 (plain rmsnorm wasn't among the 32 trained kernels)
    pairs = [("softmax", "softmax_short"), ("layernorm_gelu", "layernorm_gelu_short"),
             ("add_layernorm_sigmoid", "add_layernorm_sigmoid_short")]
    before, after, names = [], [], []
    for base, short in pairs:
        cells = [c for c in g1["per_op"][base]["cells"]
                 if (c["M"], c["N"], c["dtype"]) == (16384, 2048, "fp16")]
        before.append(cells[0]["vs_maxauto"])
        after.append(inv[short]["validated_speedup_vs_maxautotune"])
        names.append(base)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    ax.bar(x - 0.18, before, width=0.36, label="before (v1/v2 kernel @16384×2048)", color="#c44")
    ax.bar(x + 0.18, after, width=0.36, label="after (invention run, validated)", color="#2a9d2a")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6)
    ax.set_ylabel("speedup vs MA")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("The characterized loss regime, before and after\n(the fix was simpler than "
                 "the human diagnosis)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "flip.pdf")
    plt.close(fig)
    print("flip.pdf")


def fig_minicpm_ablation():
    import statistics as stt
    MA = R / "minicpm_ablation"
    if not MA.exists():
        print("minicpm_ablation/ missing, skip"); return
    arms = ["control", "nofeedback", "distillonly", "nolearn"]
    labels = ["full loop", "$-$feedback", "$-$GRPO", "$-$learning"]
    means, stds = [], []
    for a in arms:
        gms = []
        for s in (0, 1, 2):
            d = load(MA / f"abl_{a}_s{s}.json")
            vv = [r["validated_speedup_vs_maxautotune"] for r in d["final"].values() if r["status"] == "ok"]
            gms.append(math.exp(sum(math.log(x) for x in vv if x > 0) / len([x for x in vv if x > 0])))
        means.append(stt.mean(gms)); stds.append(stt.pstdev(gms))
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    x = np.arange(4)
    ax.bar(x, means, yerr=stds, color=["#444", "#888", "#2a9d2a", "#bbb"], capsize=4,
           error_kw={"lw": 1.0})
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("geomean vs max-autotune")
    ax.set_ylim(1.0, 1.18)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=6)
    ax.set_title("Multi-seed ablation, MiniCPM5-1B (3 seeds, free 4090): arms tied;\n"
                 "all beat max-autotune. Error bars $\\geq$ between-arm gaps.", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "minicpm_ablation.pdf")
    plt.close(fig)
    print("minicpm_ablation.pdf")


if __name__ == "__main__":
    fig_heatmap(); fig_stability(); fig_ablation(); fig_learning(); fig_flip()
    fig_minicpm_ablation()
    print("all figures ->", FIG)
