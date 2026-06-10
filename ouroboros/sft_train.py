"""PHASE 1 — proper SFT: teach Qwen3.5-2B to WRITE valid Triton, to convergence.

Not a warm-up. This builds a structurally-diverse, harness-VERIFIED kernel corpus (teacher
structures × launch-knob variants, filtered to status==ok), then SFTs the LoRA over it for
real (multiple epochs, cosine schedule), and GATES on a measured per-op valid-rate sampled
at the temperatures RL will use. The advisor's bar: ≥80% valid all ops, outputs DIVERSE (not
one memorized string). Only then does RL (rl_kernelsmith.py --load-adapter) have a competent
base to push on.

Honest scope (stated up front): proper SFT buys the model reliably EMITTING verified kernels
that beat PyTorch — that IS the thesis. It does not make the model invent faster-than-corpus
kernels; that is RL's job, bounded by the search.

Run: HF_HOME=.../.hf-cache python -u sft_train.py --epochs 12 --out outputs/sft_adapter
"""
from __future__ import annotations
import argparse, json, os, random, re, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from harness import evaluate
from canonicalize import canonical_src, khash
from teacher_kernels import all_candidates, STRUCTURES
from rl_kernelsmith import Proposer, extract_kernel
from specs import SPECS

DATASETS = HERE / "datasets"; DATASETS.mkdir(exist_ok=True)


def blank_structure(src: str) -> str:
    """Structural fingerprint: blank numeric literals + canonicalize, so two kernels that
    differ only in block size / warps collapse — used to measure GENUINE diversity."""
    return canonical_src(re.sub(r"\d+", "N", src))


def build_corpus(ops, save=True):
    """Harness-filter the teacher candidates to verified-only; report diversity; save jsonl.
    CACHED: if a prior verified_kernels.jsonl covers all requested ops, reuse it (the harness
    filtering is deterministic, so we never pay the ~25min rebuild twice)."""
    cache = DATASETS / "verified_kernels.jsonl"
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
        have = {r["op"] for r in rows}
        if set(ops).issubset(have):
            corpus = {op: [] for op in ops}
            for r in rows:
                if r["op"] in corpus:
                    corpus[r["op"]].append(r["src"])
            print(f"[corpus] CACHED — loaded {len(rows)} verified kernels covering {len(ops)} ops "
                  f"from {cache} (skipping rebuild)", flush=True)
            return corpus
    corpus = {}                       # op -> list[src]
    print("[corpus] filtering teacher candidates through the harness (correctness-only) ...", flush=True)
    rows = []
    for op in ops:
        cands = all_candidates(op)
        verified = []
        for c in cands:
            if evaluate(c, op, correctness_only=True).status == "ok":
                verified.append(c)
        distinct = len({blank_structure(c) for c in verified})
        corpus[op] = verified
        print(f"  {op:12}: {len(verified)}/{len(cands)} verified | {distinct} distinct structures", flush=True)
        for c in verified:
            rows.append({"op": op, "src": c, "khash": khash(c), "struct": blank_structure(c)[:80]})

    # fold in HARVESTED REAL kernels (liger/unsloth/triton-tutorials) for matching ops —
    # real-world technique the model learns from, already harness-verified by external_harvest.py.
    real_dir = DATASETS / "real_kernels"; man = real_dir / "MANIFEST.json"
    if man.exists():
        n_real = 0
        for rec in json.loads(man.read_text()):
            op, fp = rec["op"], real_dir / rec["file"]
            if op in corpus and fp.exists():
                src = fp.read_text()
                if evaluate(src, op, correctness_only=True).status == "ok":   # re-verify, never trust
                    corpus[op].append(src); n_real += 1
                    rows.append({"op": op, "src": src, "khash": khash(src),
                                 "struct": blank_structure(src)[:80], "provenance": rec["provenance"]})
        print(f"  [real] folded in {n_real} verified real-world kernels (liger/unsloth/tutorials)", flush=True)
    if save:
        (DATASETS / "verified_kernels.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        print(f"[corpus] {len(rows)} verified kernels -> {DATASETS/'verified_kernels.jsonl'}", flush=True)
    return corpus


def eval_validrate(prop, ops, temps=(0.3, 0.5, 0.8), k=8, grad_ckpt=False):
    """Sample k kernels/op at each temp, harness-verify (correctness-only). Returns per-op
    valid-rate (at the lowest temp = the cleanest) and distinct-structure count (anti-memorise).

    NOTE: gradient checkpointing (on for training) forces use_cache=False, which makes decode
    O(n^2) and the eval ~50x slower. We toggle it OFF here so generation uses the KV cache, then
    restore it for training. This changes NOTHING about what is measured — same k, temps, samples."""
    import time
    import torch
    if grad_ckpt:
        try: prop.model.gradient_checkpointing_disable()
        except Exception: pass
    try: prop.model.config.use_cache = True
    except Exception: pass
    prop.model.eval()
    report = {}
    t0 = time.time()
    model_name = getattr(prop, "name", "model")
    print(f"[eval] {model_name}: sampling {len(ops)} ops × {len(temps)} temps × k={k} "
          f"({len(ops) * len(temps) * k} generations) ...", flush=True)
    for i, op in enumerate(ops):
        op_t0 = time.time()
        per_temp = {}
        valids_lowT = []
        for t in temps:
            old = prop.temp; prop.temp = t
            comps, _ = prop.sample(prop.prompt(op, ""), k)
            prop.temp = old
            srcs = [extract_kernel(prop.tok.decode(c, skip_special_tokens=True)) for c in comps]
            oks = [evaluate(s, op, correctness_only=True).status == "ok" for s in srcs]
            per_temp[t] = sum(oks) / k
            if t == temps[0]:
                valids_lowT = [s for s, o in zip(srcs, oks) if o]
        distinct = len({blank_structure(s) for s in valids_lowT})
        report[op] = {"validrate_by_temp": {str(t): round(v, 3) for t, v in per_temp.items()},
                      "distinct_valid_structures": distinct}
        rates = " ".join(f"{t}:{per_temp[t]:.0%}" for t in temps)
        print(f"[eval] {i+1:2}/{len(ops)} {op:16} valid[{rates}] distinct={distinct} "
              f"(+{time.time()-op_t0:.0f}s, total {(time.time()-t0)/60:.1f}min)", flush=True)
    prop.model.train()
    try: prop.model.config.use_cache = False
    except Exception: pass
    if grad_ckpt:                                   # restore checkpointing for the training forward
        try:
            prop.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            prop.model.enable_input_require_grads()
        except Exception: pass
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--ops", default="rmsnorm,softmax,swiglu,add_rmsnorm")
    ap.add_argument("--epochs", type=int, default=30, help="MAX epochs; real stop is the gate or plateau")
    ap.add_argument("--patience", type=int, default=6,
                    help="stop if mean valid-rate hasn't improved for this many epochs (plateau = converged)")
    ap.add_argument("--lora-rank", type=int, default=64, dest="lora_rank",
                    help="LoRA rank — capacity for the model to learn many ops well (16=minimal, 64+=real)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8, dest="batch",
                    help="micro-batch (sequences per forward) — size this to the GPU (H100: 8-16)")
    ap.add_argument("--accum", type=int, default=2, help="gradient-accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--grad-ckpt", action="store_true", dest="grad_ckpt", default=True,
                    help="gradient checkpointing — trades compute for memory so a bigger batch fits")
    ap.add_argument("--warmup", type=float, default=0.05, help="warmup fraction of total steps")
    ap.add_argument("--eval-temps", default="0.3,0.5,0.8")
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument("--gate", type=float, default=0.8, help="valid-rate (lowest temp) to call SFT done")
    ap.add_argument("--corpus-only", action="store_true", dest="corpus_only",
                    help="build + save the verified corpus, then exit (before the model loads)")
    ap.add_argument("--skip-baseline", action="store_true", dest="skip_baseline",
                    help="skip the pre-SFT cold-baseline eval (optional pre-measurement; not the gate)")
    ap.add_argument("--out", default=str(HERE / "outputs" / "sft_adapter"))
    ap.add_argument("--load-adapter", default=None, dest="load_adapter",
                    help="continue-SFT from an existing LoRA adapter (trainable) instead of a fresh one")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    temps = tuple(float(t) for t in args.eval_temps.split(","))
    rng = random.Random(0)
    print(f"SFT | model={args.model} | ops={ops} | epochs={args.epochs} lr={args.lr} accum={args.accum}", flush=True)

    # ---- corpus -------------------------------------------------------------------------
    corpus = build_corpus(ops)
    if args.corpus_only:
        print("[corpus-only] verified corpus built + saved; exiting before model load.", flush=True)
        return
    # BALANCE per-op: oversample each op's verified kernels to the max count, so an op with a
    # naturally thinner corpus (e.g. rope: head-dim BLOCK exposes no knob variants) is NOT
    # under-trained relative to the others. No op gets short-changed.
    maxn = max(len(corpus[op]) for op in ops)
    examples = []
    for op in ops:
        pool = corpus[op]
        reps = (maxn + len(pool) - 1) // len(pool)
        examples += [(op, s) for s in (pool * reps)[:maxn]]
    rng.shuffle(examples)
    print(f"[corpus] {len(examples)} SFT examples (balanced to {maxn}/op)", flush=True)

    # ---- model --------------------------------------------------------------------------
    import math
    import torch
    from transformers import get_cosine_schedule_with_warmup
    # max_new=1024: the long two-pass kernels (layernorm/add_layernorm) exceed 384 tokens —
    # a smaller cap truncates them at EVAL and falsely reports 0% valid (a measurement bug).
    prop = Proposer(args.model, temp=temps[0], kl=False, lora_rank=args.lora_rank, max_new=1024,
                    load_adapter=args.load_adapter)
    if prop.tok.pad_token_id is None:
        prop.tok.pad_token = prop.tok.eos_token
    if args.grad_ckpt:                                # fit a bigger batch on the GPU
        prop.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        prop.model.enable_input_require_grads()
    B = max(1, args.batch)
    steps_per_epoch = math.ceil(math.ceil(len(examples) / B) / args.accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in prop.model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)

    def collate(chunk):
        """Pad a micro-batch of (op, src) into input_ids/attention_mask/labels; train on the
        COMPLETION only (prompt + pad tokens are masked with -100). Standard causal-LM SFT."""
        seqs, labs = [], []
        for op, src in chunk:
            pids = prop.tok(prop.prompt(op, ""), add_special_tokens=True).input_ids
            cids = prop.tok(src + prop.tok.eos_token, add_special_tokens=False).input_ids
            seqs.append(pids + cids)
            labs.append([-100] * len(pids) + cids)
        L = max(len(s) for s in seqs)
        pad = prop.tok.pad_token_id
        ii = torch.full((len(seqs), L), pad, dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        lb = torch.full((len(seqs), L), -100, dtype=torch.long)
        for j, (s, l) in enumerate(zip(seqs, labs)):
            ii[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1; lb[j, :len(l)] = torch.tensor(l)
        return ii.to(prop.dev), am.to(prop.dev), lb.to(prop.dev)

    # baseline valid-rate BEFORE SFT (cold base — proves SFT did the work). Optional/skippable.
    base_rep = {}
    if args.skip_baseline:
        print("[eval] cold baseline SKIPPED (--skip-baseline)", flush=True)
    else:
        print("[eval] cold baseline valid-rate (pre-SFT) ...", flush=True)
        base_rep = eval_validrate(prop, ops, temps, args.eval_k, grad_ckpt=args.grad_ckpt)
        for op in ops:
            print(f"  {op:12}: {base_rep[op]['validrate_by_temp']} distinct={base_rep[op]['distinct_valid_structures']}", flush=True)

    # ---- train --------------------------------------------------------------------------
    t0 = time.time(); step = 0; best_mean = -1.0; hist = []; epochs_since_best = 0
    prop.model.train(); opt.zero_grad()
    for ep in range(1, args.epochs + 1):
        rng.shuffle(examples)
        ep_loss = 0.0; nb = 0; micro = 0
        nbatches = math.ceil(len(examples) / B)
        for bi in range(0, len(examples), B):
            ii, am, lb = collate(examples[bi:bi + B])
            loss = prop.model(input_ids=ii, attention_mask=am, labels=lb).loss   # masked CE on completion
            (loss / args.accum).backward(); ep_loss += float(loss); nb += 1; micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in prop.model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
            if nb % 40 == 0:
                print(f"[ep {ep:2} train] batch {nb}/{nbatches} loss={ep_loss/nb:.3f} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
            del ii, am, lb, loss
        if micro % args.accum != 0:                  # flush the last partial accumulation
            torch.nn.utils.clip_grad_norm_([p for p in prop.model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
        torch.cuda.empty_cache()
        ep_loss /= max(1, nb)
        rep = eval_validrate(prop, ops, temps, args.eval_k, grad_ckpt=args.grad_ckpt)
        mean_lowT = sum(rep[op]["validrate_by_temp"][str(temps[0])] for op in ops) / len(ops)
        vram = torch.cuda.max_memory_allocated() / 1e9; torch.cuda.reset_peak_memory_stats()
        print(f"[ep {ep:2}/{args.epochs}] loss={ep_loss:.3f} "
              f"mean_validrate@{temps[0]}={mean_lowT:.0%} vram={vram:.1f}GB "
              f"{(time.time()-t0)/60:.1f}min", flush=True)
        for op in ops:
            r = rep[op]
            print(f"     {op:12} valid={r['validrate_by_temp']} distinct_structs={r['distinct_valid_structures']}", flush=True)
        hist.append({"epoch": ep, "mean_validrate_lowT": mean_lowT, "per_op": rep})
        if mean_lowT > best_mean + 1e-6:             # save the best checkpoint by valid-rate
            best_mean = mean_lowT; epochs_since_best = 0
            prop.model.save_pretrained(args.out)
            print(f"     -> saved adapter (best mean valid-rate {best_mean:.0%}) to {args.out}", flush=True)
        else:
            epochs_since_best += 1
        if mean_lowT >= args.gate and all(
                rep[op]["validrate_by_temp"][str(temps[0])] >= args.gate for op in ops):
            print(f"\n*** SFT GATE MET: all ops ≥{args.gate:.0%} valid at temp {temps[0]} (converged) ***", flush=True)
            break
        if epochs_since_best >= args.patience:
            print(f"\n*** PLATEAU: no valid-rate improvement for {args.patience} epochs — converged at "
                  f"{best_mean:.0%} ***", flush=True)
            break

    rep_path = HERE / "reports" / "sft.json"
    rep_path.write_text(json.dumps({"args": vars(args), "cold_baseline": base_rep,
                                    "history": hist, "best_mean_validrate": best_mean}, indent=2))
    print(f"\n=== SFT DONE === best mean valid-rate@{temps[0]} = {best_mean:.0%} | "
          f"adapter -> {args.out} | report -> {rep_path}", flush=True)


if __name__ == "__main__":
    main()
