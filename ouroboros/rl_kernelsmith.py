"""KERNELSMITH — the search loop that writes the kernels, grounded by the harness.

Direct descendant of `sec_sqli/discovery_specialist/rl_specialist.py`. SAME mechanics:
rejection-sampling bootstrap from the base's OWN verified outputs, RL (GRPO) fused with
self-distillation on verified winners, KL-to-reference, dedup-by-canonicalization. ONE
deliberate re-map (per the advisor): the reward is NOT novelty/QD — it is the SCALAR
measured speedup. There is exactly one prize per op: the fastest correct kernel.

  each round, for an op:
    A. PROPOSE   prompt = signature + an exemplar (a seed kernel for a DIFFERENT op) +
                 the structured feedback from the best/last attempt. Sample a group.
                 (+ a templated-mutation arm: canonical transforms of the current best —
                  the "peephole" explorer: change BLOCK, add num_warps/num_stages.)
    B. VERIFY    harness.evaluate in an isolated subprocess -> correctness BOOLEAN +
                 speedup NUMBER. Immutable; never trained. Dedup: a kernel whose khash was
                 already measured is NOT re-evaluated (costly).
    C. REWARD    ok -> 1 + speedup_vs_compile ; incorrect -> 0 ; runtime/timeout -> -0.2 ;
                 compile_fail -> -0.5.  (correctness first, then speed — one scalar.)
    D. LEARN     GRPO advantage over the group (RL half) + self-distill the round's FASTEST
                 verified kernel (self-distill half). KL-to-frozen-reference guards drift.

  ARCHIVE  best[op] = (speedup_vs_compile, src) — the product. Honest bound (stated, like
  the specialist's): these fusion ops TIE torch.compile out of the box (see
  reports/harness_selftest.json); the win this loop chases is edging past compile via
  block/warp/stage tuning + fusion the inductor default misses. Beating eager is the floor.

Run (full LM loop, needs the train venv + GPU):
  /home/tihor/webllm/.venv-train/bin/python rl_kernelsmith.py --model Qwen/Qwen3-1.7B \
      --ops rmsnorm,softmax,swiglu --rounds 40 --group 8

Run (no-LLM templated search — proves the search/dedup/archive layer cheaply):
  /home/tihor/webllm/.venv-train/bin/python rl_kernelsmith.py --no-llm --ops rmsnorm,softmax,swiglu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import evaluate                      # the immutable referee (subprocess)
from canonicalize import khash, knobs             # dedup
from specs import SPECS, get_spec

SEED_DIR = HERE / "seed_kernels"
SYS = ("You are an expert GPU kernel engineer. Write a single correct, fast Triton kernel. "
       "Output ONLY one fenced python code block defining `run(*inputs)` and its @triton.jit "
       "kernel. Accumulate reductions in float32. No prose.")


# ----------------------------------------------------------------- reward (scalar speedup) ---
def reward_of(res) -> float:
    if res.status == "ok":
        return 1.0 + float(res.speedup_compile)        # correct -> base 1.0 + the flex bar
    if res.status == "incorrect":
        return 0.0
    if res.status in ("runtime_fail", "timeout", "crash"):
        return -0.2
    return -0.5                                         # compile_fail


# ----------------------------------------------------------------- templated mutation arm ---
def mutate_src(src: str, rng: random.Random) -> str:
    """Canonical 'peephole' transforms of a kernel (the half-templated explorer)."""
    out = src
    choice = rng.randrange(4)
    if choice == 0 and "BLOCK = " in out:
        out = re.sub(r"BLOCK = \d+", f"BLOCK = {rng.choice([256, 512, 2048, 4096])}", out, count=1)
    elif choice == 1:                                  # add/replace num_warps on the launch
        nw = rng.choice([2, 4, 8, 16])
        if "num_warps=" in out:
            out = re.sub(r"num_warps=\d+", f"num_warps={nw}", out, count=1)
        else:
            out = re.sub(r"\]\(([^\n]*?)\)\n", rf"](\1, num_warps={nw})\n", out, count=1)
    elif choice == 2:                                  # add num_stages
        ns = rng.choice([1, 2, 3, 4])
        if "num_stages=" in out:
            out = re.sub(r"num_stages=\d+", f"num_stages={ns}", out, count=1)
        else:
            out = re.sub(r"\]\(([^\n]*?)\)\n", rf"](\1, num_stages={ns})\n", out, count=1)
    else:                                              # bump BLOCK up a notch
        m = re.search(r"BLOCK = (\d+)", out)
        if m:
            out = out.replace(f"BLOCK = {m.group(1)}", f"BLOCK = {int(m.group(1)) * 2}", 1)
    return out


def extract_kernel(text: str) -> str:
    """Pull the fenced python block (the kernel source) out of an LM completion."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    # keep from the first decorator/import/def onward
    i = min([body.find(k) for k in ("@triton", "import ", "def run", "def _") if body.find(k) >= 0] or [0])
    return body[i:].strip()


# ----------------------------------------------------------------- proposer (LM) wrapper ----
class Proposer:
    # default LoRA target modules: attention + MLP (real capacity, not just attention) — the
    # MLP gate/up/down carry most of the transformer's representational room.
    LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_name: str, temp: float, kl: bool = True, max_new: int = 1024,
                 load_adapter: str | None = None, lora_rank: int = 64, lora_targets=None):
        import torch
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.torch = torch; self.temp = temp; self.dev = "cuda"; self.use_kl = kl; self.max_new = max_new
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)

        def _load():
            # Qwen3.5-2B is Qwen3_5ForConditionalGeneration (a VL model) — AutoModelForCausalLM
            # can't load it. Use the text-generation auto-class; we generate text-only and the
            # LM head still yields per-token vocab logits (all GRPO/logprob needs).
            from transformers import AutoConfig, AutoModelForCausalLM
            # TEXT-ONLY fast path: multimodal models (Qwen3.6/qwen3_5, Gemma4) expose a
            # `language_model_only` flag — set it so the vision/video tower is skipped entirely.
            # We only ever generate text, so this is the correct + much faster path.
            cfg = None
            try:
                cfg = AutoConfig.from_pretrained(model_name)
                if hasattr(cfg, "language_model_only"):
                    cfg.language_model_only = True
            except Exception:
                cfg = None
            kw = {"dtype": torch.bfloat16}
            if cfg is not None:
                kw["config"] = cfg
            # Prefer SDPA attention (faster decode); fall back to default impl, then image-text-to-text.
            try:
                return AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", **kw)
            except Exception:
                pass
            try:
                return AutoModelForCausalLM.from_pretrained(model_name, **kw)
            except (ValueError, KeyError, OSError):
                from transformers import AutoModelForImageTextToText
                return AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16)

        base = _load().to(self.dev)
        print(f"[proposer] loaded {model_name} · attn={getattr(base.config, '_attn_implementation', '?')}", flush=True)
        self.ref = None
        if kl:                                       # frozen reference = the SFT'd weights too
            ref_base = _load().to(self.dev)
            if load_adapter:
                ref_base = PeftModel.from_pretrained(ref_base, load_adapter, is_trainable=False)
            self.ref = ref_base.eval()
            for p in self.ref.parameters(): p.requires_grad_(False)
        if load_adapter:
            # RESUME the SFT'd LoRA as TRAINABLE (the advisor's integration trap: PeftModel can
            # silently reload frozen; is_trainable=True keeps the adapter grads on for RL).
            self.model = PeftModel.from_pretrained(base, load_adapter, is_trainable=True)
            n_train = sum(p.requires_grad for p in self.model.parameters())
            assert n_train > 0, "loaded SFT adapter came back FROZEN — RL would be a no-op"
            print(f"[proposer] resumed SFT adapter from {load_adapter} ({n_train} trainable tensors)")
        else:
            targets = lora_targets or self.LORA_TARGETS
            self.model = get_peft_model(base, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                         lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=targets))
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[proposer] LoRA rank={lora_rank} targets={targets} -> {n_train/1e6:.1f}M trainable params")
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)

    def prompt(self, op, feedback: str) -> str:
        spec = get_spec(op)
        exemplar_op = next((o for o in SPECS if o != op), op)
        exemplar = (SEED_DIR / f"{exemplar_op}.py").read_text()
        u = (f"Op `{op}`: {spec.notes}\nSignature:\n{spec.signature_hint}\n\n"
             f"Here is a valid Triton kernel for a DIFFERENT op (`{exemplar_op}`) as a style guide:\n"
             f"```python\n{exemplar}\n```\n")
        if feedback:
            u += f"\nYour previous attempt's harness feedback: {feedback}\nFix it and make it faster."
        def _render(m):
            try:
                return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        try:
            return _render([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
        except Exception:
            # Some chat templates (e.g. Gemma) reject a `system` role — fold it into the user turn.
            return _render([{"role": "user", "content": SYS + "\n\n" + u}])

    def sample(self, prompt, n, max_new=None):
        max_new = max_new or self.max_new
        ids = self.tok(prompt, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            g = self.model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=self.temp,
                                    top_p=0.97, num_return_sequences=n, pad_token_id=self.tok.eos_token_id,
                                    use_cache=True)   # KV cache (grad-checkpointing flips config use_cache off)
        plen = ids.input_ids.shape[1]
        out = [g[i, plen:] for i in range(n)]
        del g, ids
        self.torch.cuda.empty_cache()                  # release the generation KV cache promptly
        return out, self.tok(prompt, return_tensors="pt").input_ids[0].to(self.dev)

    def _logp(self, model, ids, plen):
        import torch.nn.functional as F
        out = model(input_ids=ids).logits          # (1, T, vocab) — LM head, even on the VL model
        logp = F.log_softmax(out[0, :-1].float(), dim=-1)
        chosen = logp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        return chosen[plen - 1:].sum()

    def learn(self, pids, comps, rewards, distill_src, kl=0.02):
        """GRPO over the group + self-distill the fastest verified kernel. Per-sample backward.
        Drift penalty to the frozen reference guards against policy collapse. (Honest label:
        this is |Δ sequence-logprob|, a crude one-sample KL proxy — not a true KL.)"""
        import torch
        rt = torch.tensor(rewards, dtype=torch.float32)
        did = False
        if rt.std() > 1e-6:
            adv = (rt - rt.mean()) / (rt.std() + 1e-6)
            self.model.train()
            for i in range(len(comps)):
                ids = torch.cat([pids, comps[i]]).unsqueeze(0)
                lp = self._logp(self.model, ids, len(pids))
                li = -(adv[i].to(self.dev) * lp) / len(comps)
                if self.ref is not None:
                    with torch.no_grad():
                        lp_ref = self._logp(self.ref, ids, len(pids))
                    li = li + kl * (lp - lp_ref).abs() / len(comps)
                li.backward(); did = True
        if distill_src:
            self.model.train()
            cids = self.tok(distill_src + self.tok.eos_token, return_tensors="pt").input_ids[0].to(self.dev)
            ids = torch.cat([pids, cids]).unsqueeze(0)
            (-self._logp(self.model, ids, len(pids))).backward(); did = True
        if did:
            self.opt.step(); self.opt.zero_grad()

    def sft(self, pairs, steps, lr=2e-4):
        """Supervised warm-start: teach the model to EMIT valid Triton in the harness's
        accepted dialect by maximizing logprob of verified kernels given their op prompt.
        This is the competence step a cold base needs before RL — without it the model
        hallucinates (`import triton.core`, undefined vars) and never verifies its own work."""
        import torch
        if not pairs:
            return
        self.model.train()
        sopt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        for s in range(steps):
            op, src = pairs[s % len(pairs)]
            pr = self.prompt(op, "")
            pids = self.tok(pr, return_tensors="pt").input_ids[0].to(self.dev)
            cids = self.tok(src + self.tok.eos_token, return_tensors="pt").input_ids[0].to(self.dev)
            ids = torch.cat([pids, cids]).unsqueeze(0)
            (-self._logp(self.model, ids, len(pids))).backward()
            sopt.step(); sopt.zero_grad()
            del ids, pids, cids
        self.torch.cuda.empty_cache()


# ----------------------------------------------------------------- the loop -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--ops", default="rmsnorm,softmax,swiglu")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--no-llm", action="store_true", dest="no_llm",
                    help="templated search only (no GPU LM): proves search/dedup/archive cheaply")
    ap.add_argument("--no-kl", action="store_true", dest="no_kl",
                    help="skip the frozen reference model (halves proposer VRAM; drops the drift "
                         "penalty — honestly a |Δ seq-logprob| to the frozen ref, a crude KL proxy)")
    ap.add_argument("--max-new", type=int, default=1024, dest="max_new",
                    help="max new tokens per generated kernel (caps generation VRAM/time)")
    ap.add_argument("--load-adapter", default=None, dest="load_adapter",
                    help="path to the SFT'd LoRA adapter (PHASE 1, sft_train.py) to run RL on top of")
    ap.add_argument("--explore-frac", type=float, default=0.5, dest="explore_frac",
                    help="fraction of the group spent on templated mutations of the best "
                         "(0.0 = MODEL ONLY: the model must write every kernel itself)")
    ap.add_argument("--out", default=str(HERE / "reports" / "kernelsmith.json"))
    ap.add_argument("--kernels-dir", default="outputs/best_kernels", dest="kernels_dir",
                    help="where to checkpoint/save the best kernels (relative to the repo). "
                         "Ablation arms MUST use their own dir so they never clobber the product.")
    ap.add_argument("--save-adapter", default=None, dest="save_adapter",
                    help="save the RL-trained LoRA adapter here when done (else the RL weights are LOST)")
    # --- ABLATION ARMS (paper section 4; each isolates one ingredient of the loop) ---------
    ap.add_argument("--no-feedback", action="store_true", dest="no_feedback",
                    help="ABLATION: never feed harness feedback into the prompt (no fix->retry signal)")
    ap.add_argument("--distill-only", action="store_true", dest="distill_only",
                    help="ABLATION: drop the GRPO advantage term; learn ONLY by self-distilling the "
                         "round's fastest verified kernel (rejection-sampling distillation)")
    ap.add_argument("--no-learn", action="store_true", dest="no_learn",
                    help="ABLATION: no weight updates at all — pure best-of-N sampling from the "
                         "starting adapter (the search-without-learning baseline)")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    rng = random.Random(0)
    print(f"KERNELSMITH | mode={'TEMPLATED-SEARCH' if args.no_llm else 'LM '+args.model} | "
          f"ops={ops} | rounds={args.rounds} group={args.group}")

    prop = None if args.no_llm else Proposer(args.model, args.temp, kl=not args.no_kl,
                                             max_new=args.max_new, load_adapter=args.load_adapter)

    import time
    t_start = time.time()
    best: dict[str, tuple[float, str]] = {}          # op -> (speedup_vs_compile, src)  THE PRODUCT
    seen: dict[str, dict[str, float]] = defaultdict(dict)   # op -> {khash: reward}  (dedup cache)
    last_fb: dict[str, str] = defaultdict(str)        # op -> last harness feedback (fed forward)
    hist = []

    # attribution: prove the MODEL (not the templated arm) is writing verified kernels.
    lm_verified_total = 0       # LM-authored kernels that passed the harness
    lm_eval_total = 0           # LM-authored kernels evaluated (non-dedup)
    explore_verified_total = 0
    best_src_arm: dict[str, str] = {}   # op -> "LM" | "explore" | "seed"  (who owns the current best)
    lm_wins = 0                 # times an LM kernel took the archive lead

    # ---- BOOTSTRAP: ground the seed kernels for the archive baseline. ------------------------
    # COMPETENCE comes from a PROPER SFT phase (sft_train.py) loaded via --load-adapter, NOT a
    # warm-up crammed in here. RL runs on the already-SFT'd model.
    print("[bootstrap] grounding the hand-written seed kernels ...")
    for op in ops:
        seed_src = (SEED_DIR / f"{op}.py").read_text()
        res = evaluate(seed_src, op)
        seen[op][khash(seed_src)] = reward_of(res)
        if res.status == "ok":
            best[op] = (res.speedup_compile, seed_src); best_src_arm[op] = "seed"
            print(f"  {op:12} seed -> {res.latency_ms:.4f}ms  {res.speedup_eager:.2f}x eager  "
                  f"{res.speedup_compile:.2f}x compile")
        else:
            print(f"  {op:12} seed -> {res.status}: {res.feedback}")

    # ---- ROUNDS -----------------------------------------------------------------------------
    for r in range(1, args.rounds + 1):
        op = ops[(r - 1) % len(ops)]
        base_src = best.get(op, (0.0, (SEED_DIR / f"{op}.py").read_text()))[1]

        fastest = None
        news = 0
        fb_first = [""]                                # first informative feedback this round
        lm_ok = [0]                                    # LM-authored kernels that verified this round

        def grade(src, arm):
            """Evaluate (dedup-cached) and fold a verified kernel into the archive. `arm` is
            'LM' or 'explore'. Attribution counts every LM EMISSION (incl. dedup hits): a
            model re-emitting the valid seed it was SFT'd on is STILL the model writing a
            working kernel — that is exactly the thesis we are measuring, so don't let dedup
            hide it. reward>1.0 in the cache iff the kernel was verified (see reward_of)."""
            nonlocal fastest, news, lm_verified_total, lm_eval_total, explore_verified_total, lm_wins
            h = khash(src)
            if h in seen[op]:                          # DEDUP: don't re-measure, but DO attribute
                rew = seen[op][h]; ok = rew > 1.0
                if arm == "LM":
                    lm_eval_total += 1
                    if ok: lm_verified_total += 1; lm_ok[0] += 1
                return rew
            res = evaluate(src, op)
            rew = reward_of(res); seen[op][h] = rew; news += 1
            ok = res.status == "ok"
            if arm == "LM":
                lm_eval_total += 1
                if ok: lm_verified_total += 1; lm_ok[0] += 1
            elif ok:
                explore_verified_total += 1
            if ok and res.speedup_compile > best.get(op, (-1e9, None))[0]:
                best[op] = (res.speedup_compile, src); fastest = src
                best_src_arm[op] = arm
                if arm == "LM": lm_wins += 1
            if not fb_first[0]:
                fb_first[0] = res.feedback
            return rew

        if args.no_llm:
            for src in [mutate_src(base_src, rng) for _ in range(args.group)]:
                grade(src, "explore")
            pids = comps = None
        else:
            # LM ARM: proposals are 1:1 with comps -> their rewards drive GRPO (correct credit
            # assignment: the model is scored ONLY on kernels it actually wrote). Prompt carries
            # last round's feedback (the fix→retry reflex) — unless the ablation disables it.
            prompt = prop.prompt(op, "" if args.no_feedback else last_fb[op])
            comps, pids = prop.sample(prompt, args.group)
            lm_srcs = [extract_kernel(prop.tok.decode(c, skip_special_tokens=True)) for c in comps]
            rewards = [grade(src, "LM") for src in lm_srcs]
            # EXPLORE ARM (separate, dialable): canonical mutations of the best. Wins here flow
            # into the model via self-distill; --explore-frac 0 forces the model to do it all.
            n_explore = round(args.group * args.explore_frac)
            for src in [mutate_src(base_src, rng) for _ in range(n_explore)]:
                grade(src, "explore")
            if not args.no_learn:
                # --distill-only zeroes the rewards => GRPO's std-gate skips the advantage term
                # and ONLY the self-distill half runs (rejection-sampling distillation).
                prop.learn(pids, comps,
                           [0.0] * len(rewards) if args.distill_only else rewards, fastest)

        last_fb[op] = fb_first[0]                       # feed this round's signal into the next
        b = best.get(op, (0.0, None))[0]
        prod = {o: round(best[o][0], 3) for o in ops if o in best}
        lm_rate = lm_verified_total / max(1, lm_eval_total)
        hist.append({"round": r, "op": op, "evaluated": news,
                     "best_speedup_vs_compile": round(b, 4), "product": prod,
                     "lm_verified_this_round": lm_ok[0], "lm_verified_rate": round(lm_rate, 3),
                     "best_owner": dict(best_src_arm)})
        tag = "  <== NEW BEST" if fastest is not None else ""
        # live ETA + VRAM + ATTRIBUTION (is the MODEL writing verified kernels yet?)
        elapsed = time.time() - t_start
        eta_min = (elapsed / r) * (args.rounds - r) / 60
        vram = ""
        if prop is not None:
            gb = prop.torch.cuda.max_memory_allocated() / 1e9
            vram = f" vram={gb:.1f}GB"
            prop.torch.cuda.reset_peak_memory_stats()
        print(f"[r{r:3}/{args.rounds} {op:10}] eval={news} LMok={lm_ok[0]}/{args.group} "
              f"(LMrate={lm_rate:.0%}) best={b:.3f}x[{best_src_arm.get(op,'-')}]  "
              f"product={prod}{vram} | {elapsed/60:.1f}min, ETA {eta_min:.0f}min{tag}", flush=True)

        # CHECKPOINT the best kernels to disk every 10 rounds so a transient CUDA crash never
        # loses 25 minutes of search (the in-search src; the final validation refines them).
        # empty_cache only at the boundary (it frees UNUSED cached blocks — never weights/grads/
        # optimizer state, so it can't affect training; per-round is just needless re-alloc churn).
        if r % 10 == 0:
            if prop is not None:
                prop.torch.cuda.empty_cache()          # curb rank-128 VRAM spikes, periodically
            ckpt = HERE / args.kernels_dir; ckpt.mkdir(parents=True, exist_ok=True)
            for o in ops:
                if o in best:
                    (ckpt / f"{o}.py").write_text(best[o][1])
            Path(args.out).write_text(json.dumps({"args": vars(args), "history": hist,
                "checkpoint_round": r, "in_search_best": {o: round(best[o][0], 4) for o in ops if o in best},
                "best_owner": dict(best_src_arm)}, indent=2))

    # ---- VALIDATE + SAVE THE PRODUCT --------------------------------------------------------
    # The search kept the MAX single measurement per op (winner's-curse: that pick is biased
    # upward by selection noise). Before publishing, RE-BENCHMARK each archived best with a
    # FRESH measurement AND the strong max-autotune baseline. The published number is this
    # clean re-measurement, not the lucky in-search peak.
    print("\n[validate] re-benchmarking the archived best per op (fresh + max-autotune) ...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kern_dir = HERE / args.kernels_dir; kern_dir.mkdir(parents=True, exist_ok=True)
    final = {}
    for op in ops:
        if op not in best:
            continue
        src = best[op][1]
        v = evaluate(src, op, strong=True)
        (kern_dir / f"{op}.py").write_text(src)
        final[op] = {"in_search_speedup_vs_compile": round(best[op][0], 4),
                     "validated_speedup_vs_compile": round(v.speedup_compile, 4),
                     "validated_speedup_vs_maxautotune": round(v.speedup_maxauto, 4),
                     "validated_speedup_vs_eager": round(v.speedup_eager, 4),
                     "latency_ms": v.latency_ms, "status": v.status,
                     "authored_by": best_src_arm.get(op, "?"),    # WHO wrote this winning kernel
                     "khash": khash(src), "knobs": knobs(src)}
        print(f"  {op:12}: in-search {best[op][0]:.3f}x -> validated {v.speedup_compile:.3f}x compile, "
              f"{v.speedup_maxauto:.3f}x max-autotune, {v.speedup_eager:.2f}x eager  "
              f"[authored by {best_src_arm.get(op,'?')}]")
    # ATTRIBUTION: the honest verdict on the project's thesis — is the MODEL writing the kernels?
    lm_rate = lm_verified_total / max(1, lm_eval_total)
    attribution = {"lm_kernels_evaluated": lm_eval_total, "lm_kernels_verified": lm_verified_total,
                   "lm_verified_rate": round(lm_rate, 4), "lm_archive_wins": lm_wins,
                   "explore_kernels_verified": explore_verified_total,
                   "best_kernels_authored_by": {op: best_src_arm.get(op, "?") for op in final}}
    Path(args.out).write_text(json.dumps({"args": vars(args), "history": hist, "final": final,
                                          "attribution": attribution}, indent=2))
    print("\n=== KERNELSMITH DONE === best validated kernel per op:")
    for op in ops:
        if op in final:
            f = final[op]
            print(f"  {op:12}: {f['validated_speedup_vs_compile']:.3f}x compile  "
                  f"{f['validated_speedup_vs_maxautotune']:.3f}x max-autotune  "
                  f"by={f['authored_by']}  knobs={f['knobs']}")
    n_lm_owned = sum(1 for op in final if best_src_arm.get(op) == "LM")
    print(f"\n--- ATTRIBUTION (the thesis: does the MODEL write the kernels?) ---")
    print(f"  LM-authored kernels: {lm_verified_total}/{lm_eval_total} verified "
          f"({lm_rate:.0%}) | LM took the archive lead {lm_wins} times")
    print(f"  Final best kernels OWNED BY THE MODEL: {n_lm_owned}/{len(final)}  "
          f"(rest from the templated-explore arm)")
    print(f"best kernels -> {kern_dir}  |  report -> {args.out}")
    # SAVE the RL-trained adapter (the policy weights) — without this the RL training is lost.
    if prop is not None and args.save_adapter:
        prop.model.save_pretrained(args.save_adapter)
        import os as _os
        wrote = _os.path.exists(_os.path.join(args.save_adapter, "adapter_model.safetensors"))
        print(f"[rl] {'SAVED' if wrote else 'FAILED TO SAVE'} RL adapter -> {args.save_adapter}", flush=True)
        if not wrote:
            raise RuntimeError("save_pretrained did not write adapter_model.safetensors")


if __name__ == "__main__":
    main()
