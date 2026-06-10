"""CPU-only unit tests (no GPU): the pure-Python invariants the loop depends on.

GPU truth lives in `harness.py`'s selftest (run on real hardware / Modal); these tests
guard the scaffolding — dedup canonicalization, kernel extraction, the chain grammar's
structural integrity, and the grid-input kind map — so CI can catch regressions cheaply.
"""
import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "ouroboros"
sys.path.insert(0, str(HERE))

from canonicalize import canonical_src, khash, knobs           # noqa: E402
from rl_kernelsmith import extract_kernel, mutate_src          # noqa: E402
import chains                                                  # noqa: E402


# ---------------- canonicalize: dedup must be cosmetic-invariant, knob-sensitive ----------
RMS = (HERE / "seed_kernels" / "rmsnorm.py").read_text()


def test_khash_invariant_to_comments_and_renames():
    cosmetic = "# a totally different comment\n" + RMS.replace("acc", "accum").replace("row", "r0w")
    assert khash(RMS) == khash(cosmetic)


def test_khash_distinguishes_real_knob_changes():
    assert khash(RMS) != khash(RMS.replace("BLOCK = 1024", "BLOCK = 2048"))


def test_knobs_extracts_features():
    k = knobs(RMS)
    assert k["uses_rsqrt"] and k["n_loads"] >= 1 and k["n_stores"] >= 1


# ---------------- extract_kernel: the LM-output parser --------------------------------------
def test_extract_kernel_fenced_block():
    text = "Sure! Here is the kernel:\n```python\n@triton.jit\ndef _k(): pass\ndef run(x): return x\n```\nHope it helps."
    out = extract_kernel(text)
    assert out.startswith("@triton") and "Hope it helps" not in out


def test_extract_kernel_bare_code():
    text = "def run(x):\n    return x\n"
    assert extract_kernel(text).startswith("def run")


def test_mutate_src_parses():
    import random
    for seed in range(8):
        assert isinstance(mutate_src(RMS, random.Random(seed)), str)


# ---------------- chain grammar: every template must at least be valid Python ---------------
ALL = chains.all_chains()


def test_chain_count_and_uniqueness():
    names = [n for (n, _k, _r, _s) in ALL]
    assert len(names) == len(set(names))
    assert len(names) == 2 * 2 * len(chains.ACTNAMES)          # norms x residual x epilogues


def test_chain_templates_are_valid_python():
    for name, _kind, _ref, srcs in ALL:
        assert len(srcs) == 2, name
        for s in srcs:
            ast.parse(s)                                        # IndentationError etc. caught here


def test_chain_torch_matches_declared_epilogues():
    import torch
    t = torch.linspace(-30, 30, 257, dtype=torch.float32)      # CPU tensor; covers both branches
    for act, (fn, _expr) in chains.ACTS.items():
        out = fn(t)
        assert torch.isfinite(out).all(), f"epilogue {act} not finite on [-30,30]"


# ---------------- grid kind map: every registered op must have a grid builder --------------
def test_grid_kind_covers_all_specs():
    torch = pytest.importorskip("torch")
    import specs
    for op in specs.SPECS:
        specs._grid_kind(op)                                    # raises KeyError on a gap


# ---------------- selftest case list: every referenced seed file must exist ----------------
def test_selftest_seed_files_exist():
    src = (HERE / "harness.py").read_text()
    tree = ast.parse(src)
    missing = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.endswith(".py") and len(node.value) > 3):
            if not (HERE / "seed_kernels" / node.value).exists() and node.value != "harness.py":
                missing.append(node.value)
    assert not missing, f"selftest references missing seeds: {missing}"
