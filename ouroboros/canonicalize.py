"""Kernel canonicalization + dedup for the search loop.

The advisor's correction (recorded in CLAUDE.md): in `discovery_specialist`, diversity IS
the objective and `mech_of` powers a novelty-or-die REWARD. Here the objective is the
single FASTEST CORRECT kernel — diversity is just search exploration. So the analog of
`mech_of` lives HERE as **dedup**, NOT as a reward:

  - `khash(src)`   : a hash that is invariant to comments, whitespace, docstrings and
                     local-variable renaming, so the search never spends a (costly) harness
                     evaluation on a kernel it already measured. This is the
                     "AST-normalize then hash" move from the LLVM-peephole plan.
  - `knobs(src)`   : extract the performance-relevant features (BLOCK size, num_warps,
                     num_stages, autotune present, ops used) — useful for steering mutation
                     and for the open-trace, never as a correctness/speed signal.

Nothing in this file ever decides correctness or speed — that is the harness's sole job.
"""
from __future__ import annotations

import ast
import hashlib
import re


class _Canon(ast.NodeTransformer):
    """Alpha-rename local names to positional ids and strip docstrings, so two kernels that
    differ only cosmetically canonicalize identically."""

    def __init__(self):
        self._names: dict[str, str] = {}

    def _id(self, name: str) -> str:
        # keep dunders / library names (torch, triton, tl, range, tl.* attrs) stable;
        # only canonicalize ordinary identifiers the author chose.
        if name in _KEEP or name.startswith("__"):
            return name
        if name not in self._names:
            self._names[name] = f"v{len(self._names)}"
        return self._names[name]

    def visit_Name(self, node):
        node.id = self._id(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._id(node.arg)
        return node

    def visit_FunctionDef(self, node):
        # drop a leading string-literal docstring
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node


# identifiers that carry meaning across kernels — do NOT rename these
_KEEP = {"torch", "triton", "tl", "run", "range", "len", "float", "int", "tuple",
         "constexpr", "self", "True", "False", "None"}


def canonical_src(src: str) -> str:
    """Normalized source: parse -> strip docstrings + alpha-rename locals -> unparse.
    Falls back to comment/whitespace stripping if the source doesn't parse standalone."""
    try:
        tree = ast.parse(src)
        tree = _Canon().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        # not parseable in isolation (rare) — coarse normalize: drop comments + collapse ws
        no_comments = re.sub(r"#.*", "", src)
        return re.sub(r"\s+", " ", no_comments).strip()


def khash(src: str) -> str:
    return hashlib.sha1(canonical_src(src).encode()).hexdigest()[:16]


_INT = r"(\d+)"


def knobs(src: str) -> dict:
    """Performance-relevant features (for steering mutation + the open-trace, NOT scoring)."""
    def _find(pat, default=None, cast=int):
        m = re.search(pat, src)
        return cast(m.group(1)) if m else default
    return {
        "block": _find(rf"BLOCK\s*=\s*{_INT}"),
        "block_n": _find(rf"BLOCK_N\s*=\s*{_INT}"),
        "num_warps": _find(rf"num_warps\s*=\s*{_INT}"),
        "num_stages": _find(rf"num_stages\s*=\s*{_INT}"),
        "autotune": "autotune" in src,
        "uses_rsqrt": "rsqrt" in src,
        "uses_max_sub": bool(re.search(r"-\s*\w*max", src)) or "maximum" in src,
        "n_loads": src.count("tl.load"),
        "n_stores": src.count("tl.store"),
    }


if __name__ == "__main__":
    # demo: two cosmetically-different RMSNorm kernels canonicalize to the SAME hash;
    # a kernel with a different BLOCK is a genuinely distinct candidate (different hash).
    a = open(__file__.rsplit("/", 1)[0] + "/seed_kernels/rmsnorm.py").read()
    b = "# totally different comment\n" + a.replace("acc", "accum").replace("row", "r")
    c = a.replace("BLOCK = 1024", "BLOCK = 2048")
    print("rmsnorm.py            khash:", khash(a))
    print("renamed+recommented   khash:", khash(b), "(== a)" if khash(a) == khash(b) else "(MISMATCH!)")
    print("BLOCK 1024->2048      khash:", khash(c), "(distinct)" if khash(a) != khash(c) else "(collision!)")
    print("knobs(a):", knobs(a))
