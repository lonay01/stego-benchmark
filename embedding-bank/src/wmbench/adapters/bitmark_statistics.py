"""Narrow compatibility repair for the pinned author's optional BitMark statistics.

Sampling, RNG use, watermark logits and image reconstruction stay untouched.
The upstream code reads flip counters even when count_bit_flip is false. In
that case emit the already implemented entropy-only statistics branch instead.
No vendor checkout is modified, and an unknown upstream source is rejected.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

UPSTREAM_SHA256 = "e7a1a74ed26a42d806d4dfe2629078ff7079524a861c714dd1a3eb6480bf2eba"


def guarded_statistics_tree(source: str) -> ast.Module:
    """Change exactly the statistics branch, never the watermark sampling branch."""
    tree = ast.parse(textwrap.dedent(source))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "si in watermark.scales":
            continue
        # Only this branch writes diagnostic flip counters into ret[i].
        if (
            node.body
            and isinstance(node.body[0], ast.Assign)
            and ast.unparse(node.body[0].targets[0]).startswith("ret[")
            and any(
                isinstance(n, ast.Name) and n.id == "scale_bit_flips"
                for stmt in node.body
                for n in ast.walk(stmt)
            )
        ):
            candidates.append(node)
    if len(candidates) != 1:
        raise RuntimeError("Unknown BitMark statistics structure; refusing to patch")
    candidates[0].test = ast.BoolOp(
        op=ast.And(),
        values=[candidates[0].test, ast.parse("watermark.count_bit_flip", mode="eval").body],
    )
    return ast.fix_missing_locations(tree)


def repair_bitmark_statistics(generator_class) -> None:
    original = generator_class.autoregressive_infer_cfg
    if getattr(original, "_wmbench_statistics_guard", False):
        return
    path = Path(inspect.getsourcefile(inspect.unwrap(original)))
    normalized_source = path.read_text(encoding="utf-8")
    if hashlib.sha256(normalized_source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Unknown BitMark upstream source; statistics patch is not approved")
    tree = guarded_statistics_tree(inspect.getsource(original))
    namespace = {}
    exec(
        compile(tree, str(path) + "::wmbench-statistics-guard", "exec"),
        inspect.unwrap(original).__globals__,
        namespace,
    )
    repaired = namespace["autoregressive_infer_cfg"]
    repaired._wmbench_statistics_guard = True
    generator_class.autoregressive_infer_cfg = repaired
