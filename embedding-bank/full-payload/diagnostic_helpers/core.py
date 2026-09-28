"""Small, CPU-only diagnostics shared by the notebook controller and workers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from wmbench.io import digest, read_json, sha256, write_json
from wmbench.matching import match_psnr


RECHECK = (
    "fin", "hidden_benchmark", "trustmark_q", "trustmark_b", "trustmark_p",
    "chunkyseal", "rosteals", "arwgan", "distseal_posthoc_dcae",
)
BLOCKED = ("editguard", "omniguard", "sepwater", "fang_screen", "camera_shooting_instance")


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its root: {relative}")
    return path


def separate_roots(bank: Path, output: Path, scratch: Path) -> None:
    roots = [p.resolve() for p in (bank, output, scratch)]
    for i, left in enumerate(roots):
        for right in roots[i + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Bank, diagnostic results and scratch must be separate, non-nested folders")


def bit_metrics(bits: list, payload: list, indices: list) -> dict:
    if len(bits) != len(payload) or not payload or any(b not in (0, 1) for b in bits + payload):
        raise ValueError("Expected finite, binary, native-length decoder output")
    if len(indices) != 30 or len(set(indices)) != 30 or any(i < 0 or i >= len(bits) for i in indices):
        raise ValueError("Common payload must contain 30 distinct, in-range positions")
    errors = [i for i, (a, b) in enumerate(zip(bits, payload)) if a != b]
    common_errors = [i for i in indices if i in errors]
    return {
        "decoded_bits": list(map(int, bits)), "payload_bits": len(payload),
        "native_errors": len(errors), "native_ber": len(errors) / len(payload),
        "native_exact": not errors, "common30_errors": len(common_errors),
        "common30_ber": len(common_errors) / 30, "common30_exact": not common_errors,
        "error_positions": errors, "common_error_positions": common_errors,
        "common_indices": indices, "expected_bits": payload,
    }


def seed(image_id: str, operation: str) -> int:
    key = f"20260811|{image_id}|{operation}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def freeze(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Diagnostic specification changed: {path}; use a new diagnostic output directory")
    if not path.exists():
        write_json(path, value)


def seal(path: Path, signature: str, data: dict, artifacts: list[Path] = ()) -> None:
    write_json(path, {"signature": signature, **data, "artifacts": [
        {"path": p.name, "sha256": sha256(p)} for p in artifacts
    ]})


def cached(path: Path, signature: str) -> dict | None:
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("signature") != signature:
        raise ValueError(f"Stale diagnostic evidence: {path}")
    for item in value.get("artifacts", []):
        target = inside(path.parent, item["path"])
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise ValueError(f"Damaged diagnostic artifact: {target}; preserve and inspect it")
    return value


def scalar_audit(source: np.ndarray, native: np.ndarray, target: float,
                 tolerance: float = .05, max_alpha: float = 4.0) -> dict:
    """Independent SSE-space search on the exact *same* float64/uint8 expression.

    Equal (original pixel, integer residual) pairs are counted once, making an
    audit cheap without approximating MSE. No pixel editing or decoder calls.
    The bracket is an observed numerical gap, not a theorem of impossibility.
    """
    if source.dtype != np.uint8 or native.dtype != np.uint8 or source.shape != native.shape:
        raise ValueError("Need equally sized uint8 images")
    if not math.isfinite(target) or tolerance <= 0 or max_alpha < 1:
        raise ValueError("Invalid scalar search policy")
    _, old = match_psnr(source, native, target, tolerance, max_alpha, 64)
    x = source.astype(np.int32).reshape(-1)
    d = native.astype(np.int32).reshape(-1) - x
    keys, counts = np.unique(x * 511 + d + 255, return_counts=True)
    xx = (keys // 511).astype(np.float64)
    dd = (keys % 511 - 255).astype(np.float64)
    count = source.size
    desired = count * 255.0**2 * 10 ** (-target / 10)

    def evaluate(alpha):
        difference = np.rint(np.clip(xx + alpha * dd, 0, 255)) - xx
        sse = float(np.dot(difference * difference, counts))
        value = math.inf if not sse else 10 * math.log10(count * 255.0**2 / sse)
        return sse, value

    lo, hi = 0.0, float(max_alpha)
    candidates = {0.0, 1.0, hi, float(old["alpha"])}
    for _ in range(96):
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:
            break
        candidates.add(mid)
        if evaluate(mid)[0] < desired:
            lo = mid
        else:
            hi = mid
    candidates.update((lo, hi))
    # Test both sides of floating-point ties without altering scalar semantics.
    for alpha in (lo, hi, float(old["alpha"])):
        for direction in (-math.inf, math.inf):
            point = alpha
            for _ in range(8):
                point = float(np.nextafter(point, direction))
                if 0 <= point <= max_alpha:
                    candidates.add(point)
    best = min(candidates, key=lambda a: (abs(evaluate(a)[1] - target), a))
    best_psnr = evaluate(best)[1]
    hit = abs(best_psnr - target) <= tolerance
    zero = not np.any(d)
    return {
        "target": target, "tolerance": tolerance, "max_alpha": max_alpha,
        "baseline": old, "audit_alpha": best, "audit_psnr": best_psnr,
        "audit_absolute_error": abs(best_psnr - target), "within_policy": hit,
        "finding": "candidate_found" if hit else (
            "zero_residual" if zero else "observed_quantized_gap_or_alpha_bound"),
        "bracket": [{"alpha": a, "psnr": evaluate(a)[1]} for a in (lo, hi)],
        "upper_bound_psnr": evaluate(max_alpha)[1], "evaluations": len(candidates),
        "changed_pixels_individually": False, "decoder_used_for_selection": False,
    }
