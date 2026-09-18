"""One scalar residual adjustment; never select pixels using decoder outcomes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, array: np.ndarray) -> None:
    import io
    from .io import atomic_bytes

    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Output must be an RGB uint8 array")
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    atomic_bytes(path, stream.getvalue())


def psnr(source: np.ndarray, target: np.ndarray) -> float:
    if source.shape != target.shape:
        raise ValueError("PSNR requires exactly the same image grid")
    mse = np.mean((source.astype(np.float64) - target.astype(np.float64)) ** 2)
    return math.inf if mse == 0 else float(10 * math.log10(255**2 / mse))


def match_psnr(
    source: np.ndarray,
    native: np.ndarray,
    target: float = 42,
    tolerance: float = 0.05,
    max_alpha: float = 4,
    iterations: int = 64,
) -> tuple[np.ndarray, dict]:
    if source.dtype != np.uint8 or native.dtype != np.uint8 or source.shape != native.shape:
        raise ValueError("Matched-PSNR starts from saved, equally sized uint8 PNGs")
    x = source.astype(np.float64)
    residual = native.astype(np.float64) - x

    def candidate(alpha: float):
        unbounded = x + alpha * residual
        output = np.rint(np.clip(unbounded, 0, 255)).astype(np.uint8)
        value = psnr(source, output)
        return output, value, float(np.mean((unbounded < 0) | (unbounded > 255)))

    lo, hi = 0.0, float(max_alpha)
    best_alpha = 1.0
    best, best_psnr, clipped = candidate(best_alpha)
    for alpha in [0.0, hi]:
        image, value, fraction = candidate(alpha)
        if abs(value - target) < abs(best_psnr - target):
            best, best_psnr, clipped, best_alpha = image, value, fraction, alpha
    for _ in range(iterations):
        alpha = (lo + hi) / 2
        image, value, fraction = candidate(alpha)
        if abs(value - target) < abs(best_psnr - target):
            best, best_psnr, clipped, best_alpha = image, value, fraction, alpha
        if value > target:
            lo = alpha
        else:
            hi = alpha
    success = abs(best_psnr - target) <= tolerance
    return best, {
        "status": "ok" if success else "match_not_found_within_policy",
        "native_psnr": psnr(source, native),
        "psnr": best_psnr,
        "alpha": best_alpha,
        "clipped_fraction": clipped,
        "target": target,
        "tolerance": tolerance,
        "max_alpha": max_alpha,
        "policy": "saved_native_u8_scalar_bisection_v1",
    }
