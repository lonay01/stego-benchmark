"""Full payload metrics, with exact message recovery kept separate from BER."""

import io
import math
import numpy as np
from PIL import Image


def bits_score(logits, expected):
    logits = np.asarray(logits).reshape(-1)
    expected = np.asarray(expected, dtype=np.uint8).reshape(-1)
    if len(logits) != len(expected) or len(expected) == 0 or not np.isfinite(logits).all():
        raise ValueError("Wrong or nonfinite decoder output; it is not a valid BER result")
    if not np.isin(expected, [0, 1]).all():
        raise ValueError("Expected a binary payload")
    bits = (logits > 0).astype(np.uint8)
    errors = int((bits != expected).sum())
    return {"nbits": len(bits), "errors": errors, "ber": errors / len(bits),
            "exact": errors == 0, "word_accuracy_threshold_0_2": errors / len(bits) < 0.2,
            "bits": bits.tolist(), "logits": logits.astype(float).tolist()}


def psnr(a, b, data_range=1.0):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Quality metrics require same-shape finite images")
    mse = float(np.mean((a - b) ** 2))
    # JSON cannot represent infinity. An explicit flag distinguishes perfect equality.
    return {"mse": mse, "psnr_db": None if mse == 0 else 10 * math.log10(data_range**2 / mse),
            "identical_pixels": mse == 0}


def png_bytes(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected RGB uint8")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    result = stream.getvalue()
    if not np.array_equal(np.asarray(Image.open(io.BytesIO(result))), pixels):
        raise ValueError("PNG roundtrip altered integer pixels")
    return result


def pixels(tensor, rule="floor"):
    array = tensor.detach().float().cpu().numpy()[0].transpose(1, 2, 0)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite image")
    scaled = np.clip(array, 0, 1) * 255
    if rule == "round_even":
        scaled = np.rint(scaled)
    elif rule == "half_up":
        scaled = np.floor(scaled + .5)
    elif rule != "floor":
        raise ValueError(rule)
    return scaled.astype(np.uint8)


def quality(a, b):
    from skimage.metrics import structural_similarity
    result = psnr(a, b, 255.0)
    result["ssim"] = float(structural_similarity(a, b, channel_axis=2, gaussian_weights=True,
                              sigma=1.5, use_sample_covariance=False, data_range=255))
    return result
