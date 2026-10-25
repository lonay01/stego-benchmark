"""Frozen attack matrix. No method, mode or run-size dependent randomness."""

import hashlib
import json

POLICY = {
    "schema": "digital-channel-v1", "seed": 20261004,
    "interpolation": "Pillow bicubic", "rotation": "fixed canvas, black fill",
    "blur": "scipy gaussian_filter spatial axes, reflect, truncate=3",
    "quantization": "clip [0,255], numpy rint to uint8 after each operation",
    "jpeg": "Pillow RGB, subsampling=2, optimize=False, progressive=False",
    "noise": "PCG64 float64 512x512x3 standard normal, central native crop",
    "crop": "round(width*sqrt(area)), round(height*sqrt(area)); shared uniform location",
    "resize": "round dimensions, minimum 1; restore original dimensions",
    "compound_order": ["brightness", "rotation", "resize", "noise", "jpeg"],
    "storage": "ZIP_STORED RGB8 PNG; decode serialized pixels; no re-embedding",
    "bootstrap": {"unit": "image", "resamples": 2000, "seed": 20261004},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def matrix():
    result = [{"id": "clean", "family": "clean", "level": "clean", "repeat": 0}]
    levels = {
        "jpeg": [90, 70, 50, 30], "noise": [.005, .01, .02, .04],
        "blur": [.5, 1, 1.5, 2], "brightness": [.7, .85, 1.15, 1.3],
        "crop": [.9, .75, .5], "resize": [.75, .5, .25],
        "rotation": [-15, -10, -5, -2, 2, 5, 10, 15], "compound": [1, 2, 3, 4],
    }
    for family, values in levels.items():
        for value in values:
            for repeat in range(5 if family in {"noise", "crop", "compound"} else 1):
                label = str(value)
                result.append(dict(id=f"{family}_{label}_r{repeat + 1}", family=family,
                                   level=label, value=value, repeat=repeat + 1))
    assert len(result) == 79
    return result


ATTACKS = matrix()
BY_ID = {a["id"]: a for a in ATTACKS}


def seed(image_id, message_replicate, attack_id):
    return int(digest([POLICY["seed"], int(image_id), message_replicate, attack_id])[:16], 16)
