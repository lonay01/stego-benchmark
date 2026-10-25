"""Small, testable data contracts; no torch or model imports here."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = ("invismark", "chunkyseal")
ATTACK_IDS = ("vae_1", "vae_2", "sdxl_0.15_r1", "ip2p_lighting_r1")
CONTROLS = ("clean", "png_roundtrip", "normalization_roundtrip", "resize_only",
            "jpeg95", "unwatermarked")
BASE_SHA = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    with open(path, "rb") as stream:
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """Atomic replacement of this diagnostic's own mutable summaries only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    if path.read_bytes() != data:
        raise RuntimeError(f"Readback failed: {path}")


def commit(path, value):
    """First observation wins; changed evidence is not silently replaced."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Evidence changed: {path}; keep old results, use a new diagnostic root")
    else:
        write(path, value)


def immutable_bytes(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"Saved artifact changed: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(data)
        tmp.replace(path)
    if path.read_bytes() != data:
        raise RuntimeError(f"Artifact readback failed: {path}")
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Escaped path: {relative}")
    return path


def separate_paths(*paths):
    resolved = [Path(p).expanduser().resolve() for p in paths]
    for i, left in enumerate(resolved):
        for right in resolved[i + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Bank, old neural results, new output and scratch must be disjoint")
    return resolved


def rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"Expected an RGB PNG, got {im.mode}: {path}")
        array = np.array(im)
    if array.dtype != np.uint8 or array.shape != (512, 512, 3):
        raise ValueError(f"This two-method check requires the existing 512 RGB8 bank: {array.shape}")
    return array


def pixel_stats(array):
    a = np.asarray(array)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite pixels/tensor")
    return dict(shape=list(a.shape), dtype=str(a.dtype), min=float(a.min()),
                max=float(a.max()), mean=float(a.mean()), std=float(a.std()),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def png_bytes(array):
    a = np.asarray(array)
    if a.dtype != np.uint8 or a.shape != (512, 512, 3):
        raise ValueError("PNG boundary requires RGB512 uint8")
    output = io.BytesIO()
    Image.fromarray(a).save(output, format="PNG", compress_level=3)
    return output.getvalue()


def png_pixels(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"))


def normalization_roundtrip(array):
    normalized = array.astype(np.float32) / 255 * 2 - 1
    return np.rint(np.clip((normalized + 1) / 2, 0, 1) * 255).astype(np.uint8)


def control_pixels(name, carrier, source):
    if name in {"clean", "png_roundtrip"}:
        return png_pixels(png_bytes(carrier))
    if name == "normalization_roundtrip":
        return normalization_roundtrip(carrier)
    if name == "resize_only":
        return np.array(Image.fromarray(carrier).resize((512, 512), Image.Resampling.BICUBIC))
    if name == "unwatermarked":
        return source.copy()
    if name == "jpeg95":
        stream = io.BytesIO()
        Image.fromarray(carrier).save(stream, format="JPEG", quality=95, subsampling=0)
        return png_pixels(stream.getvalue())
    raise ValueError(name)


def verify_artifacts(root, document):
    for relative, expected in document.get("artifacts", {}).items():
        if sha(inside(root, relative)) != expected:
            raise RuntimeError(f"Corrupt diagnostic artifact: {relative}")


def compare_history(bits, png_sha, previous):
    if previous is None:
        return {"status": "no_historical_row"}
    old_sha = previous.get("image_sha256")
    old_bits = previous.get("decoded_bits")
    return dict(status="same_bytes" if old_sha == png_sha else "different_bytes",
                historical_png_sha256=old_sha, current_png_sha256=png_sha,
                historical_ber=previous.get("ber"),
                bits_equal=bits == old_bits if old_sha == png_sha else None)


def same_clean(fresh, historical):
    return all(fresh.get(k) == historical.get(k) for k in ("status", "decoded_bits", "ber", "exact"))


def report_state(rows, expected):
    if len(rows) != expected:
        return "INCOMPLETE"
    if any(r.get("status") == "error" for r in rows):
        return "TECHNICAL_ERROR"
    if any(r.get("io_agrees") is False or r.get("repeat_equal") is False
           or r.get("history", {}).get("bits_equal") is False for r in rows):
        return "REVIEW_IO_OR_REPLAY"
    if any(r.get("history", {}).get("status") == "different_bytes" for r in rows):
        return "IO_AGREEMENT_REPLAY_BYTES_DIFFER"
    return "IO_AGREEMENT"
