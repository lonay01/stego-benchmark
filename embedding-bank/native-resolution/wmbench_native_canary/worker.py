"""One model per process, one immutable crop/message/PNG checkpoint per pair."""

from __future__ import annotations

import contextlib
import random
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.embedding_bank import inference_identity
from wmbench.io import digest, read_json, sha256, write_json
from wmbench.matching import load_rgb, psnr
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside, retry_record, sealed
from wmbench_full_payload.worker import decode_metadata, decode_saved, record_path, task_id

from .contracts import POLICY, center_crop, crop_box, message, score


def signature(model, image, replicate):
    return digest({"model": model, "source_sha256": image["sha256"], "policy": POLICY,
                   "message": message(model, image["image_id"], replicate)})


def check_png(path, size):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (size, size):
            raise ValueError(f"Expected RGB PNG {size}x{size}, got {image.mode}/{image.size}")
        image.load()


def freeze_crop(root, model, image):
    """Freeze pixels and source linkage once, independently of messages."""
    size = model["native_geometry"]["width"]
    source = inside(root, image["path"])
    if sha256(source) != image["sha256"]:
        raise EvidenceError("Original source image changed")
    folder = root / "_inputs/crops" / str(size)
    path = folder / f"{int(image['image_id']):012d}.png"
    record = path.with_suffix(".json")
    key = digest({"source": image["sha256"], "box": crop_box(size), "rule": POLICY["input"]})
    old = cached(root, record, key)
    if old:
        check_png(path, size)
        return path
    with Image.open(source) as original:
        crop = center_crop(original, size)
        folder.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        crop.save(temp, format="PNG")
        temp.replace(path)
    sealed(root, record, key, {"status": "saved", "source": image,
           "crop_box_xyxy": crop_box(size), "size": size, "resampling": "none"}, [path])
    return path


@contextlib.contextmanager
def observe_boundary(adapter, assignment, observations):
    """Observers must neither alter RNG state nor choose a better decoding."""
    observations["pre_png"] = {"status": "unavailable", "boundary": "not_exposed",
        "reason": "Selected adapter exposes file/PIL IO, not a separately decodable float-image boundary"}

    def observe(operation, boundary):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch = getattr(adapter, "torch", None)
        cuda_devices = list(range(torch.cuda.device_count())) if torch is not None and torch.cuda.is_available() else []
        rng = torch.random.fork_rng(devices=cuda_devices) if torch is not None else contextlib.nullcontext()
        try:
            with rng:
                value = score(operation(), assignment)
            observations["pre_png"] = {**value, "boundary": boundary}
        except Exception as error:
            observations["pre_png"] = {"status": "error", "reason": str(error), "boundary": boundary}
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    adapter.before_png = observe
    original = getattr(adapter, "_save", None)
    compatible = original is not None and hasattr(adapter, "_extract_logits") and not hasattr(adapter, "carrier")
    if compatible:
        def save_observed(tensor, path):
            def decode_float():
                with adapter.torch.inference_mode():
                    logits = adapter._extract_logits(tensor.detach()).reshape(-1)
                if not adapter.torch.isfinite(logits).all():
                    raise ValueError("Non-finite pre-PNG logits")
                return {"bits": logits.gt(adapter.threshold).int().cpu().tolist()}
            observe(decode_float, "adapter_tensor_before_png")
            return original(tensor, path)
        adapter._save = save_observed
    try:
        yield
    finally:
        if compatible:
            adapter._save = original
        del adapter.before_png


def process_pair(adapter, model, root, image, replicate, execution):
    root = Path(root).resolve()
    source = freeze_crop(root, model, image)
    size = model["native_geometry"]["width"]
    assignment = message(model, image["image_id"], replicate)
    stem, key = task_id(image, replicate), signature(model, image, replicate)
    path = record_path(root, model, stem, "native")
    native = cached(root, path, key)
    if native and native["status"] == "error":
        retry_record(path)
        native = None
    if native is None:
        started = time.monotonic()
        pending = root / model["id"] / "_meta/attempts" / f"{stem}_{time.time_ns()}"
        pending.mkdir(parents=True)
        png, metadata = pending / "native.png", pending / "metadata.json"
        observations = {}
        try:
            seed_everything(int(key[:8], 16))
            with observe_boundary(adapter, assignment, observations):
                adapter.embed(str(source), str(png), assignment["payload"], str(metadata))
            check_png(png, size)
            raw = read_json(metadata)
            raw.update(native_geometry=model["native_geometry"], full_payload_contract=assignment["capacity"],
                       canary_policy=POLICY["schema"])
            write_json(metadata, raw)
            decoder = decode_metadata(metadata)
            target = root / model["id"] / "native" / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            png.replace(target)
            artifacts = [source, target, *sorted(p for p in pending.rglob("*") if p.is_file())]
            quality = psnr(load_rgb(source), load_rgb(target))
            native = sealed(root, path, key, {"status": "saved", "image_id": image["image_id"],
                "replicate": replicate, "mode": "native", "output": target.relative_to(root).as_posix(),
                "decoder_metadata": decoder.relative_to(root).as_posix(), "assignment": assignment,
                "pre_serialization": observations, "size": size,
                "source_crop": source.relative_to(root).as_posix(), "psnr": quality,
                "seconds": time.monotonic() - started, "execution": execution}, artifacts)
        except Exception as error:
            sealed(root, path, key, {"status": "error", "reason": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc(), "execution": execution})
            raise
    check_png(inside(root, native["output"]), size)
    return decode_saved(adapter, root, model, image, replicate, "native", assignment, key,
        inside(root, native["output"]), inside(root, native["decoder_metadata"]), execution)


def complete(root, model, images):
    if model.get("canary_exclusion"):
        return False
    for image in images:
        for replicate in range(1, POLICY["replicates"] + 1):
            key, stem = signature(model, image, replicate), task_id(image, replicate)
            for stage, statuses in (("native", ("saved",)), ("native_decode", ("measured", "rejected"))):
                row = cached(root, record_path(root, model, stem, stage), key)
                if row is None or row["status"] not in statuses:
                    return False
    return True


def work(job):
    from .adapters import validate_geometry

    root, model = Path(job["output"]), job["model"]
    meta = root / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    freeze(meta / "model_identity.json", inference_identity(prepared["identity"]))
    freeze(meta / "dependency_identity.json", {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")})
    seed_everything(0)
    print(f"NATIVE loading {model['id']}", flush=True)
    adapter = adapter_instance(read_json(prepared["config"]))
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    if hasattr(adapter, "native_bits") and adapter.native_bits != int(model["native_bits"]):
        raise ValueError("Checkpoint message length differs from the selected contract")
    geometry = validate_geometry(adapter, model)
    freeze(meta / "geometry.json", geometry)
    execution = {"session": job["session"], "fingerprint": prepared["fingerprint"], "runtime": runtime_info()}
    write_json(meta / "executions" / f"{job['session']}.json", execution)
    errors = 0
    for index, image in enumerate(job["images"], 1):
        for replicate in range(1, POLICY["replicates"] + 1):
            try:
                row = process_pair(adapter, model, root, image, replicate, execution)
                errors += row["status"] == "error"
                print(f"NATIVE {model['id']} image {index}/{len(job['images'])} message {replicate}/5: "
                      f"{row['status']} BER={row.get('ber')} exact={row.get('exact')}", flush=True)
            except EvidenceError:
                raise
            except Exception as error:
                errors += 1
                print(f"NATIVE ERROR {model['id']} {task_id(image, replicate)}: {error}", flush=True)
    if errors:
        raise RuntimeError(f"{errors} technical errors; successful measurements remain saved")
