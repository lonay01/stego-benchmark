"""Encode once, score the complete payload, checkpoint every image/mode/check."""

from __future__ import annotations

import contextlib
import time
import traceback
from pathlib import Path

from PIL import Image

from wmbench.embedding_bank import inference_identity
from wmbench.io import digest, read_json, write_json
from wmbench.matching import load_rgb, match_psnr, psnr, save_rgb
from wmbench.worker import adapter_instance, runtime_info, seed_everything

from .contracts import MODES, POLICY, message, score
from .storage import EvidenceError, cached, freeze, inside, retry_record, sealed


def task_id(image, replicate):
    return f"{int(image['image_id']):012d}_m{replicate:02d}"


def signature(model, image, replicate):
    return digest({"model": model, "image_sha256": image["sha256"],
                   "message": message(model, image["image_id"], replicate), "policy": POLICY})


def record_path(root, model, stem, stage):
    return root / model["id"] / "_meta/records" / f"{stem}_{stage}.json"


def check_png(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"Expected RGB512 PNG, got {image.mode}/{image.size}: {path}")
        image.load()


def decode_metadata(metadata):
    """Remove reference answers from the metadata supplied to blind adapters."""
    row = read_json(metadata)
    for key in ("expected_bits", "full_payload", "payload"):
        if isinstance(row.get(key), list):
            row[key] = [0] * len(row[key])
    target = metadata.with_name("decoder.json")
    freeze(target, row)
    return target


def decode_saved(adapter, root, model, image, replicate, mode, assignment, key, png, metadata, execution):
    stem = task_id(image, replicate)
    path = record_path(root, model, stem, mode + "_decode")
    prior = cached(root, path, key)
    if prior and prior["status"] != "error":
        return prior
    if prior:
        retry_record(path)
    started = time.monotonic()
    try:
        # Caller RNG state and expected bits cannot choose a better decoding.
        seed_everything(int(key[:8], 16))
        result = adapter.decode(str(png), str(metadata))
        row = score(result, assignment)
    except Exception as error:
        row = {"status": "error", "reason": f"{type(error).__name__}: {error}",
               "traceback": traceback.format_exc(), "ber": None, "exact": None}
    return sealed(root, path, key, {
        **row, "mode": mode, "image_id": image["image_id"], "replicate": replicate,
        "seconds": time.monotonic() - started, "execution": execution,
    }, [png, metadata])


@contextlib.contextmanager
def observe_tensor_boundary(adapter, assignment, observations):
    """Observe supported tensor serializers without changing the encoded image.

    File/PIL-only APIs are reported unavailable, never reconstructed by reading
    the PNG and calling it a float-before-save measurement. SSL has a different
    extractor and is intentionally not routed through the tensor decoder.
    """
    if not hasattr(adapter, "_save") or not hasattr(adapter, "_extract_logits") or hasattr(adapter, "carrier"):
        observations["pre_png"] = {"status": "unavailable", "reason": "Adapter does not expose a compatible float-image decoder"}
        yield
        return
    original = adapter._save

    def observed(tensor, target):
        try:
            with adapter.torch.inference_mode():
                logits = adapter._extract_logits(tensor.detach()).reshape(-1)
            if not adapter.torch.isfinite(logits).all():
                raise ValueError("Non-finite pre-PNG logits")
            observations["pre_png"] = score({"bits": logits.gt(adapter.threshold).int().cpu().tolist()}, assignment)
        except Exception as error:
            observations["pre_png"] = {"status": "error", "reason": str(error)}
        return original(tensor, target)

    adapter._save = observed
    try:
        yield
    finally:
        adapter._save = original


def process_pair(adapter, model, root, image, replicate, execution):
    root = Path(root).resolve()
    assignment = message(model, image["image_id"], replicate)
    stem, key = task_id(image, replicate), signature(model, image, replicate)
    source = inside(root, image["path"])
    native_path = record_path(root, model, stem, "native")
    native = cached(root, native_path, key)
    if native and native["status"] == "error":
        retry_record(native_path)
        native = None
    if native is None:
        started = time.monotonic()
        pending = root / model["id"] / "_meta/attempts" / f"{stem}_{time.time_ns()}"
        pending.mkdir(parents=True)
        output, metadata = pending / "native.png", pending / "metadata.json"
        observations = {}
        try:
            seed_everything(int(key[:8], 16))
            with observe_tensor_boundary(adapter, assignment, observations):
                adapter.embed(str(source), str(output), assignment["payload"], str(metadata))
            check_png(output)
            if not metadata.is_file():
                raise ValueError("Adapter did not save decoding metadata")
            # Add a scoring contract without altering the author's reserved bits/keys.
            raw = read_json(metadata)
            raw.update(full_payload_contract=assignment["capacity"], full_payload_policy=POLICY["schema"])
            write_json(metadata, raw)
            decoder = decode_metadata(metadata)
            target = root / model["id"] / "native" / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Only uncommitted output of this task can be replaced here.
            output.replace(target)
            artifacts = [target, *sorted(p for p in pending.rglob("*") if p.is_file())]
            native = sealed(root, native_path, key, {
                "status": "saved", "image_id": image["image_id"], "replicate": replicate,
                "mode": "native", "output": target.relative_to(root).as_posix(),
                "decoder_metadata": decoder.relative_to(root).as_posix(),
                "assignment": assignment, "pre_serialization": observations,
                "psnr": psnr(load_rgb(source), load_rgb(target)),
                "seconds": time.monotonic() - started, "execution": execution,
            }, artifacts)
        except Exception as error:
            sealed(root, native_path, key, {"status": "error", "reason": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc(), "execution": execution})
            raise
    png = inside(root, native["output"])
    metadata = inside(root, native["decoder_metadata"])
    check_png(png)
    x, y = load_rgb(source), load_rgb(png)
    rows = []
    for mode in MODES:
        if mode == "native":
            encoded = native
        else:
            path = record_path(root, model, stem, mode)
            encoded = cached(root, path, key)
            if encoded is None:
                target_psnr = int(mode.split("_")[1])
                result, match = match_psnr(x, y, target_psnr, POLICY["tolerance"], POLICY["max_alpha"], POLICY["iterations"])
                artifacts = []
                encoded = {"status": "unmatched", "matching": match, "mode": mode,
                           "image_id": image["image_id"], "replicate": replicate, "execution": execution}
                if match["status"] == "ok":
                    target = root / model["id"] / mode / f"{stem}.png"
                    save_rgb(target, result)
                    measured = psnr(x, load_rgb(target))
                    if abs(measured - target_psnr) > POLICY["tolerance"]:
                        raise EvidenceError("Saved matched PNG no longer satisfies its scalar target")
                    encoded.update(status="saved", output=target.relative_to(root).as_posix(), psnr=measured)
                    artifacts = [target]
                encoded = sealed(root, path, key, encoded, artifacts)
        if encoded["status"] == "saved":
            rows.append(decode_saved(adapter, root, model, image, replicate, mode, assignment, key,
                                     inside(root, encoded["output"]), metadata, execution))
        else:
            rows.append(encoded)
    return rows


class FullStages:
    """Reuse existing FIN/HiDDeN/DistSeal stage operations, not their common30 scorer."""
    def __init__(self, root, assignment, key):
        self.root, self.assignment, self.key = root, assignment, key
        self.root.mkdir(parents=True, exist_ok=True)
        self.payload = assignment["payload"]
        self.rows = []

    def measure(self, name, operation, *, control=False):
        key = digest({"case": self.key, "stage": name})
        path = self.root / f"{name}.json"
        old = cached(self.root, path, key)
        if old and old["status"] != "error":
            self.rows.append(old)
            return old
        if old:
            retry_record(path)
        try:
            data = operation()
            artifacts = data.pop("_files", [])
            result = ({**{k: v for k, v in data.items() if k != "bits"}, **score(data, self.assignment)}
                      if "bits" in data else {"status": "measured", **data})
        except Exception as error:
            result, artifacts = {"status": "error", "reason": str(error)}, []
        row = sealed(self.root, path, key, {**result, "stage": name, "control": control}, artifacts)
        self.rows.append(row)
        print(f"WM_DIAG FULL stage {name}: {row['status']}; BER={row.get('ber')}", flush=True)
        return row


def extra_diagnostics(adapter, root, model, image, execution):
    if model["id"] not in ("fin", "hidden_benchmark", "distseal_posthoc_dcae"):
        return True
    # The old diagnostic module is bundled unchanged; only its stage operations
    # are reused with this full-message scorer and new, independent paths.
    from bank_diagnostics.worker import fin_stages, hidden_stages, distseal_stages

    key = signature(model, image, 1)
    folder = root / model["id"] / "_meta/diagnostics" / task_id(image, 1)
    completion = folder / "complete.json"
    if cached(root, completion, key):
        return True
    stages = FullStages(folder, message(model, image["image_id"], 1), key)
    operation = {"fin": fin_stages, "hidden_benchmark": hidden_stages,
                 "distseal_posthoc_dcae": distseal_stages}[model["id"]]
    seed_everything(int(key[:8], 16))
    try:
        operation(adapter, inside(root, image["path"]), stages)
        if any(row["status"] == "error" for row in stages.rows):
            return False
        sealed(root, completion, key, {"status": "measured", "execution": execution},
               [p for p in folder.iterdir() if p.is_file() and p != completion])
        return True
    except EvidenceError:
        raise
    except Exception as error:
        write_json(folder / "technical_error.json", {"error": str(error), "traceback": traceback.format_exc()})
        return False


def work(job):
    root, model = Path(job["output"]), job["model"]
    meta = root / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    freeze(meta / "model_identity.json", inference_identity(prepared["identity"]))
    # Keep dependency resolution stable on resume while recording OS/GPU separately.
    freeze(meta / "dependency_identity.json", {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")})
    seed_everything(0)
    print(f"WM_DIAG FULL loading {model['id']}", flush=True)
    adapter = adapter_instance(read_json(prepared["config"]))
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    expected = int(model["native_bits"])
    if hasattr(adapter, "native_bits") and adapter.native_bits != expected:
        raise ValueError(f"Checkpoint exposes {adapter.native_bits} bits, registry declares {expected}")
    execution = {"session": job["session"], "fingerprint": prepared["fingerprint"], "runtime": runtime_info()}
    write_json(meta / "executions" / f"{job['session']}.json", execution)
    technical_failures = 0
    for index, image in enumerate(job["images"], 1):
        for replicate in range(1, POLICY["replicates"] + 1):
            try:
                rows = process_pair(adapter, model, root, image, replicate, execution)
                technical_failures += sum(row["status"] == "error" for row in rows)
                native = rows[0]
                print(f"WM_DIAG FULL {model['id']} image {index}/{len(job['images'])} m{replicate}/5: "
                      f"Native {native['status']}, BER={native.get('ber')}", flush=True)
            except EvidenceError:
                raise
            except Exception as error:
                technical_failures += 1
                print(f"WM_DIAG FULL ERROR {model['id']} {task_id(image, replicate)}: {error}", flush=True)
        if not extra_diagnostics(adapter, root, model, image, execution):
            technical_failures += 1
    if technical_failures:
        raise RuntimeError(f"{technical_failures} technical failures; sealed successes retained; inspect records/logs")
