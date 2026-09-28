"""Frozen historical cases and durable diagnostic observations."""

from __future__ import annotations

import time
import traceback

import numpy as np

from wmbench.io import atomic_bytes, digest, read_json, sha256
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside, retry_record, sealed
from wmbench_native_canary.contracts import message, score
from wmbench_native_canary.worker import check_png, signature, record_path, task_id

METHODS = ("hidden_benchmark", "rosteals", "sepwater", "sepmark", "distseal_posthoc_dcae")
POLICY = {
    "schema": "native-five-diagnostics-v1",
    "cases": "all_5_images_x_5_historical_messages_per_method",
    "reference": "same_checkpoint_payload_seed_and_crop_as_native_canary",
    "changes": "diagnostic_observations_only_no_production_changes",
    "serializers": ["round_even", "floor", "half_up"],
    "controls": ["clean_input", "same_latent_autoencoder_only"],
    "attacks": [], "matched_psnr": False, "training": False,
}


def stage_names(model):
    stages = ["historical_png", "baseline_png", "raw_float", "clipped_float",
              "round_even_memory", "round_even_png", "floor_png", "half_up_png",
              "clean_control", "wrapper_float"]
    if model in ("rosteals", "sepwater", "distseal_posthoc_dcae"):
        stages += ["ae_float_control", "ae_png_control"]
    if model == "sepwater":
        stages += ["author_export_png"]
    return stages


def copy_frozen(source, target):
    """Copy only verified evidence; a conflicting destination is never replaced."""
    data = source.read_bytes()
    if target.exists() and sha256(target) != sha256(source):
        raise EvidenceError(f"Changed evidence: {target}")
    if not target.exists():
        atomic_bytes(target, data)


def freeze_cases(bank, output, model, images):
    cases = []
    for image in images:
        for replicate in range(1, 6):
            stem = task_id(image, replicate)
            enc = cached(bank, record_path(bank, model, stem, "native"), signature(model, image, replicate))
            dec = cached(bank, record_path(bank, model, stem, "native_decode"), signature(model, image, replicate))
            if not enc or enc["status"] != "saved" or not dec or dec["status"] != "measured":
                raise EvidenceError(f"Historical case incomplete: {model['id']}/{stem}")
            assignment = message(model, image["image_id"], replicate)
            if enc["assignment"] != assignment or dec["expected_bits"] != assignment["payload"]:
                raise EvidenceError("Historical payload does not match the frozen protocol")
            rescored = score({"bits": dec["decoded_bits"]}, assignment)
            if rescored["ber"] != dec["ber"] or rescored["exact"] != dec["exact"]:
                raise EvidenceError("Historical decoder metrics are inconsistent")
            original = inside(bank, image["path"])
            if sha256(original) != image["sha256"]:
                raise EvidenceError("Historical original changed")
            crop = inside(bank, enc["source_crop"])
            size = model["native_geometry"]["width"]
            check_png(crop, size)
            # Validate the literal center crop, not just its file name.
            from PIL import Image
            from wmbench_native_canary.contracts import center_crop
            with Image.open(original) as im, Image.open(crop) as cut:
                if not np.array_equal(np.asarray(center_crop(im, size)), np.asarray(cut)):
                    raise EvidenceError("Historical crop is not the frozen central pixels")
            folder = output / model["id"] / "cases" / stem
            for source, filename in ((crop, "source.png"), (inside(bank, enc["output"]), "historical.png"),
                                     (inside(bank, enc["decoder_metadata"]), "decoder.json")):
                copy_frozen(source, folder / filename)
            # Verify that decoder metadata does not carry the reference answer.
            metadata = read_json(folder / "decoder.json")
            for key in ("expected_bits", "full_payload", "payload"):
                if isinstance(metadata.get(key), list) and any(metadata[key]):
                    raise EvidenceError("Historical decoder metadata exposes expected bits")
            freeze(folder / "historical_encode.json", enc)
            freeze(folder / "historical_decode.json", dec)
            case = {"model": model["id"], "stem": stem, "image": image, "replicate": replicate,
                    "assignment": assignment, "seed": int(enc["signature"][:8], 16),
                    "source_sha256": sha256(crop), "historical_sha256": sha256(folder / "historical.png"),
                    "historical_encode_sha256": digest(enc), "historical_decode_sha256": digest(dec),
                    "decoder_sha256": sha256(folder / "decoder.json")}
            freeze(folder / "case.json", case)
            cases.append(case)
    return cases


def validate_case(root, case):
    folder = root / case["model"] / "cases" / case["stem"]
    for name, key in (("source.png", "source_sha256"), ("historical.png", "historical_sha256"),
                      ("decoder.json", "decoder_sha256")):
        if sha256(folder / name) != case[key]:
            raise EvidenceError(f"Changed case artifact: {folder / name}")
    for name, key in (("historical_encode.json", "historical_encode_sha256"),
                      ("historical_decode.json", "historical_decode_sha256")):
        if digest(read_json(folder / name)) != case[key]:
            raise EvidenceError(f"Changed historical record: {name}")
    return folder


class Stages:
    def __init__(self, root, folder, key, assignment, execution):
        self.root, self.folder, self.key = root, folder, key
        self.assignment, self.execution = assignment, execution

    def measure(self, name, operation, control=False):
        path = self.folder / "stages" / f"{name}.json"
        key = digest({"case": self.key, "stage": name})
        old = cached(self.root, path, key)
        if old and old["status"] != "error":
            return old
        if old:
            retry_record(path)
        started = time.monotonic()
        try:
            data = operation()
            artifacts = data.pop("_files", [])
            if "bits" in data:
                metrics = score(data, self.assignment)
                data = {**data, **metrics}
                data.pop("bits", None)
            else:
                data.setdefault("status", "measured")
        except EvidenceError:
            raise
        except Exception as error:
            artifacts = []
            data = {"status": "error", "reason": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()}
        row = sealed(self.root, path, key, {**data, "stage": name, "control": control,
                     "seconds": time.monotonic() - started, "execution": self.execution}, artifacts)
        print(f"  {name}: {row['status']}, BER={row.get('ber', '-')}", flush=True)
        return row


def case_key(run_identity, case):
    return digest({"run": run_identity, "case": case})


def complete(root, case, key):
    folder = validate_case(root, case)
    baseline = cached(root, folder / "capture.json", key)
    if not baseline or baseline["status"] != "saved":
        return False
    for name in stage_names(case["model"]):
        row = cached(root, folder / "stages" / f"{name}.json", digest({"case": key, "stage": name}))
        if row is None or row["status"] not in ("measured", "rejected", "unavailable"):
            return False
    return True
