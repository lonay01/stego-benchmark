"""One frozen model per subprocess; all diagnostic writes use a new result root."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from wmbench.embedding_bank import inference_identity
from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb, psnr
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside, retry_record, sealed
from wmbench_full_payload.worker import decode_metadata
from wmbench_native_canary.worker import check_png

from .core import Stages, case_key, complete, validate_case
from .probes import (array_stats, audit_model, capture_forward, decode_float,
                     export_sepwater_author, quantize, reconstruct_same_latent)


def capture(adapter, model, root, folder, case, key, execution):
    path = folder / "capture.json"
    prior = cached(root, path, key)
    if prior and prior["status"] == "saved":
        return prior
    if prior:
        retry_record(path)
    attempt = folder / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    try:
        seed_everything(case["seed"])
        png, metadata = attempt / "baseline.png", attempt / "metadata.json"
        with capture_forward(adapter, model["id"]) as observations:
            adapter.embed(str(folder / "source.png"), str(png), case["assignment"]["payload"], str(metadata))
        check_png(png, model["native_geometry"]["width"])
        if "raw" not in observations:
            raise RuntimeError("Expected encoder boundary was not observed")
        raw = observations["raw"]
        if raw.shape != load_rgb(folder / "source.png").shape:
            raise ValueError("Encoder boundary geometry differs from the historical crop")
        # Control reconstruction must use exactly the sampled original latent.
        if "latent" in observations:
            latent = observations.pop("latent")
            observations["ae"] = reconstruct_same_latent(adapter, model["id"], latent)
            observations["latent_input"] = latent.detach().float().cpu().numpy().copy()
        tensors = {}
        for name, array in observations.items():
            target = attempt / f"{name}.npy"
            if not isinstance(array, np.ndarray) or not np.isfinite(array).all():
                raise ValueError(f"Invalid captured array: {name}")
            np.save(target, array, allow_pickle=False)
            tensors[name] = target.relative_to(root).as_posix()
        decoder = decode_metadata(metadata)
        return sealed(root, path, key, {"status": "saved", "tensors": tensors,
            "png": png.relative_to(root).as_posix(), "decoder": decoder.relative_to(root).as_posix(),
            "execution": execution}, list(attempt.iterdir()))
    except Exception as error:
        sealed(root, path, key, {"status": "error", "reason": str(error),
               "traceback": traceback.format_exc(), "execution": execution})
        raise


def freeze_png(path, pixels):
    if path.exists():
        if not np.array_equal(load_rgb(path), pixels):
            raise EvidenceError(f"A diagnostic PNG changed on resume: {path}")
    else:
        save_rgb(path, pixels)
    if not np.array_equal(load_rgb(path), pixels):
        raise EvidenceError("PNG writer did not preserve the supplied uint8 pixels")


def process_case(adapter, model, root, case, identity, execution):
    key = case_key(identity, case)
    if complete(root, case, key):
        return
    folder = validate_case(root, case)
    baseline = capture(adapter, model, root, folder, case, key, execution)
    stages = Stages(root, folder, key, case["assignment"], execution)
    tensor = {name: np.load(inside(root, p), allow_pickle=False) for name, p in baseline["tensors"].items()}
    raw, source = tensor["raw"], load_rgb(folder / "source.png")
    old_decode = read_json(folder / "historical_decode.json")
    original_png, new_png = folder / "historical.png", inside(root, baseline["png"])
    decoder_meta = inside(root, baseline["decoder"])

    def file_decode(path, metadata=decoder_meta):
        seed_everything(case["seed"])
        return adapter.decode(str(path), str(metadata))

    def direct(image):
        seed_everything(case["seed"])
        return decode_float(adapter, model["id"], image)

    def historic():
        data = file_decode(original_png, folder / "decoder.json")
        return {**data, "same_bits_as_history": data["bits"] == old_decode["decoded_bits"],
                "historical_ber": old_decode["ber"], "historical_exact": old_decode["exact"],
                "_files": [original_png]}

    def replay():
        data = file_decode(new_png)
        difference = load_rgb(new_png).astype(int) - load_rgb(original_png).astype(int)
        return {**data, "same_png_sha256_as_history": sha256(new_png) == sha256(original_png),
                "same_pixels_as_history": not bool(np.any(difference)),
                "max_pixel_delta_from_history": int(np.abs(difference).max()),
                "same_bits_as_history": data["bits"] == old_decode["decoded_bits"],
                "psnr": psnr(source, load_rgb(new_png)), "_files": [new_png, decoder_meta]}

    stages.measure("historical_png", historic)
    stages.measure("baseline_png", replay)
    stages.measure("raw_float", lambda: {**direct(raw), "pixels": array_stats(raw, source)})
    stages.measure("clipped_float", lambda: direct(np.clip(raw, 0, 1)))
    round_pixels = quantize(raw, "round_even")
    memory = stages.measure("round_even_memory", lambda: direct(round_pixels.astype(np.float32) / 255))

    def serialized(rule, pixels=None):
        array = quantize(raw, rule) if pixels is None else pixels
        path = folder / "serialized" / f"{rule}.png"
        freeze_png(path, array)
        reloaded = load_rgb(path)
        direct_result = direct(reloaded.astype(np.float32) / 255)
        api_result = file_decode(path)
        result = {**api_result, "serializer": rule, "png_pixels_exact": True,
                  "same_pixels_as_baseline": np.array_equal(reloaded, load_rgb(new_png)),
                  "direct_bits": direct_result["bits"],
                  "file_api_matches_direct": api_result["bits"] == direct_result["bits"],
                  "psnr": psnr(source, reloaded), "_files": [path]}
        if rule == "round_even":
            result["direct_memory_matches_png"] = direct_result["bits"] == memory.get("decoded_bits")
        return result

    for rule in ("round_even", "floor", "half_up"):
        stages.measure(f"{rule}_png", lambda rule=rule: serialized(rule))
    stages.measure("clean_control", lambda: direct(source.astype(np.float32) / 255), control=True)
    stages.measure("wrapper_float", lambda: direct(tensor["wrapper"]) if "wrapper" in tensor else
                   {"status": "unavailable", "reason": "No distinct WIBE wrapper boundary in this adapter"})
    if "ae" in tensor:
        stages.measure("ae_float_control", lambda: direct(tensor["ae"]), control=True)
        def ae_png():
            path = folder / "serialized/ae_only.png"
            array = quantize(tensor["ae"], "round_even")
            freeze_png(path, array)
            return {**file_decode(path), "psnr_original_to_ae": psnr(source, array),
                    "psnr_original_to_marked": psnr(source, load_rgb(new_png)),
                    "psnr_ae_to_marked": psnr(array, load_rgb(new_png)),
                    "same_sampled_latent": True, "_files": [path]}
        stages.measure("ae_png_control", ae_png, control=True)
    if model["id"] == "sepwater":
        stages.measure("author_export_png", lambda: serialized("author_float64_floor",
            export_sepwater_author(tensor["signed"], source)))


def work(job):
    root, model = Path(job["output"]), job["model"]
    meta = root / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    old_identity = read_json(meta / "historical_model_identity.json")
    current_identity = inference_identity(prepared["identity"])
    if current_identity != old_identity:
        raise EvidenceError("Weights/model config differ from the historical canary; refusing a false replay")
    deps = {key: prepared["identity"][key] for key in ("lock_sha256", "installed_sha256")}
    if deps != read_json(meta / "historical_dependency_identity.json"):
        raise EvidenceError("Installed dependencies differ from the historical locked environment")
    freeze(meta / "model_identity.json", current_identity)
    freeze(meta / "dependency_identity.json", deps)
    seed_everything(0)
    adapter = adapter_instance(read_json(prepared["config"]))
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    execution = {"session": job["session"], "runtime": runtime_info(), "fingerprint": prepared["fingerprint"],
                 "diagnostic_source_code": job["source_code"], "preserved_run_identity": job["identity"]}
    old_runtime = read_json(meta / "historical_runtime.json")["runtime"]
    execution["runtime_changes"] = {key: {"historical": old_runtime.get(key), "current": value}
                                    for key, value in execution["runtime"].items() if old_runtime.get(key) != value}
    write_json(meta / "audit.json", {**audit_model(adapter, model["id"]),
                                     "runtime_changes": execution["runtime_changes"]})
    write_json(meta / "executions" / f"{job['session']}.json", execution)
    errors = 0
    for i, case in enumerate(job["cases"], 1):
        print(f"DIAG {model['id']} {i}/{len(job['cases'])}: {case['stem']}", flush=True)
        try:
            process_case(adapter, model, root, case, job["identity"], execution)
            if not complete(root, case, case_key(job["identity"], case)):
                errors += 1
        except EvidenceError:
            raise
        except Exception as error:
            errors += 1
            print(f"DIAG ERROR: {type(error).__name__}: {error}", flush=True)
    if errors:
        raise RuntimeError(f"{errors} incomplete diagnostic cases; completed stages remain saved")
