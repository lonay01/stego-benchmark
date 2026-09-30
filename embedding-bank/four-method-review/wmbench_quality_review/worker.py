"""One model and one predeclared profile per fresh process."""

from pathlib import Path
import time
import traceback

import numpy as np

from wmbench.embedding_bank import inference_identity
from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, psnr, save_rgb
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench_full_payload.storage import EvidenceError, freeze, inside, retry_record, sealed
from wmbench_full_payload.worker import decode_metadata
from wmbench_native_canary.contracts import score
from wmbench_native_diagnostics.core import validate_case
from wmbench_native_diagnostics.probes import audit_model, array_stats, quantize, reconstruct_same_latent
from wmbench_author_reference.core import verify_sources

from .core import carrier, complete, row_path, task_key
from .probes import capture, decode, direct_latent, editguard_checkpoint, load_clean_checkpoint


def scored(decoded, case):
    metrics = score(decoded, case["assignment"])
    if metrics["status"] != "measured":
        raise EvidenceError(f"Invalid full-payload decoder result: {metrics}")
    return metrics


def measure(adapter, job, case, profile, attempt):
    root, bank = Path(job["output"]), Path(job["bank"])
    folder, name = validate_case(root, case), case["model"]
    original = inside(bank, case["image"]["path"])
    if sha256(original) != case["image"]["sha256"]:
        raise EvidenceError("Original COCO input changed")
    source = carrier(load_rgb(original), load_rgb(folder / "source.png"), profile)
    input_path = attempt / "input.png"
    save_rgb(input_path, source)
    native_png, metadata = attempt / "native.png", attempt / "metadata.json"
    seed_everything(case["seed"])
    api_applicable = profile in ("historical", "clean_checkpoint")
    if api_applicable:
        with capture(adapter, name) as data:
            adapter.embed(str(input_path), str(native_png), case["assignment"]["payload"], str(metadata))
        blind = decode_metadata(metadata)
        if "latent" in data:
            data["ae"] = reconstruct_same_latent(adapter, name, data.pop("latent"))
    else:
        data = direct_latent(adapter, name, source, case["assignment"]["payload"])
        save_rgb(native_png, quantize(data["raw"], "round_even"))
        blind = None  # A 256px SepWater control must not call the fixed-224 API.
    raw = data["raw"]
    if raw.shape != source.shape or not np.isfinite(raw).all():
        raise EvidenceError("Invalid encoder output geometry or range")
    np.save(attempt / "raw.npy", raw, allow_pickle=False)
    round_pixels = quantize(raw, "round_even")
    rounded = attempt / "round_even.png"
    save_rgb(rounded, round_pixels)
    if not np.array_equal(load_rgb(rounded), round_pixels):
        raise EvidenceError("PNG round-trip changed integer pixels")

    def read(image, *, quantized=True):
        seed_everything(case["seed"])
        return scored(decode(adapter, name, image, quantized), case)

    native_pixels = load_rgb(native_png)
    measurements = {
        "raw": read(raw), "clipped": read(np.clip(raw, 0, 1)),
        "round_memory": read(round_pixels.astype(np.float32) / 255),
        "round_png": read(load_rgb(rounded).astype(np.float32) / 255),
        "native_png_direct": read(native_pixels.astype(np.float32) / 255),
        "clean_input_control": read(source.astype(np.float32) / 255),
    }
    if name == "editguard":
        measurements["raw_without_decoder_quantization"] = read(raw, quantized=False)
    checks = {"png_pixels_exact": True,
              "memory_png_bits_equal": measurements["round_memory"]["decoded_bits"] == measurements["round_png"]["decoded_bits"]}
    if api_applicable:
        seed_everything(case["seed"])
        measurements["native_png_api"] = scored(adapter.decode(str(native_png), str(blind)), case)
        checks["api_direct_bits_equal"] = measurements["native_png_api"]["decoded_bits"] == measurements["native_png_direct"]["decoded_bits"]
    else:
        checks["api_direct_bits_equal"] = None
    if profile == "historical":
        checks["historical_pixels_equal"] = np.array_equal(native_pixels, load_rgb(folder / "historical.png"))
        checks["historical_bits_equal"] = measurements["native_png_api"]["decoded_bits"] == read_json(folder / "historical_decode.json")["decoded_bits"]
    quality = {"original_to_marked": psnr(source, native_pixels)}
    if "ae" in data:
        ae = data["ae"]
        if ae.shape != source.shape or not np.isfinite(ae).all():
            raise EvidenceError("Invalid same-latent reconstruction")
        np.save(attempt / "ae.npy", ae, allow_pickle=False)
        ae_pixels = quantize(ae, "round_even")
        save_rgb(attempt / "ae_only.png", ae_pixels)
        quality.update(original_to_ae=psnr(source, ae_pixels), ae_to_marked=psnr(ae_pixels, native_pixels))
        measurements["ae_float_control"] = read(ae)
    return {"status": "measured", "profile": profile, "geometry": list(source.shape[:2]),
            "measurements": measurements, "checks": checks, "psnr_db": quality,
            "raw_stats": array_stats(raw, source), "same_latent_ae": "ae" in data,
            "serializer": "frozen_adapter" if api_applicable else "diagnostic_round_even",
            "api_applicable": api_applicable}


def work(job, profile):
    root, project = Path(job["output"]), Path(job["project"])
    model, name = job["model"], job["model"]["id"]
    meta = root / name / "_meta"
    prepared = read_json(meta / "prepared.json")
    if inference_identity(prepared["identity"]) != read_json(meta / "historical_model_identity.json"):
        raise EvidenceError("Model or weights differ from the historical canary")
    deps = {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")}
    if deps != read_json(meta / "historical_dependency_identity.json"):
        raise EvidenceError("Dependencies differ from the saved environment lock")
    config = read_json(prepared["config"])
    options = config["options"]
    freeze(meta / "source_audit.json", verify_sources(project, options, name))
    execution = {"runtime": runtime_info(), "profile": profile, "session": job["session"]}
    freeze(meta / "paired_runtime.json", execution["runtime"])
    seed_everything(0)
    adapter = adapter_instance(config)
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    if name == "editguard":
        checkpoint = Path(options["weights"])
        if profile == "clean_checkpoint":
            checkpoint = root / "_assets/editguard/clean.pth"
            receipt = read_json(checkpoint.with_name("clean.pth.receipt.json"))
            if receipt["sha256"] != sha256(checkpoint):
                raise EvidenceError("Downloaded clean checkpoint changed")
            audit = load_clean_checkpoint(adapter, checkpoint)
        else:
            audit = editguard_checkpoint(adapter, checkpoint)
        audit.update(checkpoint_sha256=sha256(checkpoint), checkpoint=checkpoint.name)
    else:
        audit = audit_model(adapter, name)
    freeze(meta / f"audit_{profile}.json", audit)
    write_json(meta / "executions" / f"{job['session']}_{profile}.json", execution)
    errors = 0
    for index, case in enumerate(job["cases"], 1):
        if complete(root, case, job["identity"], profile):
            continue
        path = row_path(root, case, profile)
        if path.exists():
            retry_record(path)
        attempt = path.parent / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        started = time.monotonic()
        print(f"{name} / {profile}: {index}/25 {case['stem']}", flush=True)
        try:
            data = measure(adapter, job, case, profile, attempt)
        except EvidenceError:
            raise
        except Exception as error:
            errors += 1
            data = {"status": "error", "reason": str(error), "traceback": traceback.format_exc()}
        sealed(root, path, task_key(job["identity"], case, profile),
               {**data, "execution": execution, "seconds": time.monotonic() - started},
               sorted(p for p in attempt.iterdir() if p.is_file()))
    if errors:
        raise RuntimeError(f"{errors} technical errors; finished cases remain saved")
