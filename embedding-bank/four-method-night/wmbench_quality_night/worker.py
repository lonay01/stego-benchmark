"""One initialized method per process; atomic records after each full message."""

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
from wmbench_native_diagnostics.probes import audit_model, array_stats, quantize, reconstruct_same_latent
from wmbench_author_reference.core import verify_sources
from wmbench_quality_review.probes import capture, decode, direct_latent, editguard_checkpoint, load_clean_checkpoint

from .core import CLEAN_SHA256, carrier, complete, folder_for, row_path, task_key
from .environment import verify_dependencies


def scored(decoded, case):
    metrics = score(decoded, case["assignment"])
    if metrics["status"] != "measured":
        raise EvidenceError(f"Invalid full-length decoder result: {metrics}")
    return metrics


def measure(adapter, job, case, profile, attempt):
    root, name = Path(job["output"]), case["model"]
    folder = folder_for(root, case)
    original = inside(root, case["image"]["path"])
    if sha256(original) != case["image"]["sha256"]:
        raise EvidenceError("Original COCO input changed")
    source = carrier(load_rgb(original), load_rgb(folder / "source.png"), name, profile)
    input_path, native_png = attempt / "input.png", attempt / "native.png"
    metadata = attempt / "metadata.json"
    save_rgb(input_path, source)
    api_applicable = not (name == "rosteals" and profile == "resize256")

    def embed(output, meta, *, make_decoder_metadata=True):
        seed_everything(case["seed"])
        if api_applicable:
            with capture(adapter, name) as data:
                adapter.embed(str(input_path), str(output), case["assignment"]["payload"], str(meta))
            blind = decode_metadata(meta) if make_decoder_metadata else None
            if "latent" in data:
                data["ae"] = reconstruct_same_latent(adapter, name, data.pop("latent"))
        else:
            # Keep the old RoSteALS resize256 core exactly. SepWater always uses
            # its 224px file API; there is no 256px bypass or residual rescaling.
            data = direct_latent(adapter, name, source, case["assignment"]["payload"])
            save_rgb(output, quantize(data["raw"], "round_even"))
            blind = None
        return data, blind

    data, blind = embed(native_png, metadata)
    raw = data["raw"]
    if raw.shape != source.shape or not np.isfinite(raw).all():
        raise EvidenceError("Invalid encoder output geometry or finite range")
    np.save(attempt / "raw.npy", raw, allow_pickle=False)
    reloaded = np.load(attempt / "raw.npy", allow_pickle=False)
    if not np.array_equal(raw, reloaded):
        raise EvidenceError("Float32 serialization changed the captured tensor")
    round_pixels = quantize(raw, "round_even")
    rounded = attempt / "round_even.png"
    save_rgb(rounded, round_pixels)
    if not np.array_equal(load_rgb(rounded), round_pixels):
        raise EvidenceError("PNG changed integer pixels")

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
        "float_reload": read(reloaded),
    }
    checks = {"png_pixels_exact": True,
              "memory_png_bits_equal": measurements["round_memory"]["decoded_bits"] == measurements["round_png"]["decoded_bits"],
              "float_reload_bits_equal": measurements["raw"]["decoded_bits"] == measurements["float_reload"]["decoded_bits"]}
    for serializer in ("floor", "half_up"):
        path = attempt / f"{serializer}.png"
        save_rgb(path, quantize(raw, serializer))
        measurements[f"{serializer}_png"] = read(load_rgb(path).astype(np.float32) / 255)
    if name == "editguard":
        measurements["raw_without_decoder_quantization"] = read(raw, quantized=False)
    if api_applicable:
        seed_everything(case["seed"])
        measurements["native_png_api"] = scored(adapter.decode(str(native_png), str(blind)), case)
        checks["api_direct_bits_equal"] = measurements["native_png_api"]["decoded_bits"] == measurements["native_png_direct"]["decoded_bits"]
    if profile == "historical" and case["cohort"] == "historical":
        checks["historical_pixels_equal"] = np.array_equal(native_pixels, load_rgb(folder / "historical.png"))
        checks["historical_bits_equal"] = measurements["native_png_api"]["decoded_bits"] == read_json(folder / "historical_decode.json")["decoded_bits"]
    if case["replicate"] == 1:
        again, _ = embed(attempt / "repeat.png", attempt / "repeat_metadata.json", make_decoder_metadata=False)
        checks["repeat_raw_equal"] = np.array_equal(raw, again["raw"])
        checks["repeat_png_equal"] = np.array_equal(native_pixels, load_rgb(attempt / "repeat.png"))
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
    return {"status": "measured", "profile": profile, "cohort": case["cohort"],
            "geometry": list(source.shape[:2]), "measurements": measurements, "checks": checks,
            "psnr_db": quality, "raw_stats": array_stats(raw, source),
            "same_latent_ae": "ae" in data, "api_applicable": api_applicable,
            "serializer": "frozen_adapter" if api_applicable else "diagnostic_round_even",
            "extra_controls": "night_v2"}


def work(job, profile):
    root, project = Path(job["output"]), Path(job["project"])
    name, meta = job["model"]["id"], Path(job["output"]) / job["model"]["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    if inference_identity(prepared["identity"]) != read_json(meta / "historical_model_identity.json"):
        raise EvidenceError("Model/checkpoint differs from historical canary")
    verify_dependencies(meta, prepared)
    config = read_json(prepared["config"])
    freeze(meta / "source_audit.json", verify_sources(project, config["options"], name))
    execution = {"runtime": runtime_info(), "profile": profile, "session": job["session"]}
    freeze(meta / "paired_runtime.json", execution["runtime"])
    seed_everything(0)
    adapter = adapter_instance(config)
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    if name == "editguard":
        checkpoint = Path(config["options"]["weights"])
        if profile == "clean_checkpoint":
            checkpoint = root / "_assets/editguard/clean.pth"
            receipt = read_json(checkpoint.with_name("clean.pth.receipt.json"))
            if receipt["sha256"] != CLEAN_SHA256 or sha256(checkpoint) != CLEAN_SHA256:
                raise EvidenceError("Clean checkpoint changed")
            audit = load_clean_checkpoint(adapter, checkpoint)
        else:
            audit = editguard_checkpoint(adapter, checkpoint)
        audit.update(checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint))
    else:
        audit = audit_model(adapter, name)
    if name == "sepwater":
        from omegaconf import OmegaConf
        options = config["options"]
        configuration = Path(options.get("config", Path(options["repo_root"]) / "pre_trained_models/inference.yaml"))
        params = OmegaConf.load(configuration).model.params
        geometry = {"input_resolution": int(params.first_stage_config.params.ddconfig.resolution),
                    "control_resolution": int(params.control_config.params.resolution),
                    "control_base_resolution": int(params.control_config.params.base_res)}
        if geometry != {"input_resolution": 224, "control_resolution": 56, "control_base_resolution": 28}:
            raise EvidenceError(f"Unreviewed SepWater checkpoint geometry: {geometry}")
        audit["geometry"] = geometry
    freeze(meta / f"audit_{profile}.json", audit)
    write_json(meta / "executions" / f"{job['session']}_{profile}.json", execution)
    errors, repeated = 0, []
    for index, case in enumerate(job["cases"], 1):
        if complete(root, case, profile, job["identity"]):
            continue
        path = row_path(root, case, profile)
        if path.exists():
            retry_record(path)
        attempt = path.parent / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        started = time.monotonic()
        print(f"WM_DIAG {name}/{profile}: {index}/{len(job['cases'])} {case['stem']} ({case['cohort']})", flush=True)
        try:
            data = measure(adapter, job, case, profile, attempt)
            repeated.clear()
        except EvidenceError:
            raise
        except Exception as error:
            errors += 1
            reason = f"{type(error).__name__}: {error}"
            repeated.append(reason)
            data = {"status": "error", "reason": reason, "traceback": traceback.format_exc()}
            print(data["traceback"], flush=True)
        row = sealed(root, path, task_key(job["identity"], case, profile),
                     {**data, "execution": execution, "seconds": time.monotonic() - started},
                     sorted(p for p in attempt.iterdir() if p.is_file()))
        write_json(meta / "progress.json", {"profile": profile, "case": case["stem"],
                   "index": index, "total": len(job["cases"]), "status": row["status"],
                   "seconds": row["seconds"], "updated_unix": time.time()})
        if row["status"] == "measured":
            main = row["measurements"]["native_png_direct"]
            print(f"WM_DIAG {name}/{profile}: saved {index}/{len(job['cases'])}; BER={100 * main['ber']:.4f}%, exact={main['exact']}, {row['seconds']:.1f}s", flush=True)
        if len(repeated) >= 3 and len(set(repeated[-3:])) == 1:
            raise RuntimeError("Same technical error in three consecutive cases; stopped this profile, not other profiles")
    if errors:
        raise RuntimeError(f"{errors} technical errors; successful cases remain saved")
