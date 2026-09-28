"""Small paired checks. Each phase runs in a fresh model-specific process."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from wmbench.embedding_bank import inference_identity
from wmbench.io import read_json, write_json
from wmbench.matching import load_rgb, psnr, save_rgb
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench_full_payload.storage import EvidenceError, freeze, inside, retry_record, sealed
from wmbench_full_payload.worker import decode_metadata
from wmbench_native_canary.contracts import score
from wmbench_native_canary.worker import check_png
from wmbench_native_diagnostics.probes import capture_forward, decode_float, pixels, quantize, rgb_tensor

from .core import (REFERENCE_METHODS, SERIALIZER_METHODS, compare_arrays, complete,
                   folder_for, get_row, key_for, row_path, source_root, verify_sources)
from .upstream import Reference, serialize_author, tensor_digest


def measured(decoded, case):
    metrics = score(decoded, case["assignment"])
    if metrics["status"] != "measured":
        raise EvidenceError(f"Invalid decoder contract: {metrics}")
    return {**metrics, "logits": decoded.get("logits", decoded.get("details", {}).get("logits"))}


def read_array(root, record, field):
    return np.load(inside(root, record[field]), allow_pickle=False)


def compare_decoders(left, right):
    return {"same_bits": left["decoded_bits"] == right["decoded_bits"],
            "logits": compare_arrays(left["logits"], right["logits"])}


def fresh_attempt(folder, phase):
    attempt = folder / "attempts" / f"{phase}_{time.time_ns()}"
    attempt.mkdir(parents=True)
    return attempt


def adapter_case(adapter, model, root, case, attempt):
    import torch
    folder, name = folder_for(root, case), model["id"]
    png, meta = attempt / "adapter.png", attempt / "metadata.json"
    signed_capture = {}
    hook = None
    if name == "sepmark":
        def observe(_module, _args, result):
            signed_capture["tensor"] = result.detach().clone()
        hook = adapter.encoder.register_forward_hook(observe)
    try:
        seed_everything(case["seed"])
        with capture_forward(adapter, name) as observations:
            adapter.embed(str(folder / "source.png"), str(png), case["assignment"]["payload"], str(meta))
    finally:
        if hook is not None:
            hook.remove()
    check_png(png, model["native_geometry"]["width"])
    raw = observations["raw"]
    if raw.shape != load_rgb(folder / "source.png").shape or not np.isfinite(raw).all():
        raise EvidenceError("Invalid raw adapter image")
    array_path = attempt / "adapter_raw.npy"
    np.save(array_path, raw, allow_pickle=False)
    blind = decode_metadata(meta)
    seed_everything(case["seed"])
    raw_score = measured(decode_float(adapter, name, raw), case)
    seed_everything(case["seed"])
    png_score = measured(adapter.decode(str(png), str(blind)), case)
    png_direct = measured(decode_float(adapter, name, load_rgb(png).astype(np.float32) / 255), case)
    result = {"raw": array_path.relative_to(root).as_posix(), "png": png.relative_to(root).as_posix(),
              "blind_metadata": blind.relative_to(root).as_posix(), "raw_score": raw_score,
              "png_score": png_score, "png_direct": png_direct,
              "file_api_matches_direct": png_score["decoded_bits"] == png_direct["decoded_bits"],
              "psnr_original_to_png": psnr(load_rgb(folder / "source.png"), load_rgb(png))}
    if case["cohort"] == "historical":
        result["history"] = {
            "pixels": compare_arrays(load_rgb(png), load_rgb(folder / "historical.png")),
            "same_bits": png_score["decoded_bits"] == read_json(folder / "historical_decode.json")["decoded_bits"]}
    if name in SERIALIZER_METHODS:
        author = attempt / "upstream_export.png"
        with torch.no_grad():
            serialize_author(name, source_root(adapter.options, name), rgb_tensor(torch, raw, adapter.device),
                             author, signed=signed_capture.get("tensor"))
        check_png(author, model["native_geometry"]["width"])
        upstream_pixels = load_rgb(author)
        expected_rule = "half_up" if name == "sepmark" else "floor"
        # Same-RGB export separates quantizer effects from residual arithmetic.
        same_input = attempt / "same_rgb_author_rule.png"
        save_rgb(same_input, quantize(raw, expected_rule))
        result.update(author_png=author.relative_to(root).as_posix(), author_rule=expected_rule,
            author_score=measured(adapter.decode(str(author), str(blind)), case),
            same_rgb_author_score=measured(adapter.decode(str(same_input), str(blind)), case),
            author_vs_same_rgb=compare_arrays(upstream_pixels, load_rgb(same_input)),
            author_vs_adapter_pixels=compare_arrays(upstream_pixels, load_rgb(png)))
    return result


def reference_case(reference, root, case, attempt, identity):
    from .upstream import serialize_author
    folder = folder_for(root, case)
    base = get_row(root, case, "adapter", identity)
    if not base or base["status"] != "measured":
        raise EvidenceError("Reference requires a completed sealed adapter capture")
    source = load_rgb(folder / "source.png")
    seed_everything(case["seed"])
    signed, cover = reference.embed(source, case["assignment"]["payload"])
    raw = pixels((signed + 1) / 2)
    path = attempt / "reference_raw.npy"
    np.save(path, raw, allow_pickle=False)
    png = attempt / "reference.png"
    serialize_author(case["model"], reference.repo, (signed + 1) / 2, png,
                     signed=signed, cover=cover, source=source)
    check_png(png, source.shape[0])
    own_raw = measured(reference.decode(raw), case)
    own_png = measured(reference.decode(load_rgb(png).astype(np.float32) / 255), case)
    adapter_raw = measured(reference.decode(read_array(root, base, "raw")), case)
    adapter_png = measured(reference.decode(load_rgb(inside(root, base["png"])).astype(np.float32) / 255), case)
    return {"raw": path.relative_to(root).as_posix(), "png": png.relative_to(root).as_posix(),
            "raw_score": own_raw, "png_score": own_png,
            "on_adapter_raw": adapter_raw, "on_adapter_png": adapter_png,
            "encoder_float": compare_arrays(raw, read_array(root, base, "raw")),
            "decoder_on_adapter_raw": compare_decoders(adapter_raw, base["raw_score"]),
            "decoder_on_adapter_png": compare_decoders(adapter_png, base["png_direct"])}


def cross_case(adapter, root, case, identity):
    base = get_row(root, case, "adapter", identity)
    ref = get_row(root, case, "reference", identity)
    if not ref or ref["status"] != "measured":
        raise EvidenceError("Cross-decode requires completed reference output")
    raw = measured(decode_float(adapter, case["model"], read_array(root, ref, "raw")), case)
    png = measured(adapter.decode(str(inside(root, ref["png"])), str(inside(root, base["blind_metadata"]))), case)
    direct = measured(decode_float(adapter, case["model"], load_rgb(inside(root, ref["png"])).astype(np.float32) / 255), case)
    return {"on_reference_raw": raw, "on_reference_png": png,
            "decoder_on_reference_raw": compare_decoders(raw, ref["raw_score"]),
            "decoder_on_reference_png": compare_decoders(direct, ref["png_score"]),
            "file_api_matches_direct": png["decoded_bits"] == direct["decoded_bits"]}


def work(job, phase):
    root, project = Path(job["output"]), Path(job["project"])
    model, name = job["model"], job["model"]["id"]
    meta = root / name / "_meta"
    prepared = read_json(meta / "prepared.json")
    current = inference_identity(prepared["identity"])
    if current != read_json(meta / "historical_model_identity.json"):
        raise EvidenceError("Model/weight identity differs from historical canary")
    deps = {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")}
    if deps != read_json(meta / "historical_dependency_identity.json"):
        raise EvidenceError("Dependencies differ from historical locked environment")
    options = read_json(prepared["config"])["options"]
    proof = verify_sources(project, options, name)
    freeze(meta / "source_audit.json", proof)
    execution = {"runtime": runtime_info(), "session": job["session"], "phase": phase,
                 "fingerprint": prepared["fingerprint"]}
    # Pin the runtime within this comparison; a changed device/software cannot
    # silently become the other half of the paired experiment on resume.
    freeze(meta / "paired_runtime.json", execution["runtime"])
    seed_everything(0)
    if phase == "reference":
        instance = Reference(name, options)
        freeze(meta / "reference_weights.json", instance.weights)
        write_json(meta / "reference_contract.json", instance.details)
        freeze(meta / "weight_agreement.json", {
            "identical_inference_tensors": instance.weights == read_json(meta / "adapter_weights.json"),
            "checked_keys": len(instance.weights)})
    else:
        instance = adapter_instance(read_json(prepared["config"]))
        if hasattr(instance, "initialize"):
            instance.initialize()
        if name in REFERENCE_METHODS:
            network = instance.wrapper.encoder_decoder if name == "hidden_benchmark" else (
                instance.wrapper.model if name == "rosteals" else instance._load())
            freeze(meta / "adapter_weights.json", tensor_digest(network, () if name == "hidden_benchmark" else
                   ("ae.", "control.", "decoder.")))
    write_json(meta / "executions" / f"{job['session']}_{phase}.json", execution)
    errors = 0
    for index, case in enumerate(job["cases"], 1):
        if complete(root, case, job["identity"], phase):
            continue
        path = row_path(root, case, phase)
        if path.exists():
            retry_record(path)
        folder = folder_for(root, case)
        attempt = fresh_attempt(folder, phase)
        started = time.monotonic()
        print(f"{name} {phase} {index}/{len(job['cases'])} {case['cohort']} {case['stem']}", flush=True)
        try:
            if phase == "adapter":
                data = adapter_case(instance, model, root, case, attempt)
            elif phase == "reference":
                data = reference_case(instance, root, case, attempt, job["identity"])
            else:
                data = cross_case(instance, root, case, job["identity"])
            data["status"] = "measured"
        except EvidenceError:
            raise
        except Exception as error:
            errors += 1
            data = {"status": "error", "reason": str(error), "traceback": traceback.format_exc()}
            print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        sealed(root, path, key_for(job["identity"], case, phase),
               {**data, "execution": execution, "seconds": time.monotonic() - started},
               sorted(p for p in attempt.iterdir() if p.is_file()))
    if errors:
        raise RuntimeError(f"{errors} technical failures in {phase}; completed cases are retained")
