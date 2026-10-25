"""IO parity against the same pinned upstream weights, without payload access."""

import inspect
from pathlib import Path
import time

import numpy as np

from .common import (ATTACK_IDS, CONTROLS, commit, compare_history, digest, inside,
                     pixel_stats, read, rgb, same_clean, sha, verify_artifacts)
from .pipeline import paths


def direct(adapter, pixels):
    """Reference route: construct RGB CHW independently of adapter._image()."""
    torch = adapter.torch
    array = np.asarray(pixels)
    if array.shape != (512, 512, 3) or not np.isfinite(array).all():
        raise ValueError("Bad direct-decoder image")
    if array.dtype == np.uint8:
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255
    elif array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1:
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    else:
        raise ValueError("Direct route accepts RGB8 or float32 RGB[0,1], not an inferred range")
    with torch.no_grad():
        if adapter.model_id == "invismark":
            normalized = (tensor * 2 - 1).unsqueeze(0).to(adapter.device)
            raw = adapter.wrapper.invismark.model._decode(normalized).detach().reshape(-1)
            bits = torch.round(raw).int().cpu().tolist()
            threshold = 0.5
            if raw.numel() != 100 or not bool(torch.isfinite(raw).all()) or raw.min() < 0 or raw.max() > 1:
                raise ValueError("InvisMark must return 100 finite probabilities")
        elif adapter.model_id == "chunkyseal":
            output = adapter.seal.detect(tensor.unsqueeze(0).to(adapter.device), is_video=False)
            pred = output["preds"]
            if pred.shape[0] != 1 or pred.shape[1] != 1025 or pred.numel() != 1025:
                raise ValueError(f"Unexpected ChunkySeal output shape: {tuple(pred.shape)}")
            raw = pred[0, 1:].detach().reshape(-1)
            bits = (raw > 0).int().cpu().tolist()
            threshold = 0.0
        else:
            raise ValueError(adapter.model_id)
    values = raw.float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite decoder output")
    return dict(bits=bits, raw=values.tolist(), threshold=threshold,
                raw_stats=pixel_stats(values),
                mean_margin=float(np.abs(values - threshold).mean()),
                fraction_within_001=float((np.abs(values - threshold) < .01).mean()))


def traced(adapter, fn):
    """Observe the actual tensor entering the pinned extractor, not a guessed input."""
    module = adapter.wrapper.invismark.model.decoder if adapter.model_id == "invismark" else adapter.seal.detector
    if module.training:
        raise RuntimeError("The actual extractor is in training mode")
    observations = []

    def hook(_module, args):
        tensors = [arg for arg in args if adapter.torch.is_tensor(arg)]
        if not tensors:
            raise RuntimeError("Extractor hook did not receive a tensor")
        for tensor in tensors:
            value = pixel_stats(tensor.detach().float().cpu().numpy())
            value.update(torch_dtype=str(tensor.dtype), device=str(tensor.device))
            observations.append(value)

    handle = module.register_forward_pre_hook(hook)
    try:
        result = fn()
    finally:
        handle.remove()
    if not observations:
        raise RuntimeError("No extractor input was observed")
    return result, observations


def source_audit(adapter, root):
    """Archive the actual runtime methods as provenance, without changing them."""
    candidates = {"adapter_decode": adapter.decode, "adapter_image": adapter._image}
    if adapter.model_id == "invismark":
        candidates.update(upstream_decode=adapter.wrapper.invismark.model._decode,
                          wrapper_extract=adapter.wrapper.invismark.extract)
    else:
        candidates.update(upstream_detect=adapter.seal.detect)
    sources = {key: dict(file=inspect.getsourcefile(fn), source=inspect.getsource(fn))
               for key, fn in candidates.items()}
    # Paths can change on a new Colab VM; source content is the scientific identity.
    identity = {key: digest(value["source"]) for key, value in sources.items()}
    commit(root / "source-identity.json", identity)
    commit(root / "source-audits" / (digest(sources) + ".json"), sources)


def decode(job):
    from wmbench_digital.worker import load_adapter, measured
    from wmbench_full_payload.contracts import score
    from wmbench.worker import seed_everything, runtime_info

    root, method = Path(job["results"]), job["model"]["id"]
    base = root / method
    conditions = list(CONTROLS) if job["engine"] == "controls" else [
        a for a in ATTACK_IDS if ("vae" if a.startswith("vae") else "sdxl" if a.startswith("sdxl") else "ip2p") == job["engine"]]
    cases = read(base / "native-cases.json")
    adapter = load_adapter(job)  # Strict existing source, dependency and checkpoint checks.
    source_audit(adapter, base)
    execution = runtime_info()
    commit(base / "decoder-execution.json", execution)
    for case in cases:
        if time.time() >= job["deadline"]:
            raise TimeoutError("Decoder budget reached; completed rows retained")
        metadata = inside(Path(job["bank"]), case["decoder_metadata"])
        blind = read(metadata)
        for field in ("expected_bits", "payload", "full_payload"):
            if isinstance(blind.get(field), list) and any(blind[field]):
                raise RuntimeError("Expected payload must not enter either decoder route")
        seed = int(case["signature"][:8], 16)
        fresh = measured(adapter, inside(Path(job["bank"]), case["output"]), metadata, case, seed)
        if not same_clean(fresh, case["clean"]):
            commit(base / "clean-drift" / (case["id"] + ".json"), dict(bank=case["clean"], fresh=fresh))
            raise RuntimeError("Fresh Clean differs from full saved bit vector; stop before attacks")
        history = read(base / "historical" / (case["id"] + ".json"))
        for condition in conditions:
            result_path = base / "checks" / condition / (case["id"] + ".json")
            _, generated_path = paths(root, method, case, condition)
            generated = read(generated_path)
            verify_artifacts(root, generated)
            if result_path.exists():
                old = read(result_path)
                if old["case_key"] != case["key"] or old["generation_digest"] != digest(generated):
                    raise RuntimeError("Completed decoder row has different inputs")
                verify_artifacts(root, old)
                if old.get("io_agrees") is False or old.get("repeat_equal") is False or old.get("history", {}).get("bits_equal") is False:
                    raise RuntimeError("A saved IO/replay disagreement requires review, not an automatic rerun")
                continue
            common = dict(case_key=case["key"], case_id=case["id"], method=method,
                          image_id=case["image_id"], replicate=case["replicate"], condition=condition,
                          generation_digest=digest(generated), artifacts=generated.get("artifacts", {}))
            if generated["status"] == "blocked_generation":
                commit(result_path, dict(**common, status="blocked_generation", reason=generated["reason"]))
                continue
            path = inside(root, generated["path"])
            pixels = rgb(path)
            actual_pixels = adapter._image(str(path)).detach().cpu().numpy()
            expected_pixels = pixels.astype(np.float32).transpose(2, 0, 1) / 255
            if not np.array_equal(actual_pixels, expected_pixels):
                raise RuntimeError("Adapter input is not the declared RGB CHW float32[0,1]")
            seed_everything(seed)
            official, trace_a = traced(adapter, lambda: measured(adapter, path, metadata, case, seed))
            seed_everything(seed)
            reference, trace_b = traced(adapter, lambda: direct(adapter, pixels))
            reference_score = score(reference, case["assignment"])
            reference_score.pop("expected_bits", None)
            agrees = official["status"] == "measured" and official["decoded_bits"] == reference["bits"] and trace_a == trace_b
            repeated = measured(adapter, path, metadata, case, seed)
            repeat_equal = same_clean(official, repeated)
            historical = history["rows"].get(condition)
            if condition == "clean":
                historical = {**case["clean"], "image_sha256": case["carrier_sha256"]}
                # Lossless re-encoding may have different bytes: compare historical decoded bits explicitly.
                historical_check = dict(status="clean_pixel_replay", bits_equal=same_clean(official, case["clean"]))
            else:
                historical_check = compare_history(official.get("decoded_bits"), generated["png_sha256"], historical)
            value = dict(**common, status=official["status"], official=official, direct=reference_score,
                         raw=reference, adapter_trace=trace_a, direct_trace=trace_b,
                         io_agrees=agrees, repeat_equal=repeat_equal, history=historical_check,
                         quality=generated.get("quality"), pixel_stats=pixel_stats(pixels))
            if condition == "vae_1":
                float_path = inside(root, generated["float_path"])
                if sha(float_path) != generated["float_sha256"]:
                    raise RuntimeError("VAE float checksum changed")
                floating = np.load(float_path, allow_pickle=False)
                seed_everything(seed)
                float_reference, float_trace = traced(adapter, lambda: direct(adapter, floating))
                float_score = score(float_reference, case["assignment"])
                float_score.pop("expected_bits", None)
                value["float_control"] = dict(score=float_score, raw=float_reference, trace=float_trace,
                    note="Direct float input diagnostic; does not replace official PNG score")
                value["artifacts"][generated["float_path"]] = generated["float_sha256"]
            commit(result_path, value)
            print(f"DECODE {method} {condition} {case['id']} BER={official.get('ber')} "
                  f"IO_equal={agrees} repeat_equal={repeat_equal}", flush=True)
            if not agrees or not repeat_equal or historical_check.get("bits_equal") is False:
                raise RuntimeError("IO/replay disagreement saved; no automatic threshold or preprocessing tuning")
