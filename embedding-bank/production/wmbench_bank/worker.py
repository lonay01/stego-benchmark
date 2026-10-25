"""Transactional carrier creation followed by blind clean decoding per method."""

import time
import traceback
from pathlib import Path

from wmbench.embedding_bank import inference_identity
from wmbench.io import digest, read_json, write_json
from wmbench.matching import load_rgb, match_psnr, psnr, save_rgb
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside, retry_record, sealed
from wmbench_full_payload.worker import decode_metadata, decode_saved, record_path, task_id
from wmbench_native_canary.worker import check_png, freeze_crop

from .contracts import MODES, POLICY, message


def signature(model, image, replicate):
    return digest({"model": model, "source": image["sha256"], "policy": POLICY,
                   "message": message(model, image["image_id"], replicate)})


def reusable(root, path, key):
    row = cached(root, path, key)
    if row and row["status"] == "error":
        retry_record(path)
        return None
    return row


def encode_pair(adapter, model, root, image, replicate, execution):
    """Never rerun an encoder to create a matched target or to retry a decoder."""
    source = freeze_crop(root, model, image)
    stem, key = task_id(image, replicate), signature(model, image, replicate)
    assignment = message(model, image["image_id"], replicate)
    path = record_path(root, model, stem, "native")
    native = reusable(root, path, key)
    if native is None:
        start = time.monotonic()
        pending = root / model["id"] / "_meta/attempts" / f"{stem}_{time.time_ns()}"
        pending.mkdir(parents=True)
        png, metadata = pending / "native.png", pending / "metadata.json"
        try:
            seed_everything(int(key[:8], 16))
            adapter.embed(str(source), str(png), assignment["payload"], str(metadata))
            check_png(png, model["native_geometry"]["width"])
            decoder = decode_metadata(metadata)
            target = root / model["id"] / "native" / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            png.replace(target)
            files = [source, target, *sorted(p for p in pending.rglob("*") if p.is_file())]
            controls = {}
            if hasattr(adapter, "last_reconstruction"):
                ae = pending / "reconstruction.png"
                save_rgb(ae, adapter.last_reconstruction)
                files.append(ae)
                controls = {"reconstruction": ae.relative_to(root).as_posix(),
                    "source_to_reconstruction_psnr": psnr(load_rgb(source), load_rgb(ae)),
                    "reconstruction_to_marked_psnr": psnr(load_rgb(ae), load_rgb(target))}
            native = sealed(root, path, key, {"status": "saved", "mode": "native",
                "image_id": image["image_id"], "replicate": replicate, "assignment": assignment,
                "output": target.relative_to(root).as_posix(), "source_crop": source.relative_to(root).as_posix(),
                "decoder_metadata": decoder.relative_to(root).as_posix(), "controls": controls,
                "psnr": psnr(load_rgb(source), load_rgb(target)), "seconds": time.monotonic()-start,
                "execution": execution}, files)
        except Exception as error:
            sealed(root, path, key, {"status": "error", "reason": repr(error),
                "traceback": traceback.format_exc(), "execution": execution})
            raise
    x, y = load_rgb(source), load_rgb(inside(root, native["output"]))
    for mode in MODES[1:]:
        path = record_path(root, model, stem, mode)
        if reusable(root, path, key):
            continue
        start = time.monotonic()
        value, match = match_psnr(x, y, int(mode.split("_")[1]), POLICY["tolerance"],
                                  POLICY["max_alpha"], POLICY["iterations"])
        files = [source, inside(root, native["output"])]
        result = {"status": "unmatched", "mode": mode, "matching": match,
                  "image_id": image["image_id"], "replicate": replicate, "execution": execution}
        if match["status"] == "ok":
            target = root / model["id"] / mode / f"{stem}.png"
            save_rgb(target, value)
            actual = psnr(x, load_rgb(target))
            if abs(actual-match["target"]) > POLICY["tolerance"]:
                raise EvidenceError("Saved PNG is outside the matched PSNR tolerance")
            result.update(status="saved", output=target.relative_to(root).as_posix(), psnr=actual)
            files.append(target)
        result["seconds"] = time.monotonic()-start
        sealed(root, path, key, result, files)
    return native


def decode_pair(adapter, model, root, image, replicate, execution):
    stem, key = task_id(image, replicate), signature(model, image, replicate)
    native = cached(root, record_path(root, model, stem, "native"), key)
    if not native or native["status"] != "saved":
        return 0
    failures = 0
    for mode in MODES:
        row = cached(root, record_path(root, model, stem, mode), key)
        if not row or row["status"] != "saved":
            continue
        result = decode_saved(adapter, root, model, image, replicate, mode, native["assignment"], key,
            inside(root, row["output"]), inside(root, native["decoder_metadata"]), execution)
        failures += result["status"] == "error"
        print(f"CLEAN {model['id']} {stem} {mode}: {result['status']} BER={result.get('ber')}", flush=True)
    return failures


def complete(root, model, images):
    for image in images:
        for replicate in range(1, 6):
            stem, key = task_id(image, replicate), signature(model, image, replicate)
            for mode in MODES:
                row = cached(root, record_path(root, model, stem, mode), key)
                if not row or row["status"] not in ("saved", "unmatched"):
                    return False
                if mode == "native" and row["status"] != "saved":
                    return False
                if row["status"] == "saved":
                    dec = cached(root, record_path(root, model, stem, mode+"_decode"), key)
                    if not dec or dec["status"] not in ("measured", "rejected"):
                        return False
    return True


def work(job):
    root, model = Path(job["output"]), job["model"]
    meta = root / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    seed_everything(0)
    if model["engine"] == "author":
        from .adapters import AuthorAdapter
        adapter = AuthorAdapter(model["author_name"], prepared["source"], root / "_author")
    else:
        freeze(meta / "model_identity.json", inference_identity(prepared["identity"]))
        freeze(meta / "dependency_identity.json", {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")})
        adapter = adapter_instance(read_json(prepared["config"]))
        if hasattr(adapter, "initialize"):
            adapter.initialize()
        from wmbench_native_canary.adapters import validate_geometry
        freeze(meta / "geometry.json", validate_geometry(adapter, model))
    if hasattr(adapter, "native_bits") and adapter.native_bits != model["native_bits"]:
        raise EvidenceError("Loaded payload length differs from the frozen contract")
    execution = {"session": job["session"], "runtime": runtime_info()}
    write_json(meta / "executions" / f"{job['session']}.json", execution)
    # Deliberately finish carrier creation first, then clean-decode saved PNGs.
    # This boundary is per method, so 29 model reloads are avoided.
    errors = 0
    for stage in ("embed", "decode"):
        consecutive = 0
        for image in job["images"]:
            for replicate in range(1, 6):
                if time.time() >= job["deadline"]:
                    print("SESSION BUDGET: current durable records can be resumed", flush=True)
                    return
                try:
                    if stage == "embed":
                        encode_pair(adapter, model, root, image, replicate, execution)
                        print(f"SAVED {model['id']} {task_id(image, replicate)}: four mode records", flush=True)
                        failed = 0
                    else:
                        failed = decode_pair(adapter, model, root, image, replicate, execution)
                    errors += failed
                    consecutive = consecutive + 1 if failed else 0
                except EvidenceError:
                    raise
                except Exception as error:
                    errors += 1
                    consecutive += 1
                    print(f"ERROR {stage} {task_id(image, replicate)}: {error}", flush=True)
                write_json(meta / "progress.json", {"stage": stage, "image_id": image["image_id"],
                    "replicate": replicate, "errors": errors, "updated_unix": time.time()})
                if consecutive >= 3:
                    raise RuntimeError("Three consecutive technical failures; stopping this method, not retrying BER")
    if errors:
        raise RuntimeError(f"{errors} technical failures; successful artifacts retained")
