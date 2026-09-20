"""Portable, resumable Native/Matched image bank. No attacks or calibration jobs.

The controller uses no GPU library. Each model runs in a separate process and
environment. Scientific records use relative paths; execution provenance does
not decide whether an already sealed PNG is reusable on another computer.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import uuid

from PIL import Image

from .io import atomic_bytes, digest, read_json, sha256, write_csv, write_json
from .matching import load_rgb, match_psnr, psnr, save_rgb

MODEL_IDS = (
    "trustmark_q",
    "wam",
    "videoseal",
    "pixelseal",
    "chunkyseal",
    "vine_b",
    "vine_r",
    "rosteals",
    "invismark",
    "rivagan_benchmark",
    "dwt_dct_svd",
    "mbrs",
    "cin",
    "hidden_benchmark",
    "arwgan",
    "ssl_watermarking",
    "maskwm",
    "robust_wide",
    "sepmark",
    "stegastamp",
    "pimog",
    "ssrh",
    "editguard",
    "omniguard",
    "sepwater",
    "fang_screen",
    "camera_shooting_instance",
    "trustmark_b",
    "trustmark_p",
    "fin",
    "fnns_omniseal",
    "distseal_posthoc_dcae",
)
POLICY = {
    "schema": "embedding-bank-v1-one-message",
    "seed": 20260811,
    "targets": [39, 42, 45],
    "tolerance": 0.05,
    "max_alpha": 4.0,
    "iterations": 64,
    "matching": "saved_native_u8_scalar_bisection_v1",
    "message": "image-specific-balanced30-sha256-rank-v1",
}
MODES = ("native", "matched_39", "matched_42", "matched_45")


def common_message(image_id: str, seed: int = 20260811) -> list[int]:
    """Stable without dependence on NumPy's random-generator implementation."""
    key = f"{seed}|{int(image_id)}|common30"
    order = sorted(range(30), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).digest())
    ones = set(order[:15])
    return [int(i in ones) for i in range(30)]


def message_for_model(model: dict, image_id: str, seed: int = 20260811):
    length = int(model["native_bits"])
    if length < 30:
        raise ValueError(f"{model['id']}: no validated common30 payload interface")
    namespace = (
        "trustmark"
        if model["id"] in ("trustmark_q", "trustmark_b", "trustmark_p")
        else model.get("payload_namespace", model["id"])
    )
    prefix = f"{seed}|{namespace}|{int(image_id)}|filler"
    bits = [hashlib.sha256(f"{prefix}|{i}".encode()).digest()[0] & 1 for i in range(length)]
    indices = list(range(length - 30, length))
    bits[-30:] = common_message(image_id, seed)
    for position, value in model.get("reserved_bits", {}).items():
        position = int(position)
        if position < 0 or position >= length or position in indices or value not in (0, 1):
            raise ValueError("Invalid reserved bit or overlap with common30")
        bits[position] = int(value)
    return bits, indices


def _relative(bank: Path, path: Path) -> str:
    return path.resolve().relative_to(bank.resolve()).as_posix()


def _inside(bank: Path, relative: str) -> Path:
    path = (bank / relative).resolve()
    if not path.is_relative_to(bank.resolve()):
        raise ValueError(f"Path escapes bank: {relative}")
    return path


def _freeze(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen configuration differs: {path}. Use a NEW output folder.")
    else:
        write_json(path, value)


def freeze_inputs(coco_root: Path, bank: Path, *, require_historical=True) -> dict:
    """Copy all 100 originals unchanged; n_images selects a prefix later."""
    from .data import import_dataset
    from .protocol import LEGACY_DATASET_HASH

    bank = bank.resolve()
    target = bank / "_inputs"
    frozen = target / "manifest.json"
    if frozen.exists():
        value = read_json(frozen)
        original_manifest = target / "legacy_manifest.csv"
        if not original_manifest.is_file() or sha256(original_manifest) != value.get(
            "manifest_sha256"
        ):
            raise ValueError("Frozen source CSV manifest is missing or changed")
        ids = [int(row["image_id"]) for row in value["images"]]
        content_hash = digest(
            [{"coco_id": int(row["image_id"]), "sha256": row["sha256"]} for row in value["images"]]
        )
        if ids != sorted(set(ids)) or value.get("content_hash") != content_hash:
            raise ValueError("Frozen manifest content/order is inconsistent")
        if require_historical and len(ids) != 100:
            raise ValueError("Frozen historical corpus must contain exactly 100 images")
        for row in value["images"]:
            path = _inside(bank, row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Frozen original missing or corrupt: {path}")
        if require_historical and value.get("legacy_semantic_hash") != LEGACY_DATASET_HASH:
            raise ValueError("This bank does not contain the historical COCO100 corpus")
        return value
    source = coco_root / "test"
    manifest = coco_root / "manifest.csv"
    # Import staging is not a scientific bank and contains no generated images.
    with tempfile.TemporaryDirectory(prefix="wmbench_import_") as temporary:
        data = import_dataset(source, Path(temporary), expected_manifest=manifest)
    if require_historical and (
        len(data["images"]) != 100 or data.get("legacy_semantic_hash") != LEGACY_DATASET_HASH
    ):
        raise ValueError("Need the exact historical 100 RGB PNGs and their manifest.csv")
    rows = []
    for row in data["images"]:
        destination = target / "images" / f"{int(row['image_id']):012d}.png"
        if destination.exists() and sha256(destination) != row["sha256"]:
            raise ValueError(f"Conflicting original: {destination}")
        if not destination.exists():
            atomic_bytes(destination, Path(row["path"]).read_bytes())
        rows.append(
            {
                "image_id": row["image_id"],
                "sha256": row["sha256"],
                "path": _relative(bank, destination),
            }
        )
    value = {
        "images": rows,
        "legacy_semantic_hash": data["legacy_semantic_hash"],
        "content_hash": data["content_hash"],
        "manifest_sha256": sha256(manifest),
    }
    archived_manifest = target / "legacy_manifest.csv"
    if archived_manifest.exists() and sha256(archived_manifest) != value["manifest_sha256"]:
        raise ValueError("Conflicting frozen source manifest")
    if not archived_manifest.exists():
        atomic_bytes(archived_manifest, manifest.read_bytes())
    _freeze(frozen, value)
    return value


def bank_registry(project: Path) -> dict:
    from .protocol import load_registry

    models = load_registry(project)
    additions = project / "registry/bank_additions.json"
    if additions.exists():
        data = read_json(additions)
        for model in data if isinstance(data, list) else data["models"]:
            if model["id"] in models:
                raise ValueError(f"Duplicate model: {model['id']}")
            models[model["id"]] = model
    missing = set(MODEL_IDS) - set(models)
    if missing:
        raise ValueError(f"Incomplete release registry: {sorted(missing)}")
    return {name: models[name] for name in MODEL_IDS}


def freeze_bank_identity(project: Path, bank: Path, identity: dict) -> dict:
    """Permit only explicitly audited code-only hotfixes; retain the original seal.

    Dataset, registry and scientific policy must still match exactly. The audit
    records both source hashes; it never rewrites bank.json or saved PNG records.
    """
    path = bank / "bank.json"
    if path.exists():
        original = read_json(path)
        if original != identity:
            old_code = original.get("source_code_sha256")
            new_code = identity.get("source_code_sha256")
            comparable = {**identity, "source_code_sha256": old_code}
            approved_path = project / "compatibility/embedding_bank_hotfixes.json"
            approved = read_json(approved_path) if approved_path.is_file() else {}
            entry = next(
                (
                    row
                    for row in approved.get("transitions", [])
                    if row["from_source_sha256"] == old_code and row["to_source_sha256"] == new_code
                ),
                None,
            )
            if original != comparable or entry is None:
                raise ValueError(
                    f"Frozen configuration differs: {path}. "
                    "No audited compatible update; preserve this bank and use a NEW output folder."
                )
            receipt = bank / "_reports/source_compatibility" / f"{new_code}.json"
            # Adapter-loader fixes may be safe only before affected models have
            # produced outputs. Check once, then bind the migration receipt to
            # this bank so the next resume can reuse NEW outputs from the fix.
            if not receipt.exists():
                for name in entry.get("requires_empty_models", []):
                    if name not in original["models"] or Path(name).name != name:
                        raise ValueError(f"Invalid model in audited source transition: {name}")
                    model_dir = bank / name
                    if (
                        (model_dir / "_meta/model_identity.json").exists()
                        or any((model_dir / "_meta/records").glob("*.json"))
                        or any(model_dir.rglob("*.png"))
                    ):
                        raise ValueError(
                            f"Audited loader update requires no previous outputs for {name}. "
                            "Nothing was deleted; use a NEW bank or audit these outputs first."
                        )
            _freeze(receipt, {"original_bank_identity": digest(original), **entry})
            print(
                "RESUME: audited source hotfix; existing bank and PNG seals preserved.",
                flush=True,
            )
        return original
    _freeze(path, identity)
    return identity


def prepare_bank(project: Path, settings: dict) -> dict:
    bank = Path(settings["output_root"]).expanduser().resolve()
    scratch = Path(settings["scratch_root"]).expanduser().resolve()
    if scratch == bank or scratch.is_relative_to(bank) or bank.is_relative_to(scratch):
        raise ValueError("Persistent bank and disposable scratch must be separate directories")
    models = bank_registry(project)
    selected = settings.get("models", "all")
    selected = list(models) if selected == "all" else list(selected)
    if not selected or len(set(selected)) != len(selected) or set(selected) - models.keys():
        raise ValueError("Select unique IDs from the 32-model registry")
    n = settings.get("n_images", 100)
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("n_images must be an integer between 1 and 100")
    data = freeze_inputs(Path(settings["coco_root"]), bank)
    identity = {
        "policy": POLICY,
        "dataset": data["content_hash"],
        "models": models,
        "source_code_sha256": digest(
            {
                p.relative_to(project).as_posix(): sha256(p)
                for p in sorted((project / "src").rglob("*.py"))
            }
        ),
    }
    identity = freeze_bank_identity(project, bank, identity)
    _freeze(
        bank / "_inputs/messages.json",
        {r["image_id"]: common_message(r["image_id"]) for r in data["images"]},
    )
    for name in models:
        for mode in MODES:
            (bank / name / mode).mkdir(parents=True, exist_ok=True)
    (bank / "_reports").mkdir(exist_ok=True)
    return {
        "bank": bank,
        "scratch": scratch,
        "models": models,
        "selected": selected,
        "images": data["images"][:n],
        "identity": digest(identity),
    }


def valid_record(bank: Path, record_path: Path) -> dict | None:
    """Never silently overwrite a completed but damaged result."""
    if not record_path.exists():
        return None
    record = read_json(record_path)
    if not isinstance(record.get("artifacts"), list) or not record.get("signature"):
        raise ValueError(f"Malformed sealed record: {record_path}")
    if record.get("status") == "ok" and not record["artifacts"]:
        raise ValueError(f"Successful image record has no artifacts: {record_path}")
    for item in record.get("artifacts", []):
        path = _inside(bank, item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Sealed artifact missing/corrupt: {path}; restore its backup")
    return record


def _record_path(bank: Path, model: dict, image_id: str, mode: str) -> Path:
    return bank / model["id"] / "_meta/records" / f"{int(image_id):012d}_{mode}.json"


def _signature(model, image, policy):
    payload, indices = message_for_model(model, image["image_id"], policy["seed"])
    return digest(
        {
            "model": digest(model),
            "image_sha256": image["sha256"],
            "payload": payload,
            "common_indices": indices,
            "policy": policy,
        }
    )


def _seal(bank: Path, path: Path, record: dict, files: list[Path]) -> dict:
    record = {
        **record,
        "artifacts": [{"path": _relative(bank, p), "sha256": sha256(p)} for p in files],
        "finished_unix": time.time(),
    }
    write_json(path, record)
    return record


def _check_png(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"Adapter must produce RGB 512x512 PNG: {path}")
        image.load()


def _decode_check(adapter, png: Path, metadata: Path, payload: list, indices: list) -> dict:
    """A bad decoded message is an observation, never a request to embed again."""
    try:
        result = adapter.decode(str(png), str(metadata))
        bits = result.get("bits")
        if bits is None or len(bits) != len(payload) or any(b not in (0, 1) for b in bits):
            raise ValueError("Decoder output is not a binary native-length message")
        errors = [int(a != b) for a, b in zip(bits, payload)]
        return {
            "status": "measured",
            "decoded_bits": list(map(int, bits)),
            "common30_ber": sum(errors[i] for i in indices) / 30,
            "common30_exact": not any(errors[i] for i in indices),
            "native_ber": sum(errors) / len(errors),
            "score": result.get("score"),
            "details": result.get("details"),
        }
    except Exception as error:
        return {"status": "decode_error", "reason": f"{type(error).__name__}: {error}"}


def process_image(
    adapter,
    model: dict,
    bank: Path,
    image: dict,
    policy: dict,
    provenance: dict,
    identity_check: bool = True,
) -> list[dict]:
    """Commit Native immediately; then match/decode independently per mode."""
    from .worker import seed_everything

    bank = bank.resolve()
    model_root = bank / model["id"]
    image_id = image["image_id"]
    stem = f"{int(image_id):012d}"
    source = _inside(bank, image["path"])
    if sha256(source) != image["sha256"]:
        raise ValueError("Original changed since manifest validation")
    payload, indices = message_for_model(model, image_id, policy["seed"])
    signature = _signature(model, image, policy)
    native = model_root / "native" / f"{stem}.png"
    metadata = model_root / "_meta/keys" / stem / "metadata.json"
    records = []
    modes = ["native", *(f"matched_{t:g}" for t in policy["targets"])]
    for mode in modes:
        record_path = _record_path(bank, model, image_id, mode)
        record = valid_record(bank, record_path)
        if record is not None and record["signature"] != signature:
            raise ValueError(f"Changed message/model/policy for {model['id']}/{stem}/{mode}")
        if record is None:
            started = time.monotonic()
            output = model_root / mode / f"{stem}.png"
            # Files without a seal are unfinished writes. Preserve before replacing.
            if output.exists():
                archived = model_root / "_meta/unsealed" / f"{time.time_ns()}_{mode}_{stem}.png"
                archived.parent.mkdir(parents=True, exist_ok=True)
                output.replace(archived)
            if mode == "native":
                if metadata.parent.exists():
                    archived_keys = model_root / "_meta/unsealed" / f"{time.time_ns()}_{stem}_keys"
                    archived_keys.parent.mkdir(parents=True, exist_ok=True)
                    metadata.parent.replace(archived_keys)
                metadata.parent.mkdir(parents=True, exist_ok=True)
                # Persistent sidecars survive runtime loss and bank relocation.
                seed = int.from_bytes(
                    hashlib.sha256(f"{policy['seed']}|{image_id}|embed".encode()).digest()[:4],
                    "big",
                )
                seed_everything(seed)
                print(f"WM_DIAG BANK embed {model['id']} COCO {image_id}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                adapter.embed(str(source), str(output), payload, str(metadata))
                _check_png(output)
                if not metadata.is_file():
                    raise ValueError("Adapter did not preserve extraction metadata")
                data = {"status": "ok", "psnr": psnr(load_rgb(source), load_rgb(output))}
                files = [output, *sorted(p for p in metadata.parent.rglob("*") if p.is_file())]
            else:
                target = float(mode.split("_")[1])
                array, data = match_psnr(
                    load_rgb(source),
                    load_rgb(native),
                    target=target,
                    tolerance=policy["tolerance"],
                    max_alpha=policy["max_alpha"],
                    iterations=policy["iterations"],
                )
                files = []
                if data["status"] == "ok":
                    save_rgb(output, array)
                    data["psnr"] = psnr(load_rgb(source), load_rgb(output))
                    if abs(data["psnr"] - target) > policy["tolerance"]:
                        raise ValueError("Saved matched PNG does not meet the target tolerance")
                    files = [output, *sorted(p for p in metadata.parent.rglob("*") if p.is_file())]
            record = _seal(
                bank,
                record_path,
                {
                    "signature": signature,
                    "model_id": model["id"],
                    "image_id": image_id,
                    "mode": mode,
                    "payload": payload,
                    "common_indices": indices,
                    "metadata": _relative(bank, metadata),
                    "source": image["path"],
                    "output": _relative(bank, output) if files else None,
                    "execution": provenance,
                    "seconds": time.monotonic() - started,
                    **data,
                },
                files,
            )
        if record["status"] == "ok" and identity_check:
            check_path = record_path.with_name(record_path.stem + "_identity.json")
            checked = valid_record(bank, check_path)
            if checked and checked["signature"] != signature:
                raise ValueError("Identity-check specification differs from saved image")
            if checked and checked.get("status") == "decode_error":
                archived = (
                    model_root / "_meta/decode_errors" / f"{time.time_ns()}_{check_path.name}"
                )
                archived.parent.mkdir(parents=True, exist_ok=True)
                check_path.replace(archived)
                checked = None
            if checked is None:
                seed_everything(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{policy['seed']}|{image_id}|{mode}|decode".encode()
                        ).digest()[:4],
                        "big",
                    )
                )
                checked = _seal(
                    bank,
                    check_path,
                    {
                        "signature": signature,
                        "mode": mode,
                        "model_id": model["id"],
                        "image_id": image_id,
                        "execution": provenance,
                        **_decode_check(
                            adapter, _inside(bank, record["output"]), metadata, payload, indices
                        ),
                    },
                    [_inside(bank, item["path"]) for item in record["artifacts"]],
                )
            record = {**record, "identity_check": checked}
        records.append(record)
        print(f"WM_DIAG BANK saved {model['id']} {stem} {mode}: {record['status']}", flush=True)
    return records


def _complete(bank, model, images, identity_check):
    # Ineligible catalogue entries have no task signature. Let the controller
    # record their prerequisite/integration blocker instead of aborting the queue.
    if int(model["native_bits"]) < 30:
        return False
    for image in images:
        expected = _signature(model, image, POLICY)
        for mode in MODES:
            path = _record_path(bank, model, image["image_id"], mode)
            record = valid_record(bank, path)
            if record is None:
                return False
            if record["signature"] != expected:
                raise ValueError(f"Saved task signature differs: {path}")
            if record["status"] == "ok" and identity_check:
                checked = valid_record(bank, path.with_name(path.stem + "_identity.json"))
                if checked is None or checked.get("status") == "decode_error":
                    return False
                if checked["signature"] != expected:
                    raise ValueError(f"Saved identity-check signature differs: {path}")
    return True


def _log_tail(path: Path, lines: int = 18) -> str:
    """Bound notebook diagnostics without loading an entire model log into RAM."""
    if not path.is_file():
        return "Log file was not created."
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 8192))
        return "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-lines:])


def report(bank: Path, models: dict, images: list, selected: list) -> dict:
    """Missing/blocked methods stay in the denominator; no empty leaderboard."""
    rows, inventory = [], []
    for name in selected:
        model = models[name]
        state_path = bank / name / "_meta/status.json"
        state = read_json(state_path) if state_path.exists() else {}
        count = misses = checks = warnings = 0
        for image in images:
            for mode in MODES:
                path = _record_path(bank, model, image["image_id"], mode)
                rec = valid_record(bank, path)
                if rec is None:
                    continue
                count += rec["status"] == "ok"
                misses += rec["status"] != "ok"
                check_path = path.with_name(path.stem + "_identity.json")
                checked = valid_record(bank, check_path)
                if checked:
                    checks += checked["status"] == "measured"
                    warnings += (
                        checked.get("status") != "measured" or checked.get("common30_ber", 0) > 0
                    )
                inventory.append(
                    {
                        "model": name,
                        "image_id": image["image_id"],
                        "mode": mode,
                        "status": rec["status"],
                        "path": rec.get("output"),
                        "sha256": rec["artifacts"][0]["sha256"] if rec["artifacts"] else None,
                        "psnr": rec.get("psnr"),
                        "alpha": rec.get("alpha"),
                        "common30_ber": checked.get("common30_ber") if checked else None,
                        "identity_status": checked.get("status") if checked else "not_run",
                        "execution_id": rec.get("execution", {}).get("id"),
                    }
                )
        expected = len(images) * len(MODES)
        status = (
            "complete"
            if count == expected
            else "partial"
            if count
            else state.get("status", "pending")
        )
        rows.append(
            {
                "model": name,
                "status": status,
                "saved_png": count,
                "expected_png": expected,
                "target_misses": misses,
                "identity_measured": checks,
                "identity_warnings": warnings,
                "reason": state.get("reason", ""),
            }
        )
    result = {
        "rows": rows,
        "saved_png": sum(r["saved_png"] for r in rows),
        "expected_png": len(selected) * len(images) * 4,
    }
    out = bank / "_reports"
    write_json(out / "report.json", result)
    write_csv(out / "models.csv", rows)
    write_csv(out / "images.csv", inventory)
    text = [
        "# Embedding bank",
        "",
        f"Saved PNGs: {result['saved_png']} / {result['expected_png']}.",
        "",
        "Complete means images exist; it does not mean that every message decodes correctly.",
        "",
        "| Method | State | PNGs | Target misses | Decode warnings |",
        "|---|---|---:|---:|---:|",
    ]
    text += [
        f"| {r['model']} | {r['status']} | {r['saved_png']}/{r['expected_png']} | {r['target_misses']} | {r['identity_warnings']} |"
        for r in rows
    ]
    text += ["", "## Blockers / latest execution notes", ""]
    text += [f"- {r['model']}: {r['reason']}" for r in rows if r["reason"]]
    atomic_bytes(out / "REPORT.md", ("\n".join(text) + "\n").encode())
    return result


def _cleanup(slot: Path, identity: str):
    """Only named regenerable children of this release's owned per-model slot."""
    slot = slot.resolve()
    if read_json(slot / ".embedding-bank-slot.json").get("owner") != identity:
        raise ValueError("Scratch ownership mismatch; refusing cleanup")
    for name in ("environments", "assets", "sources", "cache", "tmp", "locks"):
        path = slot / name
        if path.is_symlink() or not path.resolve().is_relative_to(slot):
            raise ValueError(f"Unsafe cleanup target: {path}")
        if path.exists():
            shutil.rmtree(path)


def run(project: Path, settings: dict) -> Path:
    from .colab_canary import blocked_reason, controller_lease, monitored_process
    from .environment import worker_environment

    project = project.resolve()
    context = prepare_bank(project, settings)
    bank, scratch, models, images = (context[k] for k in ("bank", "scratch", "models", "images"))
    selected = context["selected"]
    manual = Path(settings.get("manual_assets_root", bank / "_manual_assets")).resolve()
    start = time.monotonic()
    max_seconds = float(settings.get("max_hours", 12)) * 3600
    limits = {
        "min_free_disk_gib": settings.get("min_free_disk_gib", 20),
        "min_free_ram_gib": settings.get("min_free_ram_gib", 2),
        "max_slot_gib": settings.get("max_slot_gib", 150),
        "telemetry": True,
    }
    checks = settings.get("identity_check", True)
    with controller_lease(bank / "_reports") as owned:
        report(bank, models, images, selected)
        for index, name in enumerate(selected, 1):
            owned()
            model = models[name]
            state = bank / name / "_meta/status.json"
            if _complete(bank, model, images, checks):
                print(f"[{index}/{len(selected)}] REUSE verified {name}", flush=True)
                continue
            if time.monotonic() - start >= max_seconds:
                print("Session time budget exhausted. Run again to resume.", flush=True)
                break
            reason = blocked_reason(model, manual)
            if reason is None and int(model["native_bits"]) < 30:
                reason = "unsupported_payload: no validated common30 payload interface"
            if reason:
                write_json(state, {"status": "blocked", "reason": reason})
                print(f"[{index}/{len(selected)}] BLOCKED {name}: {reason}", flush=True)
                report(bank, models, images, selected)
                continue
            slot = scratch / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            _freeze(slot / ".embedding-bank-slot.json", {"owner": context["identity"]})
            if shutil.disk_usage(slot).free / 1024**3 < limits["min_free_disk_gib"] + 10:
                raise RuntimeError(
                    "Insufficient scratch space; stop before installing another model"
                )
            session = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
            meta = bank / name / "_meta"
            job = {
                "project": str(project),
                "slot": str(slot),
                "bank": str(bank),
                "model": model,
                "images": images,
                "settings": settings,
                "session": session,
                "manual": str(manual),
            }
            job_path = meta / "jobs" / f"{session}.json"
            write_json(job_path, job)
            env = worker_environment(project, slot, model)
            env.update(
                UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(scratch / "python")
            )
            print(f"[{index}/{len(selected)}] PREPARE {name}", flush=True)
            try:
                for stage in ("prepare", "work"):
                    minutes = (
                        settings.get("prepare_minutes", 60)
                        if stage == "prepare"
                        else settings.get("model_minutes", 180)
                    )
                    remaining = max_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError("Session time budget exhausted")
                    if stage == "prepare":
                        python = sys.executable
                    else:
                        python = read_json(meta / "prepared.json")["python"]
                    command = [
                        sys.executable,
                        "-m",
                        "wmbench.process_guard",
                        "--",
                        str(python),
                        "-m",
                        "wmbench.embedding_bank",
                        stage,
                        "--config",
                        str(job_path),
                    ]
                    log_path = meta / "logs" / f"{session}_{stage}.log"
                    stats = monitored_process(
                        command,
                        env,
                        log_path,
                        slot,
                        min(float(minutes) * 60, remaining),
                        limits,
                        assert_owned=owned,
                    )
                    write_json(meta / "timings" / f"{session}_{stage}.json", stats)
                    if stats["exit_code"] or stats["reason"]:
                        print(
                            f"{stage.upper()} ERROR {name}; log: {log_path}\n{_log_tail(log_path)}",
                            flush=True,
                        )
                        raise RuntimeError(
                            stats["reason"]
                            or f"{stage} exited with code {stats['exit_code']}; see {log_path}"
                        )
                write_json(state, {"status": "processed", "reason": ""})
            except Exception as error:
                write_json(state, {"status": "failed", "reason": str(error)})
                print(f"FAILED {name}: {error}", flush=True)
            finally:
                # Copy small lock/provenance evidence before deleting any disposable assets.
                if (slot / "locks").exists():
                    shutil.copytree(
                        slot / "locks", meta / "environment" / session, dirs_exist_ok=True
                    )
                report(bank, models, images, selected)
                if settings.get("cleanup", True):
                    _cleanup(slot, context["identity"])
    report(bank, models, images, selected)
    return bank


def prepare_child(job: dict):
    from .environment import prepare_model

    project, slot, bank = (Path(job[k]).resolve() for k in ("project", "slot", "bank"))
    model = job["model"]
    for asset in model.get("manual_assets", []):
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manual asset path")
        source = Path(job["manual"]) / model["id"] / relative
        dest = slot / "assets" / model["id"] / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    prepared = prepare_model(project, slot, model, job["settings"].get("device", "cuda"))
    write_json(bank / model["id"] / "_meta/prepared.json", prepared)


def inference_identity(identity: dict) -> dict:
    """Separate narrowly known transport/log outputs from inference inputs.

    The full original inventory is retained in execution provenance. No unknown
    file is ignored. Extracted checkpoint hashes, not ZIP timestamps, are frozen.
    """
    from .maintenance import ARCHIVE_RULES, INVISMARK_CHECKPOINT, INVISMARK_EVENT

    model = identity["registry"]
    weights = {key.replace("\\", "/"): value for key, value in identity["weights"].items()}
    rule = ARCHIVE_RULES.get(model["id"])
    if (
        rule
        and model.get("source", {}).get("revision") == "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
    ):
        namespace, files = rule
        if all(f"model_files/{namespace}/{name}" in weights for name in files):
            weights.pop(f"downloads/{namespace}.zip", None)
    if (
        model["id"] == "invismark"
        and model.get("source", {}).get("revision") == "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
        and weights.get("model_files/invismark/invismark.ckpt") == INVISMARK_CHECKPOINT
    ):
        weights = {
            key: value for key, value in weights.items() if not INVISMARK_EVENT.fullmatch(key)
        }
    return {"registry": model, "weights": weights}


def work_child(job: dict):
    from .worker import adapter_instance, runtime_info, seed_everything

    bank = Path(job["bank"]).resolve()
    model = job["model"]
    meta = bank / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    # Runtime/OS information is per execution; model/checkpoint content is fixed.
    _freeze(meta / "model_identity.json", inference_identity(prepared["identity"]))
    config = read_json(prepared["config"])
    print(f"WM_DIAG BANK loading {model['id']}", flush=True)
    seed_everything(0)
    adapter = adapter_instance(config)
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    provenance = {
        "id": job["session"],
        "fingerprint": prepared["fingerprint"],
        "runtime": runtime_info(),
        "prepared_identity": prepared["identity"],
    }
    write_json(meta / "executions" / f"{job['session']}.json", provenance)
    compact = {"id": job["session"], "fingerprint": prepared["fingerprint"]}
    failures = 0
    for index, image in enumerate(job["images"], 1):
        try:
            process_image(
                adapter,
                model,
                bank,
                image,
                POLICY,
                compact,
                identity_check=job["settings"].get("identity_check", True),
            )
        except Exception as error:
            failures += 1
            write_json(
                meta / "errors" / f"{job['session']}_{image['image_id']}.json",
                {"error": str(error), "traceback": traceback.format_exc()},
            )
            print(f"WM_DIAG BANK failed COCO {image['image_id']}: {error}", flush=True)
            # Corrupt sealed results are not a per-image inference failure.
            if "Sealed artifact" in str(error):
                raise
        print(f"WM_DIAG BANK progress {model['id']}: {index}/{len(job['images'])}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} images failed for {model['id']}; inspect {meta / 'errors'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "work", "report"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.action == "prepare":
        prepare_child(config)
    elif args.action == "work":
        work_child(config)
    else:
        project = Path(__file__).resolve().parents[2]
        if args.action == "run":
            run(project, config)
        else:
            bank = Path(config["output_root"]).resolve()
            data = read_json(bank / "_inputs/manifest.json")
            models = bank_registry(project)
            selected = list(models) if config.get("models", "all") == "all" else config["models"]
            value = report(bank, models, data["images"][: config.get("n_images", 100)], selected)
            print(
                f"Saved {value['saved_png']}/{value['expected_png']} PNGs. {bank / '_reports/REPORT.md'}"
            )


if __name__ == "__main__":
    main()
