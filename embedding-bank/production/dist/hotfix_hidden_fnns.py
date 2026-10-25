"""Narrow, audited repair layer for the frozen production release.

The original source archive stays byte-identical. Only Meta HiDDeN identity
comparison and FNNS damaged-carrier recovery are changed. No BER-based retries.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import types

META = "meta_hidden_revisited"
FNNS = "fnns_omniseal"
TARGETS = (META, FNNS)
VERSION = "hidden-fnns-storage-v1"
PYC = "hidden/__pycache__/attenuations.cpython-310.pyc"


def canonical_meta(identity):
    """Ignore exactly one generated cache, never a source or checkpoint."""
    from wmbench.adapters.meta_hidden import ASSET_HASHES
    from wmbench_full_payload.storage import EvidenceError

    result = copy.deepcopy(identity)
    if result.get("registry", {}).get("id") != META:
        raise EvidenceError("Meta cache exception requested for another method")
    weights = result.get("weights", {})
    weights.pop(PYC, None)
    if weights != ASSET_HASHES:
        raise EvidenceError("Meta source/checkpoint inventory differs from pinned assets")
    return result


def freeze_meta(path, value, original_freeze):
    from wmbench.io import read_json, write_json
    from wmbench_full_payload.storage import EvidenceError

    path = Path(path)
    if path.name != "model_identity.json" or path.parent.parent.name != META:
        return original_freeze(path, value)
    canonical_meta(value)
    if not path.exists():
        return original_freeze(path, value)
    previous = read_json(path)
    if previous == value:
        return previous
    if canonical_meta(previous) != canonical_meta(value):
        raise EvidenceError("Meta registry/source/checkpoint identity changed")
    receipt = {"fix": VERSION, "ignored_generated_file": PYC,
               "previous_identity": previous, "current_identity": value}
    write_json(path.parent / "repairs" / f"meta_cache_{time.time_ns()}.json", receipt)
    print("META: only generated .pyc drift; original identity retained", flush=True)
    return previous


def meta_source_classes(root):
    """Same pinned author classes, compiled from verified .py, not cached .pyc."""
    import torch
    from wmbench.adapters.meta_hidden import verify_assets

    root = Path(root)
    verify_assets(root)
    attenuation_path = root / "hidden/attenuations.py"
    attenuation = types.ModuleType("wmbench_meta_attenuations")
    attenuation.__file__ = str(attenuation_path)
    exec(compile(attenuation_path.read_text("utf-8"), str(attenuation_path), "exec"),
         attenuation.__dict__)
    source = root / "hidden/models.py"
    wanted = {"ConvBNRelu", "HiddenEncoder", "HiddenDecoder", "EncoderWithJND"}
    nodes = [node for node in ast.parse(source.read_text("utf-8")).body
             if isinstance(node, ast.ClassDef) and node.name in wanted]
    if len(nodes) != 4 or {node.name for node in nodes} != wanted:
        raise ValueError("Unexpected Meta inference classes")
    namespace = {"__name__": "wmbench_meta_classes", "torch": torch,
                 "nn": torch.nn, "attenuations": attenuation}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def verified_copy(source, target, expected=None):
    """Keep a second copy; flush and read it back before considering it durable."""
    from wmbench.io import atomic_bytes, sha256
    from wmbench_full_payload.storage import EvidenceError

    source, target = Path(source), Path(target)
    data = source.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    if expected is not None and checksum != expected:
        raise EvidenceError(f"Recovery source changed: {source}")
    atomic_bytes(target, data)
    if sha256(target) != checksum:
        raise EvidenceError(f"Recovery copy did not persist: {target}")
    return checksum


def backup_path(root, checksum):
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid SHA256")
    return Path(root) / FNNS / "_meta/png_backups" / f"{checksum}.png"


def inspect_pair(root, model, image, replicate):
    """Validate all evidence; classify ONLY missing/changed carrier PNGs repairable."""
    from wmbench.io import read_json, sha256
    from wmbench_bank.contracts import MODES, message
    from wmbench_bank.worker import signature
    from wmbench_full_payload.worker import task_id, record_path
    from wmbench_full_payload.storage import EvidenceError, inside

    if model["id"] != FNNS:
        raise EvidenceError("FNNS recovery cannot operate on another method")
    root = Path(root)
    stem, key = task_id(image, replicate), signature(model, image, replicate)
    carriers = {f"{FNNS}/{mode}/{stem}.png" for mode in MODES}
    stages = [*MODES, *(mode + "_decode" for mode in MODES)]
    records, expected, damaged = [], {}, {}
    for stage in stages:
        path = record_path(root, model, stem, stage)
        if not path.exists():
            continue
        row = read_json(path)
        if row.get("signature") != key or "artifacts" not in row:
            raise EvidenceError(f"Invalid FNNS record: {path}")
        if stage == "native" and row.get("status") == "saved":
            if row.get("assignment") != message(model, image["image_id"], replicate):
                raise EvidenceError("FNNS payload contract changed")
        records.append(path)
        for artifact in row["artifacts"]:
            relative, checksum = artifact["path"], artifact["sha256"]
            if relative in expected and expected[relative] != checksum:
                raise EvidenceError("Conflicting FNNS evidence hashes")
            expected[relative] = checksum
            target = inside(root, relative)
            actual = sha256(target) if target.is_file() else None
            if actual != checksum:
                if relative not in carriers:
                    raise EvidenceError(f"Non-carrier evidence damaged; not auto-repairable: {relative}")
                damaged[relative] = {"expected": checksum, "actual": actual}
    return records, carriers, damaged


def finish_retirement(root, folder):
    """Resume an interrupted quarantine without deleting any unarchived bytes."""
    from wmbench.io import read_json, sha256, write_json
    from wmbench_full_payload.storage import EvidenceError, inside

    root, folder = Path(root), Path(folder)
    plan = read_json(folder / "plan.json")
    for entry in plan["files"]:
        relative = entry["path"]
        # Plans may retire only this FNNS pair's records and four carrier paths.
        stem = plan["stem"]
        allowed = {f"{FNNS}/{mode}/{stem}.png" for mode in ("native", "matched_39", "matched_42", "matched_45")}
        allowed.update(f"{FNNS}/_meta/records/{stem}_{mode}{suffix}.json"
                       for mode in ("native", "matched_39", "matched_42", "matched_45")
                       for suffix in ("", "_decode"))
        if relative not in allowed:
            raise EvidenceError("Unsafe FNNS recovery plan path")
        archived = inside(folder / "original", relative)
        if not archived.is_file() or sha256(archived) != entry["sha256"]:
            raise EvidenceError("Recovery archive incomplete; originals will not be removed")
    # Validate ALL backups before retiring ANY active evidence.
    for entry in plan["files"]:
        target = inside(root, entry["path"])
        if target.exists():
            if sha256(target) != entry["sha256"]:
                raise EvidenceError("Active file changed during recovery; stop concurrent runners")
            target.unlink()  # Exact pair only; verified archived copy remains on Drive.
    write_json(folder / "retired.json", {"time": time.time(), "fix": VERSION})


def repair_pair(root, model, image, replicate):
    """Restore exact backup bytes, else quarantine this pair for one technical retry."""
    from wmbench.io import read_json, sha256, write_json
    from wmbench_full_payload.worker import task_id
    from wmbench_full_payload.storage import inside, EvidenceError

    root = Path(root)
    if model["id"] != FNNS:
        raise EvidenceError("Recovery is restricted to FNNS")
    stem = task_id(image, replicate)
    base = root / FNNS / "_meta/repairs" / stem
    for plan_path in sorted(base.glob("*/plan.json")):
        folder = plan_path.parent
        if not (folder / "retired.json").exists():
            finish_retirement(root, folder)
    records, carriers, damaged = inspect_pair(root, model, image, replicate)
    if not damaged:
        return None
    for relative, evidence in list(damaged.items()):
        backup = backup_path(root, evidence["expected"])
        if backup.is_file() and sha256(backup) == evidence["expected"]:
            folder = base / str(time.time_ns())
            target = inside(root, relative)
            if target.is_file():
                verified_copy(target, folder / "damaged.png", evidence["actual"])
            write_json(folder / "exact_restore.json", {"path": relative, **evidence, "fix": VERSION})
            verified_copy(backup, target, evidence["expected"])
            del damaged[relative]
            print(f"FNNS: restored original bytes from verified backup: {relative}", flush=True)
    if not damaged:
        return None
    # Never compare BER or optimize settings to decide whether to repeat a pair.
    folder = base / str(time.time_ns())
    candidates = records + [inside(root, p) for p in sorted(carriers) if inside(root, p).is_file()]
    files = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        checksum = verified_copy(path, folder / "original" / relative)
        files.append({"path": relative, "sha256": checksum})
    write_json(folder / "plan.json", {"fix": VERSION, "stem": stem, "files": files,
        "damaged": damaged, "reason": "carrier_file_integrity_failure",
        "policy": "same image, same payload and task seed; replace all dependent modes; no BER selection"})
    finish_retirement(root, folder)
    print(f"FNNS: archived damaged pair {stem}; regenerate only this pair and dependent modes", flush=True)
    return folder


def install():
    """Patch narrowly scoped call sites; keep the frozen release files untouched."""
    from wmbench.io import read_json, sha256, write_json
    from wmbench_full_payload.storage import EvidenceError, inside
    from wmbench_bank import worker, runner
    from wmbench import colab_canary
    from wmbench.adapters import meta_hidden

    old_freeze, old_encode, old_sealed = worker.freeze, worker.encode_pair, worker.sealed
    old_work = worker.work
    old_complete = runner.complete
    old_monitor = colab_canary.monitored_process
    meta_hidden.load_classes = meta_source_classes
    worker.freeze = lambda path, value: freeze_meta(path, value, old_freeze)

    def protected_seal(root, path, key, data, artifacts=()):
        # Only FNNS carriers created now gain content-addressed backup copies.
        if Path(path).parent.parent.parent.name == FNNS and data.get("status") == "saved":
            png = inside(root, data["output"])
            worker.check_png(png, 512)
            checksum = sha256(png)
            backup = backup_path(root, checksum)
            if not backup.is_file() or sha256(backup) != checksum:
                verified_copy(png, backup, checksum)
            data = {**data, "storage_hotfix": VERSION, "png_backup": backup.relative_to(root).as_posix()}
        return old_sealed(root, path, key, data, artifacts)

    class DurableFNNS:
        def __init__(self, adapter):
            self.adapter = adapter

        def __getattr__(self, name):
            return getattr(self.adapter, name)

        def embed(self, source, png, payload, metadata):
            self.adapter.embed(source, png, payload, metadata)
            worker.check_png(png, 512)
            # Original worker moves pending/native.png. Retain a verified second
            # copy BEFORE that move, so a later Drive truncation is recoverable.
            verified_copy(png, Path(png).with_name("native_backup.png"))

    def encode(adapter, model, root, image, replicate, execution):
        if model["id"] != FNNS:
            return old_encode(adapter, model, root, image, replicate, execution)
        stem = worker.task_id(image, replicate)
        base = Path(root) / FNNS / "_meta/repairs" / stem
        for plan in sorted(base.glob("*/plan.json")):
            if not (plan.parent / "retired.json").exists():
                finish_retirement(root, plan.parent)
        try:
            row = old_encode(DurableFNNS(adapter), model, root, image, replicate, execution)
        except EvidenceError:
            # Healthy pairs use exactly the original cache checks. Do not add
            # another full-bank hash scan over the Drive mount on every resume.
            _, _, damaged = inspect_pair(root, model, image, replicate)
            if not damaged:
                raise
            repair_pair(root, model, image, replicate)
            row = old_encode(DurableFNNS(adapter), model, root, image, replicate, execution)
        for plan in sorted(base.glob("*/plan.json")):
            folder = plan.parent
            if (folder / "recomputed.json").exists():
                continue
            write_json(folder / "recomputed.json", {"fix": VERSION, "execution": execution,
                "new_native_artifacts": row["artifacts"], "old_evidence": "original/"})
        return row

    def complete(root, model, images):
        if model["id"] == FNNS:
            pending = Path(root) / FNNS / "_meta/repairs"
            if any(not (p.parent / "retired.json").exists() for p in pending.glob("*/*/plan.json")):
                return False
        try:
            return old_complete(root, model, images)
        except EvidenceError as error:
            if model["id"] != FNNS:
                raise
            # Only a named carrier integrity error can be deferred to recovery.
            # Unknown metadata/signature/checkpoint differences remain fatal.
            for image in images:
                for replicate in range(1, 6):
                    stem = worker.task_id(image, replicate)
                    for mode in worker.MODES:
                        png = Path(root) / FNNS / mode / f"{stem}.png"
                        if str(error).startswith(f"Damaged saved artifact: {png};"):
                            if inspect_pair(root, model, image, replicate)[2]:
                                return False
            raise

    def monitor(command, *args, **kwargs):
        command = list(command)
        if "wmbench_bank.runner" in command and "--settings" in command:
            job_path = Path(command[command.index("--settings") + 1])
            job = read_json(job_path)
            if job.get("model", {}).get("id") in TARGETS:
                index = command.index("wmbench_bank.runner")
                if command[index - 1] != "-m":
                    raise EvidenceError("Unexpected production worker invocation")
                command[index - 1:index + 1] = [str(Path(__file__).resolve())]
        return old_monitor(command, *args, **kwargs)

    def work(job):
        if job["model"]["id"] in TARGETS:
            receipt = Path(job["output"]) / job["model"]["id"] / "_meta/repairs" / f"hotfix_{job['session']}.json"
            write_json(receipt, {"fix": VERSION, "helper_sha256": sha256(Path(__file__)),
                "session": job["session"], "model": job["model"]["id"],
                "scientific_release_files_modified": False})
        return old_work(job)

    worker.sealed = protected_seal
    worker.encode_pair = encode
    worker.work = work
    runner.complete = complete
    colab_canary.monitored_process = monitor


def main():
    install()
    from wmbench_bank.runner import main as original_main
    original_main()


if __name__ == "__main__":
    main()
