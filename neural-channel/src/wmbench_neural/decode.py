"""Unchanged reviewed decoders, strict Clean replay and full-payload scoring."""

from collections import defaultdict
import hashlib
import io
from pathlib import Path
import time
import zipfile
from PIL import Image

from wmbench.io import read_json, write_json, atomic_bytes, sha256
from wmbench_full_payload.storage import EvidenceError, inside
from wmbench_digital import worker
from .spec import ATTACKS, digest
from .storage import (TERMINAL, read_result, result_path, save_result, manifest_path,
                      package_key, committed, remove_committed_package, complete, terminal_index)


FIELDS = ("status", "decoded_bits", "ber", "exact")


def same(a, b):
    return all(a.get(k) == b.get(k) for k in FIELDS)


def blind_metadata(job, case):
    path = inside(Path(job["bank"]), case["decoder_metadata"])
    value = read_json(path)
    for field in ("expected_bits", "payload", "full_payload"):
        if isinstance(value.get(field), list) and any(value[field]):
            raise EvidenceError("Expected payload must not be passed to the decoder")
    return path


def load(job):
    if job["model"]["id"] == "stegastamp":
        from stegastamp_paired_hotfix import install_numerical_profile
        install_numerical_profile()
    return worker.load_adapter(job)


def check_clean(job, adapter, case, record):
    path = inside(Path(job["bank"]), case["output"])
    if sha256(path) != case["carrier_sha256"]:
        raise EvidenceError("Carrier hash changed before Clean replay")
    metadata = blind_metadata(job, case)
    row = worker.measured(adapter, path, metadata, case, int(case["signature"][:8], 16))
    execution = dict(runtime=worker.runtime_info(), session=job["session"])
    if case["method"] == "stegastamp":
        from stegastamp_paired_hotfix import check_clean as check_paired
        check_paired(row, case, adapter, metadata, int(case["signature"][:8], 16),
                     Path(job["results"]), execution, record["rows"], job)
    elif not same(row, case["clean"]):
        write_json(Path(job["results"]) / case["method"] / "clean-drift" / (case["id"]+".json"),
                   dict(bank=case["clean"], fresh=row, execution=execution))
        raise EvidenceError("Fresh Clean differs from the bank; stop, do not reinterpret as an attack")
    old = record["rows"].get("clean")
    if old and not same(row, old):
        raise EvidenceError("Saved neural Clean differs from current extraction")
    if not old:
        record["rows"]["clean"] = {**row, "execution": execution, "image_sha256": case["carrier_sha256"]}
    return metadata, execution


def controls(job):
    root, name = Path(job["results"]), job["model"]["id"]
    cases = read_json(root / name / "cases.json")
    if name == "stegastamp":
        # The same audited 3+1 repeatability rule used in the latest digital run.
        from stegastamp_paired_hotfix import preflight
        preflight(job)
    adapter = load(job)
    for index, case in enumerate(cases, 1):
        if time.time() >= job["deadline"]:
            raise TimeoutError("Clean preflight budget reached")
        path = result_path(root, case)
        record = read_result(path, case) or dict(case_key=case["key"], case=case, rows={})
        if complete(record):
            continue  # No new measurements will be made for this completed carrier.
        metadata, execution = check_clean(job, adapter, case, record)
        if "resize_only" not in record:
            evidence = read_json(root / name / "geometry-inputs.json")[case["id"]]
            temporary = inside(Path(job["slot"]) / "tmp/neural-resize", case["id"]+".png")
            if evidence["case_key"] != case["key"] or sha256(temporary) != evidence["sha256"]:
                raise EvidenceError("Resize-only input does not match common geometry record")
            row = worker.measured(adapter, temporary, metadata, case, int(case["signature"][:8], 16))
            row.update(execution=execution, image_sha256=sha256(temporary),
                       note="Auxiliary native-to-512-to-native resize; not one of the 21 attacks")
            if row["status"] == "error":
                save_result(path, record)
                raise RuntimeError(f"Resize-only decoder technical error: {row}")
            record["resize_only"] = row
        save_result(path, record)
        if index % 10 == 0 or index == len(cases):
            print(f"WM_DIAG NEURAL CLEAN/RESIZE CONTROL {name}: {index}/{len(cases)}", flush=True)


def run(job):
    root, name = Path(job["results"]), job["model"]["id"]
    cases = read_json(root / name / "cases.json")
    adapter = None
    temporary = Path(job["slot"]) / "tmp/neural-decode.png"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for case in cases:
        by_image[case["image_id"]].append(case)
    attacks = [a for a in ATTACKS[1:] if a["engine"] == job["engine"]]
    done = terminal_index(root, cases)
    execution_hashes = {}
    count = 0
    for group in by_image.values():
        if all(a["id"] in done[c["id"]] for a in attacks for c in group):
            for attack in attacks:
                manifest = manifest_path(root, name, attack, group[0]["image_id"])
                if manifest.exists() and inside(root, read_json(manifest)["archive"]).exists():
                    remove_committed_package(job, group, attack)
            continue
        if adapter is None:
            adapter = load(job)
        # One Clean replay for every carrier used in this decoder session.
        records, metas, executions = {}, {}, {}
        for case in group:
            record = read_result(result_path(root, case), case)
            if not record or not record.get("resize_only"):
                raise EvidenceError("Clean/resize preflight did not finish")
            metas[case["id"]], executions[case["id"]] = check_clean(job, adapter, case, record)
            records[case["id"]] = record
        for attack in attacks:
            manifest_file = manifest_path(root, name, attack, group[0]["image_id"])
            if committed(root, group, attack):
                if manifest_file.exists():
                    remove_committed_package(job, group, attack)
                continue
            manifest = read_json(manifest_file)
            if manifest["key"] != package_key(job["identity"], group, attack):
                raise EvidenceError("Attack manifest differs from selected inputs")
            archive_path = inside(root, manifest["archive"])
            if sha256(archive_path) != manifest["sha256"]:
                raise EvidenceError("Attack ZIP checksum mismatch")
            with zipfile.ZipFile(archive_path) as archive:
                for case in group:
                    if time.time() >= job["deadline"]:
                        raise TimeoutError("Decode budget reached; partial results and packages retained")
                    record = records[case["id"]]
                    old = record["rows"].get(attack["id"])
                    if old and old["status"] in TERMINAL:
                        continue
                    entry = manifest["entries"][case["id"]]
                    if entry["case_key"] != case["key"]:
                        raise EvidenceError("Wrong package entry")
                    reference = entry["execution"]
                    expected_execution_path = "_execution/"+attack["engine"]+".json"
                    if expected_execution_path not in execution_hashes:
                        execution_hashes[expected_execution_path] = digest(read_json(inside(root, expected_execution_path)))
                    if (reference["path"] != expected_execution_path or
                            execution_hashes[expected_execution_path] != reference["identity_sha256"]):
                        raise EvidenceError("Neural execution identity changed")
                    if entry["status"] == "blocked_generation":
                        row = dict(status="blocked_generation", ber=None, exact=None, seconds=None,
                                   reason=entry["reason"], decoded_bits=None)
                    else:
                        data = archive.read(entry["member"])
                        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                            raise EvidenceError("Attack PNG member checksum mismatch")
                        with Image.open(io.BytesIO(data)) as image:
                            if image.mode != "RGB" or image.size != (512, 512):
                                raise EvidenceError("Unexpected neural PNG geometry")
                        decoder_data = archive.read(entry["decoder_member"])
                        if hashlib.sha256(decoder_data).hexdigest() != entry["decoder_sha256"]:
                            raise EvidenceError("Native decoder-input PNG checksum mismatch")
                        with Image.open(io.BytesIO(decoder_data)) as image:
                            if image.mode != "RGB" or list(image.size) != case["size"]:
                                raise EvidenceError("Native decoder-input geometry mismatch")
                        atomic_bytes(temporary, decoder_data)
                        row = worker.measured(adapter, temporary, metas[case["id"]], case, int(case["signature"][:8], 16))
                        row["decoder_input_sha256"] = sha256(temporary)
                    row.update(package_key=manifest["key"], parameters=entry["parameters"], seed=entry["seed"],
                               image_sha256=entry.get("sha256"), attack_quality=entry.get("attack_quality"),
                               attack_timing=entry["timing"], generation_execution=entry["execution"],
                               execution=executions[case["id"]])
                    if old:
                        row["previous_technical_attempts"] = [*old.get("previous_technical_attempts", []),
                            {k: v for k, v in old.items() if k != "previous_technical_attempts"}]
                    record["rows"][attack["id"]] = row
                    # A verified durable result precedes every cache deletion.
                    save_result(result_path(root, case), record)
                    if row["status"] == "error":
                        raise RuntimeError(f"Decoder failed: {name}/{case['id']}/{attack['id']}: {row.get('reason')}")
                    count += 1
            remove_committed_package(job, group, attack)
            print(f"WM_DIAG NEURAL DECODE {name}: committed {count} new rows; removed {attack['id']}/{group[0]['image_id']} ZIP", flush=True)
