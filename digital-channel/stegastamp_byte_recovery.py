"""Restore four empty StegaStamp PNGs only if original SHA256 hashes are reproduced.

Run in the frozen StegaStamp environment. Existing benchmark records are read-only.
Candidate generation and commit are separate commands; commit never runs a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

TARGETS = {
    "000000106563_m03": "85f15c7e12a264e5b00248c253cbe1eb5cb139735ebb591ba23fcc0f324e3afa",
    "000000106563_m04": "fb658f2566c70d53c670430086b8788f1967afbd8f3f14dec66dc78783d0a639",
    "000000319607_m04": "a5749e420e2cc8759649b442578d95c40e7f7f3985562b2071ea54f9a9aa0338",
    "000000319607_m05": "3bf744565ec9763e646ba7b53850ffa328ce438ad34c307a8b4ae5756a6ee8db",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes bank: {relative}")
    return path


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def collect_evidence(bank):
    """Verify every referenced artifact except the four explicitly empty targets."""
    records = {}
    checked = {}
    missing = {f"stegastamp/native/{stem}.png": value for stem, value in TARGETS.items()}
    for stem, expected in TARGETS.items():
        paths = sorted((bank / "stegastamp/_meta/records").glob(f"{stem}_*.json"))
        if not paths:
            raise ValueError(f"No sealed records for {stem}")
        for path in paths:
            records[path.relative_to(bank).as_posix()] = sha(path)
            row = read(path)
            if not row.get("signature") or not isinstance(row.get("artifacts"), list):
                raise ValueError(f"Unsealed record: {path}")
            for item in row["artifacts"]:
                relative, digest = item["path"], item["sha256"]
                artifact = inside(bank, relative)
                if relative in missing:
                    if digest != missing[relative]:
                        raise ValueError(f"Conflicting original hash: {relative}")
                    if artifact.stat().st_size != 0 and sha(artifact) != digest:
                        raise ValueError(f"Nonempty changed artifact: {relative}")
                else:
                    if relative not in checked:
                        checked[relative] = sha(artifact)
                    if checked[relative] != digest:
                        raise ValueError(f"Other damaged artifact: {relative}")
        native = read(bank / f"stegastamp/_meta/records/{stem}_native.json")
        if native["output"] != f"stegastamp/native/{stem}.png" or native["status"] != "saved":
            raise ValueError(f"Unexpected native record: {stem}")
        if not any(a == {"path": native["output"], "sha256": expected} for a in native["artifacts"]):
            raise ValueError(f"Native hash missing: {stem}")
    return records


def candidates(job_path, work):
    from wmbench_digital.worker import load_adapter
    from wmbench.worker import seed_everything, runtime_info

    job = read(job_path)
    if job["model"]["id"] != "stegastamp":
        raise ValueError("This recovery is StegaStamp-only")
    bank = Path(job["bank"]).resolve()
    work = Path(work).resolve()
    if work.is_relative_to(bank) or bank.is_relative_to(work):
        raise ValueError("Recovery output must be separate from the bank")
    work.mkdir(parents=True, exist_ok=False)
    evidence = collect_evidence(bank)
    save_new(work / "original_record_hashes.json", evidence)
    # Includes checkpoint, dependency lock, installed-package and geometry checks.
    from stegastamp_cuda_hotfix import install_worker_guard
    install_worker_guard()
    adapter = load_adapter(job)
    runtime = runtime_info()
    first = read(bank / "stegastamp/_meta/records/000000106563_m03_native.json")
    original_runtime = first["execution"]["runtime"]
    for field in ("python", "architecture", "torch", "cuda_build", "gpu", "cudnn"):
        if runtime.get(field) != original_runtime.get(field):
            raise ValueError(f"Original runtime differs: {field}: {original_runtime.get(field)} -> {runtime.get(field)}")
    outcomes = []
    for stem, expected in TARGETS.items():
        row = read(bank / f"stegastamp/_meta/records/{stem}_native.json")
        source = inside(bank, row["source_crop"])
        png, metadata = work / f"{stem}.png", work / f"{stem}.metadata.json"
        seed = int(row["signature"][:8], 16)
        seed_everything(seed)
        adapter.embed(str(source), str(png), row["assignment"]["payload"], str(metadata))
        actual = sha(png)
        # Blind extraction uses the old, checksum-verified decoder metadata.
        seed_everything(seed)
        decoded = adapter.decode(str(png), str(inside(bank, row["decoder_metadata"])))
        clean = read(bank / f"stegastamp/_meta/records/{stem}_native_decode.json")
        old_bits = clean.get("decoded_bits")
        new_bits = decoded.get("bits")
        if hasattr(new_bits, "tolist"):
            new_bits = new_bits.tolist()
        entry = {"stem": stem, "expected_sha256": expected, "actual_sha256": actual,
                 "byte_identical": actual == expected, "clean_bits_identical": new_bits == old_bits,
                 "candidate_size": png.stat().st_size, "old_clean_ber": clean.get("ber"),
                 "decoded_bits": new_bits, "old_decoded_bits": old_bits}
        outcomes.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if "decoded_bits" not in k}), flush=True)
    if collect_evidence(bank) != evidence:
        raise ValueError("Original records changed during recovery")
    report = {"runtime": runtime, "outcomes": outcomes, "bank": str(bank), "candidate_dir": str(work),
              "ready": all(r["byte_identical"] and r["clean_bits_identical"] for r in outcomes)}
    save_new(work / "candidate_report.json", report)
    print("READY_FOR_BYTE_IDENTICAL_RESTORE" if report["ready"] else "NO_BANK_CHANGES: candidates not identical", flush=True)


def commit(work):
    """Replace only zero-byte targets, after validating all four candidates together."""
    work = Path(work).resolve()
    report = read(work / "candidate_report.json")
    bank = Path(report["bank"]).resolve()
    if not report["ready"] or work.is_relative_to(bank) or bank.is_relative_to(work):
        raise ValueError("No approved byte-identical candidate set")
    evidence = read(work / "original_record_hashes.json")
    if collect_evidence(bank) != evidence:
        raise ValueError("Original records changed")
    for stem, expected in TARGETS.items():
        if sha(work / f"{stem}.png") != expected:
            raise ValueError(f"Candidate changed: {stem}")
    backup = work / "empty_originals"
    backup.mkdir(exist_ok=True)
    receipt = []
    for stem, expected in TARGETS.items():
        target = bank / f"stegastamp/native/{stem}.png"
        if sha(target) == expected:
            receipt.append({"stem": stem, "status": "already_identical", "sha256": expected})
            continue
        if target.stat().st_size != 0:
            raise ValueError(f"Refusing to overwrite nonempty target: {target}")
        original = backup / f"{stem}.png"
        if not original.exists():
            with original.open("xb") as handle:
                handle.write(target.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
        temporary = target.with_name(target.name + f".recovery-{time.time_ns()}")
        with temporary.open("xb") as handle:
            handle.write((work / f"{stem}.png").read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        if sha(temporary) != expected or target.stat().st_size != 0:
            raise ValueError("Readback/race check failed; original left unchanged")
        os.replace(temporary, target)
        if sha(target) != expected:
            raise ValueError(f"Replacement readback failed: {stem}")
        receipt.append({"stem": stem, "status": "restored_identical", "sha256": expected})
    if collect_evidence(bank) != evidence:
        raise ValueError("Record verification changed after restore")
    save_new(work / f"restore_receipt_{time.time_ns()}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print("RESTORED: 4/4 original SHA256 hashes; all referenced record bytes unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("candidates", "commit"))
    parser.add_argument("--job", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "candidates":
        if args.job is None:
            parser.error("--job is required for candidates")
        candidates(args.job, args.work)
    else:
        commit(args.work)


if __name__ == "__main__":
    main()
