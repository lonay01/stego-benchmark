"""User-authorized replacement of four lost carriers, not byte-exact restoration.

Only the fixed four image/message pairs may be replaced. Frozen production
encode/match/decode functions are reused; measured outcomes are never selected.
Every old byte is backed up before a resumable commit. Other methods are excluded.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from wmbench.io import atomic_bytes, read_json, sha256, write_json
from stegastamp_byte_recovery import TARGETS

REVISION = "stegastamp-lost-four-user-approved-20261016"
MODES = ("native", "matched_39", "matched_42", "matched_45")


def signature(path):
    return sha256(path) if path.is_file() else None


def copy_checked(source, target):
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError(f"Backup/staging conflict: {target}")
    else:
        atomic_bytes(target, data)
    if target.read_bytes() != data:
        raise RuntimeError(f"Copy readback failed: {target}")


def safe(root, relative):
    p = root / relative
    if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
        raise ValueError(f"Unsafe recovery path: {relative}")
    return p


def bank_targets(bank):
    paths = set()
    for stem in TARGETS:
        for mode in MODES:
            paths.add(f"stegastamp/{mode}/{stem}.png")
            for suffix in (mode, mode + "_decode"):
                relative = f"stegastamp/_meta/records/{stem}_{suffix}.json"
                paths.add(relative)
                p = bank / relative
                if p.exists():
                    for artifact in read_json(p).get("artifacts", []):
                        name = artifact["path"]
                        if name.startswith("stegastamp/_meta/attempts/"):
                            paths.add(name)
    return sorted(paths)


def stage(job_path, work):
    from wmbench_bank.worker import encode_pair, decode_pair
    from wmbench_full_payload.storage import cached
    from wmbench_digital import worker
    from stegastamp_cuda_hotfix import install_worker_guard

    job = read_json(job_path)
    bank, model = Path(job["bank"]), job["model"]
    if model["id"] != "stegastamp":
        raise ValueError("StegaStamp only")
    staged = work / "replacement_bank"
    manifest_path = work / "replacement_manifest.json"
    if manifest_path.exists():
        plan = read_json(manifest_path)
        for name, digest in plan["new_files"].items():
            if signature(safe(staged, name)) != digest:
                raise RuntimeError(f"Staged replacement changed: {name}")
        print("Reusing the first sealed replacement; no outcome retry", flush=True)
        return
    if staged.exists():
        raise RuntimeError("Unsealed replacement stage exists; inspect it before any retry")
    # Capture exact current bytes before staging. Only the known empty Native
    # files are allowed to disagree with their old sealed artifact hashes.
    from stegastamp_byte_recovery import collect_evidence
    collect_evidence(bank)
    if any((bank / f"stegastamp/native/{stem}.png").stat().st_size for stem in TARGETS):
        raise RuntimeError("Expected exactly the four known empty Native files")
    old = {name: signature(safe(bank, name)) for name in bank_targets(bank)}
    staged.mkdir(parents=True)
    images = {int(x["image_id"]): x for x in read_json(bank / "_inputs/manifest.json")["images"]}
    for image_id in sorted({int(s.split("_")[0]) for s in TARGETS}):
        image = images[image_id]
        if signature(bank / image["path"]) != image["sha256"]:
            raise RuntimeError("Original COCO source changed")
        copy_checked(bank / image["path"], staged / image["path"])
        for suffix in (".png", ".json"):
            rel = f"_inputs/crops/400/{image_id:012d}{suffix}"
            copy_checked(bank / rel, staged / rel)
    install_worker_guard()
    adapter = worker.load_adapter(job)
    adapter._load()  # actual CUDA provider is required before recording runtime
    execution = dict(session=REVISION, runtime=worker.runtime_info(), recovery={
        "revision": REVISION, "reason": "four zero-byte Drive carriers; no prior revisions",
        "user_authorized": True, "byte_exact_restoration": False,
        "old_native_hashes": TARGETS,
        "policy": "one re-embedding per lost pair; no selection by BER/PSNR"})
    for stem in TARGETS:
        image = images[int(stem.split("_")[0])]
        rep = int(stem.split("_m")[1])
        prior = read_json(bank / f"stegastamp/_meta/records/{stem}_native.json")
        row = encode_pair(adapter, model, staged, image, rep, execution)
        if row["assignment"] != prior["assignment"] or row["signature"] != prior["signature"]:
            raise RuntimeError("Payload or task signature changed")
        if decode_pair(adapter, model, staged, image, rep, execution):
            raise RuntimeError("Technical Clean decode failure; no bank changes")
        for mode in MODES:
            path = staged / f"stegastamp/_meta/records/{stem}_{mode}.json"
            encoded = cached(staged, path, row["signature"])
            if encoded["status"] not in {"saved", "unmatched"}:
                raise RuntimeError("Incomplete matched record")
            if encoded["status"] == "saved":
                decoded = cached(staged, path.with_name(stem + "_" + mode + "_decode.json"), row["signature"])
                if decoded["status"] not in {"measured", "rejected"}:
                    raise RuntimeError("Invalid Clean control")
                print(json.dumps(dict(stem=stem, mode=mode, psnr=encoded["psnr"], ber=decoded.get("ber"))), flush=True)
    new = {p.relative_to(staged).as_posix(): sha256(p)
           for p in sorted((staged / "stegastamp").rglob("*")) if p.is_file()}
    # Shared source crops remain exactly the original files.
    for p in (staged / "_inputs").rglob("*"):
        if p.is_file() and signature(p) != signature(bank / p.relative_to(staged)):
            raise RuntimeError("Shared source/crop differs; cannot commit")
    if old != {name: signature(safe(bank, name)) for name in old}:
        raise RuntimeError("Bank changed during staging")
    write_json(manifest_path, dict(revision=REVISION, bank=str(bank), stage=str(staged),
        old_files=old, new_files=new, created_unix=time.time(), execution=execution))
    print("Replacement stage sealed; no active bank files replaced yet", flush=True)


def digital_targets(results):
    folder = results / "stegastamp"
    relative = {"cases.json", "coverage.json", "packages-ready.json",
                "_fast/catalogue.json.gz", "_fast/complete.json.gz", "_fast/completed-selection.json.gz"}
    for image in {int(s.split("_")[0]) for s in TARGETS}:
        relative.add(f"_fast/bank-images/{image}.json.gz")
        # ResultCache uses the unpadded decimal image ID.
        relative.add(f"_fast/results/{image}.json.gz")
        relative.add(f"_fast/results/{image:012d}.json.gz")
    for stem in TARGETS:
        for mode in MODES:
            case = stem + "_" + mode
            relative.update({f"records/{case}.json.gz", f"quality/{case}.json", f"packages/{case}.json"})
    return ["stegastamp/" + n for n in sorted(relative) if (folder / n).is_file()]


def commit(work, results):
    """Idempotent journaled commit. No result or carrier is discarded without backup."""
    plan = read_json(work / "replacement_manifest.json")
    if plan["revision"] != REVISION:
        raise RuntimeError("Unexpected replacement revision")
    bank, staged = Path(plan["bank"]), Path(plan["stage"])
    if staged.resolve() != (work / "replacement_bank").resolve():
        raise RuntimeError("Unexpected staging directory")
    if work.is_relative_to(bank) or bank.is_relative_to(work):
        raise RuntimeError("Bank and recovery must be disjoint")
    allowed = {f"stegastamp/{mode}/{stem}.png" for stem in TARGETS for mode in MODES}
    allowed.update(f"stegastamp/_meta/records/{stem}_{mode}{suffix}.json"
                   for stem in TARGETS for mode in MODES for suffix in ("", "_decode"))
    for name in set(plan["old_files"]) | set(plan["new_files"]):
        attempt = any(name.startswith(f"stegastamp/_meta/attempts/{stem}_") for stem in TARGETS)
        if name not in allowed and not attempt:
            raise RuntimeError(f"Replacement exceeds the four approved pairs: {name}")
        safe(bank, name)
    receipt = bank / "stegastamp/_meta/recoveries" / (REVISION + ".json")
    if receipt.exists():
        done = read_json(receipt)
        if done.get("state") != "committed":
            raise RuntimeError("Unexpected recovery state")
        for name, digest in plan["new_files"].items():
            if signature(safe(bank, name)) != digest:
                raise RuntimeError(f"Committed replacement changed: {name}")
        print("Approved replacement already committed; no re-embedding", flush=True)
        return
    for name, digest in plan["new_files"].items():
        if signature(safe(staged, name)) != digest:
            raise RuntimeError(f"Staged replacement changed: {name}")
    journal_path = work / "commit_journal.json"
    if not journal_path.exists():
        # Back up ALL affected bank bytes before making the first active change.
        for name, digest in plan["old_files"].items():
            if signature(safe(bank, name)) != digest:
                raise RuntimeError(f"Original changed before backup: {name}")
            if digest is not None:
                copy_checked(bank / name, work / "previous_bank" / name)
        digital = {name: signature(safe(results, name)) for name in digital_targets(results)}
        for name in digital:
            copy_checked(results / name, work / "previous_digital" / name)
        write_json(journal_path, dict(revision=REVISION, digital=digital, state="backed_up"))
    journal = read_json(journal_path)
    # Revalidate the backups and every existing target, including partial resumes.
    for name, digest in plan["old_files"].items():
        if digest is not None and signature(work / "previous_bank" / name) != digest:
            raise RuntimeError("Original backup readback failed")
        current = signature(safe(bank, name))
        if current not in {digest, plan["new_files"].get(name)}:
            raise RuntimeError(f"Unexpected active bank bytes: {name}")
    # New attempt sidecars first, then PNGs, then sealed metadata records.
    order = sorted(plan["new_files"], key=lambda n: ("/_meta/records/" in n, "/_meta/attempts/" not in n, n))
    for name in order:
        target, digest = safe(bank, name), plan["new_files"][name]
        if name not in plan["old_files"] and target.exists() and signature(target) != digest:
            raise RuntimeError("New sidecar collision")
        if signature(target) != digest:
            atomic_bytes(target, (staged / name).read_bytes())
        if signature(target) != digest:
            raise RuntimeError(f"Replacement readback failed: {name}")
    # An old saved matched image may now be unattainable. Archive stale active
    # rows/images; never leave them masquerading as new replacement evidence.
    for name in plan["old_files"]:
        if name in plan["new_files"] or "/_meta/attempts/" in name:
            continue
        target = safe(bank, name)
        if target.exists():
            destination = work / "withdrawn_bank" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, destination)
    for name, digest in journal["digital"].items():
        if signature(work / "previous_digital" / name) != digest:
            raise RuntimeError("Digital backup changed")
        target = safe(results, name)
        if target.exists():
            if signature(target) != digest:
                raise RuntimeError(f"Digital file changed concurrently: {name}")
            destination = work / "withdrawn_digital" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, destination)
    report = dict(revision=REVISION, state="committed", targets=list(TARGETS),
        original_native_sha256=TARGETS, new_files=plan["new_files"],
        previous_bank=str(work / "previous_bank"), previous_digital=str(work / "previous_digital"),
        execution=plan["execution"], committed_unix=time.time(),
        note="Four lost pairs replaced with user approval; not an exact restoration. Other cases unchanged.")
    write_json(receipt, report)
    write_json(work / "replacement_receipt.json", report)
    print("COMMITTED: four approved replacement pairs; old evidence archived; StegaStamp caches invalidated", flush=True)


def run(job_path, work, results):
    from wmbench.colab_canary import controller_lease
    from wmbench_fast.launch import alive
    from stegastamp_cuda_hotfix import library_environment
    job = read_json(job_path)
    bank = Path(job["bank"])
    if (work.is_relative_to(bank) or bank.is_relative_to(work)
            or work.is_relative_to(results) or results.is_relative_to(work)):
        raise ValueError("Recovery directory must be separate")
    launcher = results / "_reports/fast_launcher.json"
    if launcher.exists() and alive(read_json(launcher)):
        raise RuntimeError("Stop the active digital runner before replacement")
    with controller_lease(results / "_reports"):
        work.mkdir(parents=True, exist_ok=True)
        receipt = work / "replacement_receipt.json"
        if not receipt.exists():
            prepared = read_json(Path(job["output"]) / "stegastamp/_meta/prepared.json")
            if prepared["status"] != "ready":
                raise RuntimeError("Run the original isolated StegaStamp preparation first")
            subprocess.run([prepared["python"], "-u", __file__, "stage", "--job", str(job_path),
                            "--work", str(work), "--results", str(results)],
                           env=library_environment(os.environ.copy()), check=True, timeout=1800)
        commit(work, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "stage"))
    for key in ("job", "work", "results"):
        p.add_argument("--" + key, required=True, type=lambda x: Path(x).resolve())
    args = p.parse_args()
    if args.action == "stage":
        stage(args.job, args.work)
    else:
        run(args.job, args.work, args.results)
