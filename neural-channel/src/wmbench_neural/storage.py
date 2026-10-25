"""Checksummed results and explicitly owned, disposable image packages."""

import hashlib
import io
from pathlib import Path
import zipfile

from wmbench.io import atomic_bytes, read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError, freeze, inside
from wmbench_digital.storage import read_result, result_path, save_result
from .spec import ATTACKS, digest, pack_id

TERMINAL = {"measured", "rejected", "blocked_generation"}


def terminal_index(root, cases):
    """One Drive read per carrier, not one full JSON read per attack."""
    result = {}
    for case in cases:
        record = read_result(result_path(root, case), case)
        result[case["id"]] = {k for k, row in (record or {}).get("rows", {}).items() if row.get("status") in TERMINAL}
    return result


def complete(record):
    return bool(record and set(record["rows"]) == {a["id"] for a in ATTACKS}
                and all(r.get("status") in TERMINAL for r in record["rows"].values())
                and record.get("resize_only", {}).get("status") in {"measured", "rejected"})


def manifest_path(root, method, attack, image_id):
    return Path(root) / method / "packages" / (pack_id(attack, image_id)+".json")


def package_key(identity, cases, attack):
    return digest({"run": identity, "case_keys": [c["key"] for c in cases], "attack": attack})


def cache_root(job):
    # Drive survives Colab runtime loss. ZIPs contain one image's message/mode
    # variants for one attack, not millions of individual Drive PNG files.
    base = Path(job["results"]).resolve()
    root = base / "_temporary" / job["identity"] / job["model"]["id"]
    for candidate in (root, root.parent, root.parent.parent):
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(base):
            raise EvidenceError("Refusing linked or escaped temporary cache")
    root.mkdir(parents=True, exist_ok=True)
    freeze(root / "owner.json", {"identity": job["identity"], "method": job["model"]["id"]})
    return root


def save_package(job, cases, attack, entries, pngs):
    key = package_key(job["identity"], cases, attack)
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(pngs.items()):
            archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data)
    target = inside(cache_root(job), pack_id(attack, cases[0]["image_id"])+".zip")
    data = blob.getvalue()
    checksum = hashlib.sha256(data).hexdigest()
    path = manifest_path(job["results"], job["model"]["id"], attack, cases[0]["image_id"])
    if path.exists():
        old = read_json(path)
        if old["key"] != key or old["sha256"] != checksum:
            raise EvidenceError("Regenerated package differs from its recorded bytes; old evidence retained")
    atomic_bytes(target, data)
    if sha256(target) != checksum:
        raise EvidenceError("Drive ZIP readback failed")
    relative = target.relative_to(Path(job["results"]).resolve()).as_posix()
    value = dict(key=key, archive=relative, sha256=checksum, entries=entries,
                 case_keys=[c["key"] for c in cases], attack=attack, bytes=len(data))
    if path.exists():
        # Keep first timing/provenance, never choose the most favourable rerun.
        value = read_json(path)
        value["archive"] = relative
    write_json(path, value)
    if read_json(path) != value:
        raise EvidenceError("Package manifest readback failed")
    return value


def cached(job, cases, attack):
    path = manifest_path(job["results"], job["model"]["id"], attack, cases[0]["image_id"])
    if not path.exists():
        return False
    value = read_json(path)
    if value["key"] != package_key(job["identity"], cases, attack):
        raise EvidenceError("Package belongs to different inputs")
    target = inside(Path(job["results"]), value["archive"])
    if not target.is_file():
        return False
    if sha256(target) != value["sha256"]:
        raise EvidenceError("Stored attack package is corrupt; do not overwrite")
    return True


def committed(root, cases, attack):
    for case in cases:
        record = read_result(result_path(root, case), case)
        if not record or record["rows"].get(attack["id"], {}).get("status") not in TERMINAL:
            return False
    return True


def remove_committed_package(job, cases, attack):
    if not committed(job["results"], cases, attack):
        raise EvidenceError("Refusing deletion before durable decoding")
    manifest = manifest_path(job["results"], job["model"]["id"], attack, cases[0]["image_id"])
    value = read_json(manifest)
    if value["key"] != package_key(job["identity"], cases, attack):
        raise EvidenceError("Package identity changed")
    expected = inside(cache_root(job), pack_id(attack, cases[0]["image_id"])+".zip")
    target = inside(Path(job["results"]), value["archive"])
    if target.resolve() != expected.resolve() or target.is_symlink():
        raise EvidenceError("Refusing unowned image-package deletion")
    for case in cases:
        row = read_result(result_path(job["results"], case), case)["rows"][attack["id"]]
        if row.get("package_key") != value["key"]:
            raise EvidenceError("Decoder row does not identify this package")
    receipt = dict(package_key=value["key"], archive_sha256=value["sha256"],
        result_sha256={c["id"]: sha256(result_path(job["results"], c)) for c in cases})
    receipt_file = manifest.with_suffix(".deleted.json")
    write_json(receipt_file, receipt)
    if read_json(receipt_file) != receipt:
        raise EvidenceError("Deletion receipt readback failed")
    if target.exists():
        if sha256(target) != value["sha256"]:
            raise EvidenceError("Package changed before deletion")
        target.unlink()  # Exact verified ZIP only. Source bank and records remain.
