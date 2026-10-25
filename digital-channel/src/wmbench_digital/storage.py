"""Atomic checkpoints and tightly scoped deletion of owned attack packages."""

import gzip
import hashlib
import json
from pathlib import Path

from wmbench.io import atomic_bytes, read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError, freeze, inside
from .spec import ATTACKS, digest


def result_path(root, case):
    return Path(root) / case["method"] / "records" / (case["id"] + ".json.gz")


def save_result(path, value):
    envelope = {"content_sha256": digest(value), "record": value}
    encoded = gzip.compress(json.dumps(envelope, sort_keys=True, allow_nan=False).encode(), mtime=0)
    atomic_bytes(path, encoded)
    # Readback is required before any intermediate is eligible for deletion.
    if sha256(path) != hashlib.sha256(encoded).hexdigest():
        raise EvidenceError(f"Durable checkpoint readback failed: {path}")


def read_result(path, case=None):
    path = Path(path)
    if not path.exists():
        return None
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    value = envelope["record"]
    if digest(value) != envelope["content_sha256"]:
        raise EvidenceError(f"Result content checksum mismatch: {path}")
    if case and value["case_key"] != case["key"]:
        raise EvidenceError(f"Checkpoint belongs to a different carrier: {path}")
    return value


def complete(value):
    return bool(value and set(value["rows"]) == {a["id"] for a in ATTACKS}
                and all(r["status"] in {"measured", "rejected"} for r in value["rows"].values()))


def cache_root(base, identity, method):
    target = Path(base).resolve() / "attack-cache" / identity / method
    target.mkdir(parents=True, exist_ok=True)
    freeze(target / "owner.json", {"identity": identity, "method": method})
    return target


def remove_packages(root, cases, manifests, allowed_bases, identity, method):
    """Never remove the bank, a broad directory, or an uncommitted package."""
    receipts = []
    for case in cases:
        path = result_path(root, case)
        result = read_result(path, case)
        if not complete(result):
            raise EvidenceError("Refusing cache cleanup before all durable decodes are complete")
        receipts.append({"case_key": case["key"], "result_sha256": sha256(path)})
    write_json(Path(root) / method / "commits" / (digest(receipts) + ".json"), receipts)
    allowed = [(Path(b).resolve() / "attack-cache" / identity / method) for b in allowed_bases]
    names = {c["key"]: c["id"]+".zip" for c in cases}
    for manifest in manifests:
        folder = Path(manifest["archive"]).parent.resolve()
        if folder not in allowed or folder.is_symlink():
            raise EvidenceError(f"Unsafe cache deletion: {folder}")
        if names.get(manifest.get("case_key")) != Path(manifest["archive"]).name:
            raise EvidenceError("Cache filename is not an expected carrier ZIP")
        if read_json(folder / "owner.json") != {"identity": identity, "method": method}:
            raise EvidenceError("Cache owner mismatch")
        file = inside(folder, Path(manifest["archive"]).name)
        if file.is_symlink():
            raise EvidenceError("Refusing linked cache file")
        if file.exists():
            if sha256(file) != manifest["sha256"]:
                raise EvidenceError("Cache changed before cleanup")
            file.unlink()  # Only this verified, disposable ZIP. Receipts remain.
