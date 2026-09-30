"""Copy verified evidence into a new run; the parent is always read-only."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from authornight.storage import EvidenceError, cached, commit, digest, disjoint, freeze, inside, read, sha256

PARENT_ID = "6b1ba1f5878b65736fce691053685c079300448ada3f1ea3120599bc758469d8"
PARENT_RELEASE = "9f186b48fe81c1443b6db48d91a7c0b828b2ebe2b9246b732b59bb08ea0e8f61"
METHODS = ("rosteals", "sepwater")


def make_plan(settings, code_hash):
    parent, root, scratch = [Path(settings[k]).resolve() for k in ("parent", "output", "scratch")]
    disjoint(parent, root, scratch)
    if not 0 < float(settings["max_hours"]) <= 24:
        raise ValueError("MAX_HOURS must be in (0, 24]")
    prior = read(parent / "plan.json")
    if prior.get("identity") != PARENT_ID or prior.get("code_sha256") != PARENT_RELEASE:
        raise EvidenceError("This recovery is for the reviewed author-night run, not another experiment")
    if digest({k: v for k, v in prior.items() if k != "identity"}) != PARENT_ID:
        raise EvidenceError("Parent plan checksum mismatch")
    if root.exists() and not (root / "plan.json").exists():
        unexpected = set(p.name for p in root.iterdir()) - {"_release", "_logs", "_reports"}
        if unexpected:
            raise EvidenceError(f"Use a new recovery output folder: {sorted(unexpected)}")
    plan = {**prior, "schema": "two-method-author-recovery-v1", "code_sha256": code_hash,
            "parent_identity": PARENT_ID, "parent_root": str(parent),
            "profiles": {m: prior["profiles"][m] for m in METHODS},
            "recovery": {"rosteals": "slackclient==2.9.3; same author network/checkpoint",
                         "sepwater": "same 224px network; BCH decode+correct evaluated separately",
                         "attacks": "same internal floor PNGs, seeds and full severity grid",
                         "distseal": "not scheduled or copied"}}
    plan.pop("identity")
    plan["identity"] = digest(plan)
    freeze(root / "plan.json", plan)
    return plan


def copy_verified(source, target, expected):
    """Stream large files once. Commit only after the expected hash matches."""
    source, target = Path(source), Path(target)
    if target.exists():
        if sha256(target) != expected:
            raise EvidenceError(f"Changed recovery copy: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".import-", dir=target.parent)
    try:
        hasher = hashlib.sha256()
        with os.fdopen(fd, "wb") as outgoing, source.open("rb") as incoming:
            for block in iter(lambda: incoming.read(4 * 1024**2), b""):
                outgoing.write(block)
                hasher.update(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if hasher.hexdigest() != expected:
            raise EvidenceError(f"Parent artifact hash mismatch: {source}")
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def import_record(parent, root, relative, identity):
    old = cached(parent, relative, PARENT_ID, verify=False)
    existing = cached(root, relative, identity, verify=False)
    if existing:
        # A corruption grid may have been extended since import. Do not overwrite it.
        if existing.get("imported_from", {}).get("record_sha256") != old["record_sha256"]:
            raise EvidenceError(f"Conflicting import provenance: {relative}")
        return False
    artifacts = []
    for name, expected in old.get("artifacts", {}).items():
        target = inside(root, name)
        copy_verified(inside(parent, name), target, expected)
        artifacts.append(target)
    body = {k: v for k, v in old.items() if k not in ("identity", "record_sha256", "artifacts", "saved_unix")}
    body["imported_from"] = {"identity": PARENT_ID, "record_sha256": old["record_sha256"],
                             "path": relative, "saved_unix": old["saved_unix"]}
    commit(root, relative, identity, body, artifacts)
    return True


def import_parent(settings, code_hash):
    parent, root = Path(settings["parent"]), Path(settings["output"])
    plan = make_plan(settings, code_hash)
    originals = read(parent / "_inputs/originals.json")
    if [x["image_id"] for x in originals] != [int(x["image_id"]) for x in plan["images"]]:
        raise EvidenceError("Parent originals do not match the frozen image order")
    for item in originals:
        copy_verified(inside(parent, item["path"]), inside(root, item["path"]), item["sha256"])
    freeze(root / "_inputs/originals.json", originals)
    # Do not import old logs, mutable prepared paths, DistSeal assets or environments.
    for method in METHODS:
        for filename in ("execution.json", "weight_audit.json", "author_ecc_selftest.json", "f4_asset_audit.json"):
            path = parent / "_meta" / method / filename
            if path.exists():
                freeze(root / "_meta" / method / filename, read(path))
    # SepWater must keep exactly the measured dependency set. RoSteALS gains Slack.
    for filename in ("installed_packages.json", "environment_request.json"):
        path = parent / "_meta/sepwater" / filename
        if path.exists():
            freeze(root / "_meta/sepwater" / filename, read(path))
    for source_name in ("rosteals", "sepwater", "taming", "imagenetc"):
        path = parent / "_meta/sources" / (source_name + ".json")
        if path.exists():
            freeze(root / "_meta/sources" / path.name, read(path))
    copied = 0
    for method in METHODS:
        for folder in ("coco/records", "corruptions"):
            paths = sorted((parent / method / folder).glob("*.json"))
            for index, path in enumerate(paths):
                copied += import_record(parent, root, path.relative_to(parent).as_posix(), plan["identity"])
                if index % 25 == 0 or index + 1 == len(paths):
                    print(f"IMPORT {method}/{folder}: {index+1}/{len(paths)} records checked", flush=True)
    # Reuse only receipted model/source downloads; all destinations are in the new root.
    for directory in ("rosteals", "sepwater", "sources"):
        for receipt in sorted((parent / "_assets" / directory).glob("*.receipt.json")):
            name = receipt.name.removesuffix(".receipt.json")
            if directory == "sepwater" and name != "last.ckpt":
                continue  # f4.ckpt was audited, not selected for inference.
            if directory == "sources" and not name.startswith(("rosteals-", "sepwater-", "taming-", "imagenetc-")):
                continue
            metadata = read(receipt)
            target = root / "_assets" / directory / name
            copy_verified(receipt.with_name(name), target, metadata["sha256"])
            freeze(target.with_name(receipt.name), metadata)
            print(f"CACHE reused: {directory}/{name}", flush=True)
    print(f"Imported {copied} new records. Parent and DistSeal remain unchanged.", flush=True)
    return plan
