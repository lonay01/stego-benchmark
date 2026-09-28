"""Small sealed records. Completed images are never inferred from filenames."""

from __future__ import annotations

import time
from pathlib import Path

from wmbench.io import read_json, sha256, write_json


class EvidenceError(ValueError):
    """Conflicting or damaged evidence must stop a resume, not trigger regeneration."""


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise EvidenceError(f"Path outside result root: {relative}")
    return target


def disjoint(*roots):
    roots = [Path(root).resolve() for root in roots]
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Source bank, output and scratch must be separate, non-nested folders")


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise EvidenceError(f"Frozen specification differs: {path}. Keep it and choose a new output folder.")
    if not path.exists():
        write_json(path, value)


def sealed(root, path, signature, data, artifacts=()):
    files = [{"path": Path(p).resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256(p)}
             for p in artifacts]
    row = {**data, "signature": signature, "artifacts": files, "finished_unix": time.time()}
    write_json(path, row)
    return row


def cached(root, path, signature=None):
    if not path.is_file():
        return None
    row = read_json(path)
    if not row.get("signature") or not isinstance(row.get("artifacts"), list):
        raise EvidenceError(f"Unsealed result: {path}")
    if row.get("status") == "saved" and not row["artifacts"]:
        raise EvidenceError(f"Saved image record has no sealed artifacts: {path}")
    if signature is not None and row["signature"] != signature:
        raise EvidenceError(f"Changed task identity: {path}")
    for item in row["artifacts"]:
        target = inside(root, item["path"])
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise EvidenceError(f"Damaged saved artifact: {target}; restore a backup, do not overwrite")
    return row


def retry_record(path):
    """Archive a technical error before one retry; measured BER is never retried."""
    if path.exists():
        history = path.parent / "technical_error_history"
        history.mkdir(exist_ok=True)
        path.replace(history / f"{time.time_ns()}_{path.name}")
