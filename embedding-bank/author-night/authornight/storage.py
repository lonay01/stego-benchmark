"""Small atomic records; artifacts are committed before their completion record."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


class EvidenceError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def write(path, value):
    atomic(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise EvidenceError(f"Immutable evidence changed: {path}. Use a NEW output folder.")
    else:
        write(path, value)


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise EvidenceError(f"Path escapes run directory: {relative}")
    return path


def disjoint(*paths):
    paths = [Path(p).resolve() for p in paths]
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise EvidenceError(f"Paths must not overlap: {left}, {right}")


def cached(root, relative, identity, verify=True):
    path = inside(root, relative)
    if not path.exists():
        return None
    row = read(path)
    if row.get("identity") != identity:
        raise EvidenceError(f"Result identity mismatch: {path}")
    if row.get("record_sha256") != digest({k: v for k, v in row.items() if k != "record_sha256"}):
        raise EvidenceError(f"Modified result record: {path}")
    if verify:
        for name, expected in row.get("artifacts", {}).items():
            artifact = inside(root, name)
            if not artifact.is_file() or sha256(artifact) != expected:
                raise EvidenceError(f"Missing or changed result artifact: {artifact}")
    return row


def commit(root, relative, identity, data, artifacts=None):
    row = {**data, "identity": identity, "saved_unix": time.time(), "artifacts": {}}
    for path in artifacts or []:
        path = Path(path)
        name = path.relative_to(root).as_posix()
        row["artifacts"][name] = sha256(path)
    row["record_sha256"] = digest(row)
    write(inside(root, relative), row)
    return row


def image_zip(path, members):
    """A batch archive avoids tens of thousands of slow Drive file operations."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(members.items()):
            if name.startswith("/") or ".." in Path(name).parts:
                raise EvidenceError("Unsafe archive member")
            archive.writestr(name, data)
    atomic(path, stream.getvalue())


def extract_safe(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = inside(destination, entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise EvidenceError("Archive symlinks are not allowed")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                atomic(target, source.read(entry))
