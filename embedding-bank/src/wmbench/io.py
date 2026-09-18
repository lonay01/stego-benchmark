"""Content-addressed, atomic local persistence and safe archive handling."""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import math
import os
import tempfile
import urllib.request
import urllib.error
import http.client
import time
import zipfile
from pathlib import Path
from typing import Any


class StorageUnavailable(OSError):
    """Transient storage failure persisted; never treat it as a missing result."""


def retry_read(operation, path):
    """Four attempts for transport errors only; no retries of writes or corruption."""
    transient = {errno.EIO, errno.ETIMEDOUT, errno.EAGAIN, getattr(errno, "ESTALE", 116)}
    for attempt in range(4):
        try:
            return operation()
        except OSError as error:
            if error.errno not in transient:
                raise
            if attempt == 3:
                raise StorageUnavailable(error.errno, f"Storage unavailable after 4 reads: {path}") from error
            delay = 2 ** (attempt + 1)
            print(f"STORAGE RETRY {attempt + 1}/3 in {delay}s: {path}: {error}", flush=True)
            time.sleep(delay)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: str | Path) -> str:
    return retry_read(lambda: _sha256_once(path), path)


def _sha256_once(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_json(value: Any) -> Any:
    """JSON has no NaN/Infinity. Preserve infinities explicitly, never as fake zeroes."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if hasattr(value, "item"):
        return finite_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value == math.inf else "-Infinity" if value == -math.inf else None
    return value


def atomic_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    atomic_bytes(
        path,
        json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(retry_read(lambda: Path(path).read_text(encoding="utf-8-sig"), path))


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    import io

    fields = fields or sorted({key for row in rows for key in row})
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in finite_json(row).items()
            }
        )
    atomic_bytes(path, text.getvalue().encode("utf-8"))


def download(url: str, target: Path, expected_sha256: str | None = None) -> None:
    """Retry transport errors; resume only with a matching server validator.

    Truncated HTTP bodies never become cache entries. TLS verification stays on.
    Controller stage limits still bound slow servers and repeated downloads.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (not expected_sha256 or sha256(target) == expected_sha256):
        return
    part = target.with_name(target.name + ".part")
    metadata = target.with_name(target.name + ".part.json")
    for attempt in range(3):
        offset = 0
        headers = {"User-Agent": "wmbench/0.1", "Accept-Encoding": "identity"}
        if part.exists() and metadata.exists():
            previous = read_json(metadata)
            if previous.get("url") == url and previous.get("validator"):
                offset = part.stat().st_size
                headers.update(Range=f"bytes={offset}-", **{"If-Range": previous["validator"]})
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = response.status == 206 and offset > 0
                if response.status == 206 and not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid Content-Range in resumed download")
                validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                if validator and validator.startswith("W/"):
                    validator = response.headers.get("Last-Modified")
                write_json(metadata, {"url": url, "validator": validator})
                expected = response.headers.get("Content-Length")
                received = 0
                last_print = time.monotonic()
                with part.open("ab" if resumed else "wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                        received += len(block)
                        if time.monotonic() - last_print > 30:
                            print(
                                f"Download {target.name}: {(received + (offset if resumed else 0)) / 1024**2:.0f} MiB",
                                flush=True,
                            )
                            last_print = time.monotonic()
                if expected is not None and received != int(expected):
                    raise http.client.IncompleteRead(b"", int(expected) - received)
            break
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            ConnectionError,
            TimeoutError,
        ) as error:
            status = getattr(error, "code", None)
            if status is not None and status not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            print(
                f"Download retry {attempt + 1}/2: {target.name}: {type(error).__name__}", flush=True
            )
            time.sleep(2 ** (attempt + 1))
    if expected_sha256 and sha256(part) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {url}")
    os.replace(part, target)
    metadata.unlink(missing_ok=True)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is not allowed: {member.filename}")
        handle.extractall(destination)
