"""Restore the exact saved VideoSeal checkpoint before the frozen prepare step.

Only the checkpoint transport changes. Model code, options, dependencies and
all original evidence checks are delegated to the unchanged release. The SDXL
identity overlay remains in place, including its worker propagation.
"""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from unittest.mock import patch


VERSION = "videoseal-download-v1"
ENTRYPOINT = "neural_videoseal_download_fix"
HASH_ENV = "WMBENCH_VIDEOSEAL_DOWNLOAD_FIX_SHA256"
BASE_ZIP_SHA256 = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"
IDENTITY_FIX_SHA256 = "94bb725be25d6f550284c75ee76051674b92a5dddd8922eda956b8895aefa383"
WIBE_REVISION = "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
CHECKPOINT_KEY = "model_files/videoseal/y_256b_img.pth"
CHECKPOINT_SHA256 = "3d2ff2523d2a89e3532c6dfdcf693098799326e9b3e74c185e815e9baa8340a3"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth"
MIRROR_URL = ("https://huggingface.co/Svetozar1993/videoseal-1.0/resolve/"
    "35116c2e949486b08c2632df4ad8f73f9b177b55/y_256b_img.pth")
CHUNK_BYTES = 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, expected):
    """Never replace an existing file whose provenance cannot be established."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Checkpoint must not be a symlink: {path}")
    if not path.exists():
        return False
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch; file preserved: {path}")
    return True


def publish(source, target, expected):
    """Verified copy with a same-directory temporary file and readback check."""
    source, target = Path(source), Path(target)
    if verified(target, expected):
        return
    if not verified(source, expected):
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".videoseal-", suffix=".part", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        verified(temporary, expected)
        # The original controller lease serializes jobs. Recheck nevertheless;
        # a conflicting pre-existing file must never be overwritten.
        if not verified(target, expected):
            os.replace(temporary, target)
        verified(target, expected)
    finally:
        temporary.unlink(missing_ok=True)  # Only this call's private partial file.


def fetch_checkpoint(target, expected, deadline, attempts=3):
    """Try Meta then a pinned byte-identical mirror, always enforcing bank SHA-256."""
    target = Path(target)
    if verified(target, expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, attempts + 1):
        url = CHECKPOINT_URL if attempt == 1 else MIRROR_URL
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Session budget reached before VideoSeal download")
        descriptor, name = tempfile.mkstemp(prefix=".videoseal-download-", suffix=".part", dir=target.parent)
        temporary = Path(name)
        try:
            print(f"VIDEOSEAL DOWNLOAD {attempt}/{attempts}: {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "wmbench-videoseal-recovery/1"})
            with os.fdopen(descriptor, "wb") as output:
                with urllib.request.urlopen(request, timeout=min(60, remaining)) as response:
                    total = response.headers.get("Content-Length")
                    total = int(total) if total is not None else None
                    count, reported = 0, 0
                    while True:
                        if time.time() >= deadline:
                            raise TimeoutError("Session budget reached during VideoSeal download")
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        output.write(chunk)
                        count += len(chunk)
                        if count - reported >= 16 * CHUNK_BYTES:
                            print(f"VIDEOSEAL DOWNLOAD: {count / CHUNK_BYTES:.0f} MiB received", flush=True)
                            reported = count
                    if total is not None and count != total:
                        raise OSError(f"Truncated checkpoint: {count} of {total} bytes")
                output.flush()
                os.fsync(output.fileno())
            # A wrong complete response is not retried or accepted as new weights.
            verified(temporary, expected)
            publish(temporary, target, expected)
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            last_error = error
            print(f"VIDEOSEAL download interrupted ({type(error).__name__}); results are unchanged.", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(
        "VideoSeal checkpoint download failed at Meta and the pinned mirror. Rerun with the same output folder. "
        f"Alternatively place the exact SHA-256 {expected} checkpoint at {target}. "
        "No weights or results were substituted."
    ) from last_error


def expected_checkpoint(job):
    """Require the original saved bank, not a newly chosen model/checkpoint."""
    path = Path(job["bank"]) / "videoseal/_meta/model_identity.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("registry") != job["model"]
            or job["model"].get("source", {}).get("revision") != WIBE_REVISION):
        raise ValueError("VideoSeal registry/source differs from the saved bank; recovery refused")
    if saved.get("weights", {}).get(CHECKPOINT_KEY) != CHECKPOINT_SHA256:
        raise ValueError("Saved bank does not use the audited VideoSeal checkpoint; recovery refused")
    return path


def ensure_videoseal_checkpoint(job):
    """Preseed WIBE's existing required-file path; do not patch its adapter."""
    if job.get("model", {}).get("id") != "videoseal":
        return None
    session = str(job["session"])
    if not session.isdecimal():
        raise ValueError("Invalid recovery session identifier")
    identity_path = expected_checkpoint(job)
    root = Path(job["results"]) / "_repairs" / VERSION
    cache = root / "assets" / CHECKPOINT_SHA256 / "y_256b_img.pth"
    local = Path(job["slot"]) / "assets/videoseal" / CHECKPOINT_KEY
    # Check both existing locations before doing any copy/download.
    has_local = verified(local, CHECKPOINT_SHA256)
    has_cache = verified(cache, CHECKPOINT_SHA256)
    source = "drive-cache" if has_cache else "existing-slot" if has_local else "verified-download"
    if not has_cache:
        if has_local:
            publish(local, cache, CHECKPOINT_SHA256)
        else:
            fetch_checkpoint(cache, CHECKPOINT_SHA256, job["deadline"])
    publish(cache, local, CHECKPOINT_SHA256)
    from wmbench_full_payload.storage import freeze

    # This audit is outside model assets and outside scientific identities.
    freeze(root / "receipts" / f"{session}.json", dict(
        repair=VERSION, repair_sha256=file_sha256(__file__),
        base_zip_sha256=BASE_ZIP_SHA256, identity_fix_sha256=IDENTITY_FIX_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_key=CHECKPOINT_KEY,
        permitted_download_urls=[CHECKPOINT_URL, MIRROR_URL], bank_identity_sha256=file_sha256(identity_path),
        checkpoint_size=local.stat().st_size,
        scope="transport only; original model/checkpoint/dependency/Clean checks remain enabled",
    ))
    print(f"VIDEOSEAL CHECKPOINT VERIFIED ({source}): {CHECKPOINT_SHA256}", flush=True)
    print(f"VIDEOSEAL durable cache: {cache}", flush=True)
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate", "prepare", "quality",
        "geometry", "controls", "generate", "decode", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    actual = file_sha256(__file__)
    if os.environ.get(HASH_ENV) != actual:
        raise RuntimeError("VideoSeal recovery checksum absent or changed; rerun the repair cell")
    import neural_sdxl_identity_fix as identity_fix

    if (identity_fix.module_sha256() != IDENTITY_FIX_SHA256
            or os.environ.get(identity_fix.HASH_ENV) != IDENTITY_FIX_SHA256):
        raise RuntimeError("Original identity-recovery module/hash changed; recovery refused")
    if args.action == "prepare":
        ensure_videoseal_checkpoint(json.loads(args.settings.read_text(encoding="utf-8")))
    # Existing forwarding adds its own checksum; the new checksum must also be
    # explicit in decoder environments, which may not inherit os.environ.
    original_redirect = identity_fix.redirect_child

    def redirect(command, env, expected_sha256):
        forwarded, child_env = original_redirect(command, env, expected_sha256)
        if forwarded is not command:
            child_env = {**child_env, HASH_ENV: actual}
        return forwarded, child_env

    with patch.object(identity_fix, "ENTRYPOINT", ENTRYPOINT), \
            patch.object(identity_fix, "redirect_child", redirect):
        identity_fix.main()


if __name__ == "__main__":
    main()
