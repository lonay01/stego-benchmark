"""Bounded Drive/FUSE retries for control files, not benchmark measurements.

The frozen controller still owns acquisition, token checks and release. A missing
lease is never recreated by a reader and a different token is never accepted.
"""

import errno
import json
import os
from pathlib import Path
import time
import uuid


TRANSIENT = {errno.ENOENT, errno.EIO, errno.EAGAIN, errno.ETIMEDOUT,
             getattr(errno, "ESTALE", 116), getattr(errno, "ENOTCONN", 107),
             errno.ECONNRESET}


def read_lease(path, attempts=8, delay=1.0):
    """Retry failed reads with at most 39 seconds of backoff by default.

    This bounds backoff, not a kernel-level stalled FUSE read. No cached ownership
    is substituted. Persistent absence/corruption and permission errors fail closed.
    """
    if attempts < 1 or delay < 0:
        raise ValueError("Invalid control-file retry settings")
    path = Path(path)
    for attempt in range(attempts):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            if isinstance(error, OSError) and error.errno not in TRANSIENT:
                raise
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Control file unavailable after {attempts} reads: {path}. "
                    "Stop and reconnect Drive; retain results. Ownership was not bypassed."
                ) from error
            seconds = min(delay * 2 ** attempt, 8.0)
            try:
                print(f"CONTROL READ RETRY {attempt + 1}/{attempts - 1} "
                      f"in {seconds:g}s: {path.name}: {type(error).__name__}", flush=True)
            except OSError:
                pass  # The diagnostic log may share the unavailable mount.
            time.sleep(seconds)
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Control file is not a JSON object: {path}")
        return value


def install():
    """Replace only the frozen lease reader in this controller process."""
    from wmbench import colab_canary
    colab_canary.read_lease = read_lease


def storage_preflight(settings):
    """Refuse to start against a disconnected mount or absent recovery folders.

    The only write is an exclusive, uniquely named probe; no result, bank file,
    ownership file or checkpoint is changed. Failed probes remain for diagnosis.
    """
    bank, root = Path(settings["bank"]), Path(settings["output"])
    drive = Path("/content/drive")
    for path in (bank, root):
        if path.is_relative_to(drive) and not os.path.ismount(drive):
            raise RuntimeError("Google Drive is not mounted. Connect Drive before resuming; "
                               "do not create replacement local result folders.")
    reports = root / "_reports"
    if not bank.is_dir() or not reports.is_dir():
        raise RuntimeError("Existing bank/results folders are unavailable. Check the Drive "
                           "account and mount; this recovery release will not create a new bank.")
    token = uuid.uuid4().hex
    probe = reports / (".storage-probe_" + token + ".json")
    value = {"token": token, "purpose": "digital-resume-storage-preflight"}
    with probe.open("x", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if read_lease(probe) != value:
        raise RuntimeError(f"Drive probe readback differs; retained for diagnosis: {probe}")
    probe.unlink()  # Only our own just-created and verified probe.
    print("STORAGE PREFLIGHT: existing bank/results visible; write/readback passed", flush=True)
