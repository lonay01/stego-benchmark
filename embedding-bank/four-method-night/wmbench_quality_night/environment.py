"""Restore build-only dependencies without changing the historical model lock."""

from contextlib import contextmanager
from pathlib import Path
import re

from wmbench.io import digest, read_json, sha256, write_json
from wmbench_full_payload.storage import EvidenceError, freeze


def pins(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise EvidenceError(f"Snapshot must contain only exact package pins: {line}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in result:
            raise EvidenceError(f"Duplicate package pin: {name}")
        result[name] = match[2]
    if not result:
        raise EvidenceError("Empty installed package snapshot")
    return result


def dependency_diff(expected, actual):
    before, after = pins(expected), pins(actual)
    return {name: {"expected": before.get(name), "actual": after.get(name)}
            for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}


@contextmanager
def constrain_build_dependencies(snapshot, python_version=None):
    """BasicSR's extra requirements were not in the original compile lock.

    Preserve that lock's hash, but constrain *every* subsequent install with the
    actual historical installed snapshot, including platformdirs and tb-nightly.
    The ordinary sync happens first so PyTorch exists before BasicSR is built.
    """
    from wmbench import environment
    snapshot = Path(snapshot)
    pins(snapshot.read_text("utf-8"))
    original = environment.command

    def constrained(args, cwd=None, env=None):
        values = list(map(str, args))
        if python_version and "venv" in values and "--python" in values:
            values[values.index("--python") + 1] = python_version
        if "pip" in values and "install" in values:
            values += ["--constraint", str(snapshot)]
        return original(values, cwd=cwd, env=env)

    environment.command = constrained
    try:
        yield
    finally:
        environment.command = original


def verify_dependencies(meta, prepared):
    expected = read_json(meta / "historical_dependency_identity.json")
    snapshot = meta / "historical_installed.freeze.txt"
    actual = Path(prepared["config"]).parent / "installed.freeze.txt"
    if sha256(snapshot) != expected["installed_sha256"]:
        raise EvidenceError("Historical package snapshot was modified")
    diff = dependency_diff(snapshot.read_text("utf-8"), actual.read_text("utf-8"))
    audit = {"expected": expected,
             "actual": {k: prepared["identity"][k] for k in expected}, "package_diff": diff}
    # Keep every observed audit, including failed attempts, without making a
    # transient setup failure an immutable blocker for a repaired environment.
    freeze(meta / "dependency_audits" / f"{digest(audit)}.json", audit)
    write_json(meta / "dependency_audit.json", audit)
    if diff or audit["actual"] != expected:
        raise EvidenceError(f"Historical environment not restored exactly: {diff or audit}; see dependency_audit.json")
    return audit
