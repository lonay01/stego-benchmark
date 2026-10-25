"""Build unchanged method reports on SSD; publish verified copies to durable storage.

No streaming gzip file or rename operation is performed on the Drive mount.
An immutable staged version and a backup of the previous tables precede each
publication. A failed copy can resume from that version without recomputing metrics.
The controller lease is held by the caller throughout this operation.
"""

from concurrent.futures import ThreadPoolExecutor
import gzip
from pathlib import Path
import tempfile
import time

from wmbench.io import read_json, sha256, write_json
from wmbench_digital import report
from wmbench_digital.spec import digest
from wmbench_digital.storage import complete, result_path
from wmbench_full_payload.storage import EvidenceError
from .cache import fingerprint

TABLES = ("counts.json", "reference_and_clean.csv", "attack_summary.csv",
          "macro_summary.csv", "case_levels.json.gz", "per_attack.csv.gz")


def copy_verified(source, destination, expected, guard, attempts=3):
    """Retry I/O only, always with the same verified bytes; never rerun a model."""
    source, destination = Path(source), Path(destination)
    if sha256(source) != expected:
        raise EvidenceError(f"Report source checksum mismatch: {source}")
    payload = source.read_bytes()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != expected:
        raise EvidenceError("Report source changed while reading")
    for attempt in range(attempts):
        guard()
        try:
            if destination.is_file() and sha256(destination) == expected:
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately no replace/rename on Drive: the source remains on SSD
            # or in an immutable, checked report version until readback succeeds.
            with destination.open("wb") as handle:
                handle.write(payload)
                handle.flush()
            if sha256(destination) != expected:
                raise OSError(f"Report copy readback mismatch: {destination}")
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            print(f"REPORT COPY RETRY {attempt+1}/{attempts-1}: {destination.name}", flush=True)
            time.sleep(2 ** attempt)


def json_copy(value, target, local, guard):
    source = local / (target.name + ".publication.json")
    write_json(source, value)
    copy_verified(source, target, sha256(source), guard)


def validate_tables(folder, expected_count):
    for name in TABLES:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise EvidenceError(f"Report table missing/empty: {name}")
        if name.endswith(".gz"):
            # Consume to EOF to validate the gzip checksum, not only its header.
            with gzip.open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
    counts = read_json(folder / "counts.json")
    if counts.get("carriers") != expected_count or counts.get("complete_carriers") != expected_count:
        raise EvidenceError("Refusing to publish an incomplete method report")
    return counts, {name: sha256(folder / name) for name in TABLES}


def evidence(root, name, cases, cache):
    """Bind a staged report to the same inputs and validated result inventory."""
    rows = {}
    for case in cases:
        path = result_path(root, case)
        if not complete(cache.read(path, case)):
            raise EvidenceError("Report needs complete decoded evidence")
        rows[path.name] = cache.stamps.get(path.name)
    quality = root / name / "quality"
    inventory = {p.name: fingerprint(p) for p in quality.iterdir()} if quality.exists() else {}
    return dict(case_keys=[c["key"] for c in cases], records=rows,
                quality={c["id"] + ".json": inventory.get(c["id"] + ".json") for c in cases},
                selection=read_json(root / "selection.json"),
                coverage_sha256=sha256(root / name / "coverage.json"),
                report_code_sha256=sha256(Path(report.__file__)))


def publish(root, name, version, manifest, local, guard):
    """Back up all old tables before the first canonical table can change."""
    if set(manifest["hashes"]) != set(TABLES):
        raise EvidenceError("Unexpected staged report file set")
    for filename, expected in manifest["hashes"].items():
        if sha256(version / filename) != expected:
            raise EvidenceError(f"Staged report changed: {filename}")
    tables = root / name / "tables"
    archive = version / "previous_tables"
    backup_receipt = archive / "manifest.json"
    if backup_receipt.exists():
        previous = read_json(backup_receipt)
        for filename, expected in previous.items():
            if filename not in TABLES or sha256(archive / filename) != expected:
                raise EvidenceError("Previous report backup is damaged")
    else:
        previous = {}
        for filename in TABLES:
            target = tables / filename
            if target.is_file():
                previous[filename] = sha256(target)
                copy_verified(target, archive / filename, previous[filename], guard)
        json_copy(previous, backup_receipt, local, guard)
    for filename, expected in manifest["hashes"].items():
        copy_verified(version / filename, tables / filename, expected, guard)
    for filename, expected in manifest["hashes"].items():
        if sha256(tables / filename) != expected:
            raise EvidenceError("Report set changed before publication commit")
    receipt = dict(manifest, state="published", previous_tables=str(archive),
                   published_unix=time.time())
    json_copy(receipt, root / name / "_fast/report-publication.json", local, guard)
    print(f"REPORT PUBLISHED {name}: {len(TABLES)} SHA256-checked tables; previous tables archived", flush=True)
    return read_json(tables / "counts.json")


def method_report(root, name, scratch, cache, cases):
    root, scratch = Path(root).resolve(), Path(scratch).resolve()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid method directory")
    if scratch.is_relative_to(root) or root.is_relative_to(scratch):
        raise ValueError("Report SSD staging must be outside durable results")
    selection_hash = sha256(root / "selection.json")
    def guard():
        if sha256(root / "selection.json") != selection_hash:
            raise EvidenceError("Durable results unavailable or selection changed")
    expected = evidence(root, name, cases, cache)
    scratch.mkdir(parents=True, exist_ok=True)
    local = Path(tempfile.mkdtemp(prefix=f"report-{name}-", dir=scratch))
    staged = root / name / "_fast/report-staged.json"
    if staged.exists():
        manifest = read_json(staged)
        if manifest.get("evidence_key") == digest(expected):
            version = root / name / "_report_versions" / manifest["version"]
            if version.parent != (root / name / "_report_versions") or version.is_symlink():
                raise EvidenceError("Unsafe staged report version")
            print(f"REPORT RESUME {name}: reuse verified staged tables; no metric recomputation", flush=True)
            return publish(root, name, version, manifest, local, guard)
    print(f"REPORT SSD {name}: staging inputs for {len(cases)} carriers at {local}", flush=True)
    write_json(local / "selection.json", expected["selection"])
    write_json(local / name / "cases.json", cases)
    write_json(local / name / "coverage.json", read_json(root / name / "coverage.json"))
    def stage_quality(case):
        source = root / name / "quality" / (case["id"] + ".json")
        stamp = expected["quality"][source.name]
        if stamp is not None:
            value = read_json(source)
            if value.get("case_key") != case["key"] or fingerprint(source) != stamp:
                raise EvidenceError("Quality evidence changed during report staging")
            write_json(local / name / "quality" / source.name, value)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(stage_quality, cases), 1):
            if index % 200 == 0 or index == len(cases):
                print(f"REPORT INPUTS {name}: {index}/{len(cases)}", flush=True)
    previous = report.read_result
    report.read_result = lambda path, case=None: cache.read(root / name / "records" / Path(path).name, case)
    try:
        # Frozen numerical implementation; only the filesystem root is different.
        report.method_report(local, name)
    finally:
        report.read_result = previous
    counts, hashes = validate_tables(local / name / "tables", len(cases))
    if evidence(root, name, cases, cache) != expected:
        raise EvidenceError("Report inputs changed; refusing publication")
    guard()
    version_name = str(time.time_ns())
    version = root / name / "_report_versions" / version_name
    manifest = dict(version=version_name, evidence_key=digest(expected), hashes=hashes,
                    state="staged", counts=counts, created_unix=time.time())
    for filename, checksum in hashes.items():
        copy_verified(local / name / "tables" / filename, version / filename, checksum, guard)
    json_copy(manifest, version / "manifest.json", local, guard)
    json_copy(manifest, staged, local, guard)
    return publish(root, name, version, manifest, local, guard)
