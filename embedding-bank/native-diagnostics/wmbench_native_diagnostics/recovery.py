"""Narrow, recorded compatibility for the 2026-09-22 source-inspection bug.

This is not a general force-resume option. The original identity stays immutable;
only the named diagnostic audit/controller/provenance patch may reuse its keys.
"""

from wmbench.io import digest, read_json, sha256
from wmbench_full_payload.storage import EvidenceError, freeze

PREVIOUS_ARCHIVE_SHA256 = "ba01ad3bd6c13131c8491031a5986cd35af0e987f9a37824604cd1b6506bbfc4"
PREVIOUS_SOURCE_SHA256 = "9475f5da44fe8824dcd7f1547b5d0748b550880abdcdc0670677d3883aaaad51"
UNCHANGED_SOURCE_SHA256 = "9421aa5122edd7f20b6b753ccb504f68c3220d86dac0b6e92144454f36e6a4f8"
PATCH_FILES = frozenset({
    "wmbench_native_diagnostics/probes.py",     # Source lookup only; tensor operations unchanged.
    "wmbench_native_diagnostics/runner.py",     # Recorded compatibility and status history.
    "wmbench_native_diagnostics/worker.py",     # Execution provenance only.
    "wmbench_native_diagnostics/recovery.py",   # This narrowly scoped compatibility check.
})


def source_manifest(project):
    return {p.relative_to(project / "src").as_posix(): sha256(p)
            for p in sorted((project / "src").rglob("*.py"))}


def freeze_run_identity(project, root, proposed):
    """Preserve saved keys only for the recognized audit-only predecessor."""
    files = source_manifest(project)
    current = digest(files)
    if proposed["source_code"] != current:
        raise EvidenceError("Proposed source identity does not match deployed code")
    path = root / "diagnostic_identity.json"
    previous = read_json(path) if path.exists() else None
    if previous is None or previous == proposed:
        freeze(path, proposed)
        return proposed
    old_spec = {k: v for k, v in previous.items() if k != "source_code"}
    new_spec = {k: v for k, v in proposed.items() if k != "source_code"}
    if old_spec != new_spec:
        raise EvidenceError("Historical trial, model selection or diagnostic policy changed; cannot reuse this run")
    if previous.get("source_code") != PREVIOUS_SOURCE_SHA256:
        raise EvidenceError("Unrecognized diagnostic predecessor; refusing automatic source migration")
    common = {name: value for name, value in files.items() if name not in PATCH_FILES}
    if digest(common) != UNCHANGED_SOURCE_SHA256:
        raise EvidenceError("Inference/scoring/storage code changed outside the reviewed audit patch")
    receipt = {"kind": "source-inspection-recovery-2026-09-26",
               "previous_archive_sha256": PREVIOUS_ARCHIVE_SHA256,
               "preserved_run_identity": digest(previous),
               "previous_source_code": previous["source_code"], "current_source_code": current,
               "unchanged_source_digest": digest(common),
               "patched_source_files": {n: files[n] for n in sorted(PATCH_FILES)},
               "reason": "Handle transient WIBE classes during source inspection; inference and scoring unchanged"}
    freeze(root / "_meta/code_revisions" / f"{current}.json", receipt)
    print("RECOVERY: recognized audit-only update; preserving historical run and stage signatures.", flush=True)
    return previous
