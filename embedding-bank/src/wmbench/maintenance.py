"""Explicit, fail-closed continuation of the audited September 8 release.

This is not a generic 'ignore fingerprints' switch. The release certificate pins
both source inventories. Protocol inputs and model registry must match. Runtime
changes require a separately named, narrow user authorization and visible audit.
Old records/protocol remain immutable; every new execution carries an amendment.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from .io import digest, read_json, sha256, write_json

AMENDMENT = "2026-09-10-storage-recovery-v6"
V5_AMENDMENT = "2026-09-10-explicit-runtime-migration-v5"
V5_CODE = "846b38757bc785c3aa82c9dd1c35d4844bde09c1668fb1a30ed0b4b6752696dd"
V5_CERTIFICATE = "9267d15ba9ec7c9ac147892d9fa1c8dfb6918dfd18587aec4cc796b80f4ce6c4"
V4_AMENDMENT = "2026-09-09-archive-packaging-v4"
V4_CODE = "aeda5a00f9733c729a78641e7d4801cf1722523a2643eb276c010a7a3ddfb544"
V4_CERTIFICATE = "114cdd59f43c13794ddfa0f8e04039cbafbf612866f3ffe529ba49cf17adc1a0"
RUNTIME_MIGRATION = "colab-20260910-py310-glibc239"


def runtime_transition(before: dict, after: dict, authorization: str | None = None) -> dict:
    """Permit the explicitly approved host update, NOT numerical equivalence.

    All unlisted fields (GPU, driver, packages, torch, CUDA, architecture) stay
    strict. Worker Python may be unchanged; only the observed patch pair can
    change. Original identities are never edited by this function.
    """
    changes = {
        key: {"previous": before.get(key), "actual": after.get(key)}
        for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)
    }
    allowed = {
        "python": ("3.10.12", "3.10.21"),
        "platform": ("Linux-6.6.122+-x86_64-with-glibc2.35", "Linux-6.6.122+-x86_64-with-glibc2.39"),
    }
    if changes and (authorization != RUNTIME_MIGRATION or any(
        key not in allowed or (value["previous"], value["actual"]) != allowed[key]
        for key, value in changes.items()
    )):
        raise ValueError(f"Unapproved runtime change: {changes}. Saved results are not deleted.")
    return changes
PARENT_AMENDMENT = "2026-09-09-trustmark-bitmark-ip2p"
PARENT_CODE = "3978eb7d7c736c33f8b8233eab9705eb984894302f9b542ab817aa6cea790851"
PARENT_CERTIFICATE = "461b73e4cd81ada35fb928addd7b495409729380e812e0420651cb7ff3039399"
V2_AMENDMENT = "2026-09-09-wam-packaging-v2"
V2_CODE = "8259b1c83dc0c92efd6265f4226a5659fff750d6118eecbab6e360edfc1c430e"
V2_CERTIFICATE = "68549cba87bc44b68e9febae08811efb7f51afcb99e4566fae7211437fdf02fd"
V3_AMENDMENT = "2026-09-09-videoseal-packaging-v3"
V3_CODE = "9fb9c7509544500c4ca916ff584857412edad2f65a3fda3ea1c4a124b3924b89"
V3_CERTIFICATE = "8ed35a888cc033f7dc75987210ccb10361a23d00d6669b60b7ab018db468b1d6"
BASE_CODE = "b112563db9c8cfbd3e5233f6b505d43b0da098d58197bb4c91114d74df053863"

# Exact archive/checkpoint mappings from the pinned WIBE wrappers and saved
# dependency manifests. Unknown archives, all other files, and every model/env
# field remain strict. These are transport ZIPs, not inference inputs.
ARCHIVE_RULES = {
    "trustmark_q": ("trustmark", (
        "encoder_Q.ckpt", "decoder_Q.ckpt", "trustmark_Q.yaml",
    )),
    "wam": ("watermark_anything", ("wam_mit.pth", "LICENSE", "README.md")),
    "videoseal": ("videoseal", ("y_256b_img.pth",)),
    "pixelseal": ("pixelseal", ("pixel_seal_checkpoint.pth",)),
    "chunkyseal": ("chunkyseal", ("checkpoint.pth",)),
    "vine_b": ("vine", (
        "VINE-B-Enc/model.safetensors", "VINE-B-Enc/config.json",
        "VINE-B-Dec/model.safetensors", "VINE-B-Dec/config.json",
    )),
    "rosteals": ("rosteals", ("VQ4_mir_inference.yaml", "epoch=000017-step=000449999.ckpt")),
    "invismark": ("invismark", ("invismark.ckpt",)),
    "mbrs": ("mbrs", (
        "results/MBRS_256_m256/models/EC_42.pth",
        "results/MBRS_256_m256/test_JpegTest(50)_s1_params.json",
        "results/MBRS_Diffusion_128_m30/models/EC_114.pth",
        "results/MBRS_Diffusion_128_m30/test_Crop(0.19,0.19)_s1_params.json",
    )),
    "cin": ("cin", ("cinNet&nsmNet.pth", "opt.yml")),
}
INVISMARK_CHECKPOINT = "409f4259a88865ccc0cb75884810e6a6435a8235050333c1b04c8d539a54d9bb"
INVISMARK_EVENT = re.compile(r"runs2/events\.out\.tfevents\.[0-9]+\.[A-Za-z0-9_-]+\.[0-9]+\.[0-9]+")


def _invismark_event_outputs(before: dict, after: dict) -> bool:
    """Separate one pinned model's write-only TensorBoard events from inputs.

    WIBE's pinned InvisMark constructor creates SummaryWriter(config.log_dir).
    Its _encode/_decode paths never read these events. The deployed checkpoint
    stores runs2 as log_dir. No other log path, checkpoint or cache is exempted.
    """
    registry = before.get("registry", {})
    if (
        registry.get("id") != "invismark"
        or registry.get("adapter") != "wmbench.adapters.legacy:WibeAdapter"
        or registry.get("source", {}).get("url") != "https://github.com/ispras/wibe.git"
        or registry.get("source", {}).get("revision") != "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
        or registry.get("options", {}).get("wrapper")
        != "wibench.algorithms.invismark.wrapper:InvisMarkWrapper"
        or "submodules/invismark" not in registry.get("source", {}).get("submodules", [])
        or before.get("weights", {}).get("model_files/invismark/invismark.ckpt") != INVISMARK_CHECKPOINT
        or after.get("weights", {}).get("model_files/invismark/invismark.ckpt") != INVISMARK_CHECKPOINT
    ):
        return False
    events = []
    for identity in (before, after):
        weights = identity.get("weights", {})
        selected = {key: weights[key] for key in weights if INVISMARK_EVENT.fullmatch(key)}
        events.append(selected)
        for key in selected:
            del weights[key]
    return events[0] != events[1]


def source_inventory(project: Path) -> dict:
    return {
        p.relative_to(project).as_posix(): sha256(p)
        for p in sorted((project / "src").rglob("*.py"))
    }


def resume_identity(project: Path, save: Path, current: dict, run_id: str | None,
                    runtime_migration: str | None = None):
    """Return immutable protocol identity plus optional audited code amendment."""
    if not run_id:
        return current, None
    if len(run_id) != 20 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid explicit resume_run_id")
    root = save / "runs" / run_id
    old = read_json(root / "protocol.json")
    if digest(old)[:20] != run_id:
        raise ValueError("Stored run identity does not match directory")
    if old == current:
        return old, None
    certificate = read_json(project / "configs/maintenance_20260909.json")
    if certificate["amendment"] != AMENDMENT or certificate["base_code"] != BASE_CODE:
        raise ValueError("Unknown maintenance certificate")
    if old["code"] != BASE_CODE or source_inventory(project) != certificate["new_source_files"]:
        raise ValueError("Source is not an approved maintenance release")
    execution_changes = runtime_transition(old.get("execution", {}), current.get("execution", {}), runtime_migration)
    keys = (set(old) | set(current)) - {"code", "project_root", "execution"}
    changed = sorted(k for k in keys if old.get(k) != current.get(k))
    if changed:
        raise ValueError(
            f"Cannot resume on changed protocol/runtime: {changed}. Keep the old run; use a new run on different hardware."
        )
    audit_path = root / "amendments" / (AMENDMENT + ".json")
    parent_path = root / "amendments" / (PARENT_AMENDMENT + ".json")
    parent = None
    if parent_path.exists():
        parent = read_json(parent_path)
        parent_identity = parent.get("current_identity", {})
        if (
            parent.get("amendment") != PARENT_AMENDMENT
            or parent.get("previous_code") != BASE_CODE
            or parent.get("current_code") != PARENT_CODE
            or parent.get("certificate_sha256") != PARENT_CERTIFICATE
            or parent_identity.get("code") != PARENT_CODE
            or any(old.get(k) != parent_identity.get(k)
                   for k in (set(old) | set(parent_identity)) - {"code", "project_root"})
        ):
            raise ValueError("Incompatible predecessor amendment")
    v2_path = root / "amendments" / (V2_AMENDMENT + ".json")
    v2 = None
    if v2_path.exists():
        v2 = read_json(v2_path)
        v2_identity = v2.get("current_identity", {})
        if (
            v2.get("amendment") != V2_AMENDMENT
            or v2.get("previous_code") != BASE_CODE
            or v2.get("current_code") != V2_CODE
            or v2.get("certificate_sha256") != V2_CERTIFICATE
            or v2_identity.get("code") != V2_CODE
            or any(old.get(k) != v2_identity.get(k)
                   for k in (set(old) | set(v2_identity)) - {"code", "project_root"})
        ):
            raise ValueError("Incompatible v2 predecessor amendment")
        # Verify the whole deployed chain, not merely the presence of v2.
        linked_parent = v2.get("parent_amendment_sha256")
        if linked_parent is not None or v2.get("compatible_previous_codes"):
            if (
                parent is None
                or linked_parent != sha256(parent_path)
                or v2.get("compatible_previous_codes") != [PARENT_CODE]
            ):
                raise ValueError("Incompatible v2 predecessor chain")
    v3_path = root / "amendments" / (V3_AMENDMENT + ".json")
    v3 = None
    if v3_path.exists():
        v3 = read_json(v3_path)
        v3_identity = v3.get("current_identity", {})
        if (
            v3.get("amendment") != V3_AMENDMENT
            or v3.get("previous_code") != BASE_CODE
            or v3.get("current_code") != V3_CODE
            or v3.get("certificate_sha256") != V3_CERTIFICATE
            or v3_identity.get("code") != V3_CODE
            or any(old.get(k) != v3_identity.get(k)
                   for k in (set(old) | set(v3_identity)) - {"code", "project_root"})
        ):
            raise ValueError("Incompatible v3 predecessor amendment")
        expected_codes = []
        for field, previous_audit, path, code in (
            ("parent_amendment_sha256", parent, parent_path, PARENT_CODE),
            ("v2_amendment_sha256", v2, v2_path, V2_CODE),
        ):
            if field in v3:
                if previous_audit is None or v3[field] != sha256(path):
                    raise ValueError("Incompatible v3 predecessor chain")
                expected_codes.append(code)
        if v3.get("compatible_previous_codes", []) != expected_codes:
            raise ValueError("Incompatible v3 predecessor codes")
    v4_path = root / "amendments" / (V4_AMENDMENT + ".json")
    v4 = None
    if v4_path.exists():
        v4 = read_json(v4_path)
        v4_identity = v4.get("current_identity", {})
        if (
            v4.get("amendment") != V4_AMENDMENT
            or v4.get("previous_code") != BASE_CODE
            or v4.get("current_code") != V4_CODE
            or v4.get("certificate_sha256") != V4_CERTIFICATE
            or v4_identity.get("code") != V4_CODE
            or any(old.get(k) != v4_identity.get(k)
                   for k in (set(old) | set(v4_identity)) - {"code", "project_root"})
        ):
            raise ValueError("Incompatible v4 predecessor amendment")
        expected_codes = []
        for field, predecessor, path, code in (
            ("parent_amendment_sha256", parent, parent_path, PARENT_CODE),
            ("v2_amendment_sha256", v2, v2_path, V2_CODE),
            ("v3_amendment_sha256", v3, v3_path, V3_CODE),
        ):
            if field in v4:
                if predecessor is None or v4[field] != sha256(path):
                    raise ValueError("Incompatible v4 predecessor chain")
                expected_codes.append(code)
        if v4.get("compatible_previous_codes", []) != expected_codes:
            raise ValueError("Incompatible v4 predecessor codes")
    audit = {
        "amendment": AMENDMENT,
        "previous_code": old["code"],
        "current_code": current["code"],
        "current_identity": current,
        "certificate_sha256": sha256(project / "configs/maintenance_20260909.json"),
        "policy": "unchanged successful measurements; explicit terminal safety outcomes; BitMark diagnostics-only guard",
    }
    if execution_changes:
        audit["runtime_migration"] = {
            "authorization": runtime_migration,
            "changes": execution_changes,
            "numerical_equivalence_verified": False,
            "policy": "Explicit heterogeneous-runtime continuation; retain old measurements and actual provenance. Review runtime effects before scientific pooling.",
        }
    if parent is not None:
        # Preserve the predecessor on disk; bind this additive amendment to it.
        audit["parent_amendment_sha256"] = sha256(parent_path)
        audit["compatible_previous_codes"] = [PARENT_CODE]
    if v2 is not None:
        audit["v2_amendment_sha256"] = sha256(v2_path)
        audit.setdefault("compatible_previous_codes", []).append(V2_CODE)
    if v3 is not None:
        audit["v3_amendment_sha256"] = sha256(v3_path)
        audit.setdefault("compatible_previous_codes", []).append(V3_CODE)
    if v4 is not None:
        audit["v4_amendment_sha256"] = sha256(v4_path)
        audit.setdefault("compatible_previous_codes", []).append(V4_CODE)
    v5_path = root / "amendments" / (V5_AMENDMENT + ".json")
    if v5_path.exists():
        v5 = read_json(v5_path)
        expected_v5 = copy.deepcopy(audit)
        expected_v5.update(
            amendment=V5_AMENDMENT, current_code=V5_CODE, certificate_sha256=V5_CERTIFICATE,
        )
        # The historical source path is operational; all other fields, runtime
        # migration authorization and predecessor links must match exactly.
        expected_v5["current_identity"]["code"] = V5_CODE
        expected_v5["current_identity"]["project_root"] = v5.get("current_identity", {}).get("project_root")
        if v5 != expected_v5:
            raise ValueError("Incompatible v5 predecessor amendment")
        audit["v5_amendment_sha256"] = sha256(v5_path)
        audit.setdefault("compatible_previous_codes", []).append(V5_CODE)
    if audit_path.exists():
        if read_json(audit_path) != audit:
            raise ValueError("An incompatible amendment already exists for this run")
    elif parent is None and v2 is None and v3 is None and v4 is None:
        # The only model-specific code repair must not mix two successful
        # implementations. Failed attempts are retained as evidence.
        for path in (root / "records").glob("*.json"):
            record = read_json(path)
            if record.get("status") == "ok" and record.get("spec", {}).get("operator") == "bitmark":
                raise ValueError("Existing successful BitMark records require a separate run")
    return old, audit


def compatible_preparation(
    previous: dict, candidate: dict, operator: str, amendment: dict | None, audit_path: Path
) -> dict:
    """Retain the logical environment ID only for precisely audited changes."""
    for value in (previous, candidate):
        if digest(value["identity"]) != value["fingerprint"]:
            raise ValueError("Corrupted dependency fingerprint")
    before, after = copy.deepcopy(previous["identity"]), copy.deepcopy(candidate["identity"])
    differences = []
    previous_codes = {amendment["previous_code"]} if amendment else set()
    if amendment:
        previous_codes.update(
            {PARENT_CODE, V2_CODE, V3_CODE, V4_CODE, V5_CODE} & set(amendment.get("compatible_previous_codes", []))
        )
    if amendment and before.get("controller_code") in previous_codes:
        if after.get("controller_code") == amendment["current_code"]:
            after["controller_code"] = before["controller_code"]
            differences.append("audited_controller_maintenance")
    migration = (amendment or {}).get("runtime_migration", {})
    if before.get("runtime") != after.get("runtime") and migration:
        runtime_transition(before.get("runtime", {}), after.get("runtime", {}), migration.get("authorization"))
        # Comparison only: the returned execution_identity retains the new
        # runtime. Logical identity is a reuse key, not a claim of equivalence.
        after["runtime"] = copy.deepcopy(before.get("runtime", {}))
        differences.append("explicit_heterogeneous_runtime_not_equivalence")
    if operator == "invismark" and _invismark_event_outputs(before, after):
        differences.append("invismark_tensorboard_output_only")
    if operator in ARCHIVE_RULES:
        namespace, files = ARCHIVE_RULES[operator]
        a, b = before.get("weights", {}), after.get("weights", {})
        archive = f"downloads/{namespace}.zip"
        required = {archive} | {f"model_files/{namespace}/{name}" for name in files}
        if required <= set(a) and set(a) == set(b) and a[archive] != b[archive]:
            b[archive] = a[archive]
            label = "trustmark" if operator == "trustmark_q" else operator
            differences.append(f"{label}_zip_packaging_only")
    if before != after:
        raise RuntimeError(
            f"Dependency fingerprint changed for {operator}; retain evidence and investigate before resuming"
        )
    write_json(
        audit_path,
        {
            "amendment": AMENDMENT,
            "differences": differences,
            "approved": previous,
            "executed": candidate,
        },
    )
    # Paths/config hashes come from the current installation; identity and
    # logical fingerprint refer to the unchanged approved measurement bank.
    return {
        **candidate,
        "identity": previous["identity"],
        "fingerprint": previous["fingerprint"],
        "execution_identity": candidate["identity"],
        "execution_fingerprint": candidate["fingerprint"],
        "compatibility_audit": str(audit_path),
    }
