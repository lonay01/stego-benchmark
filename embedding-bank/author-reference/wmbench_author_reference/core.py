"""Fixed cohorts, source contracts and sealed per-case records."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.io import digest, read_json, sha256
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside
from wmbench_native_canary.contracts import center_crop, message, models_for_trial
from wmbench_native_canary.worker import signature, task_id
from wmbench_native_diagnostics.core import METHODS, copy_frozen, freeze_cases, validate_case

REFERENCE_METHODS = METHODS[:3]
SERIALIZER_METHODS = METHODS[3:]
POLICY = {
    "schema": "author-reference-v1",
    "replay": "first_5_historical_images_x_5_full_messages_all_5_methods",
    "holdout": "next_5_manifest_images_x_5_full_messages_sepmark_and_distseal_only",
    "reference": "separate_process_fresh_upstream_network_selected_checkpoint",
    "attacks": [], "matched_psnr": False, "training": False,
    "serializer_selection": "pinned_source_not_test_BER",
    "float_atol": 1e-6, "float_rtol": 1e-5,
}


def phase_names(name):
    return ("adapter", "reference", "cross") if name in REFERENCE_METHODS else ("adapter",)


def folder_for(root, case):
    folder = root / case["model"] / "cases" / case["stem"]
    if case["cohort"] == "historical":
        validate_case(root, case)
    elif sha256(folder / "source.png") != case["source_sha256"]:
        raise EvidenceError("Holdout crop changed")
    return folder


def row_path(root, case, phase):
    return root / case["model"] / "cases" / case["stem"] / f"{phase}.json"


def key_for(identity, case, phase):
    return digest({"identity": identity, "case": case, "phase": phase})


def get_row(root, case, phase, identity):
    folder_for(root, case)
    return cached(root, row_path(root, case, phase), key_for(identity, case, phase))


def complete(root, case, identity, phase=None):
    for current in (phase,) if phase else phase_names(case["model"]):
        row = get_row(root, case, current, identity)
        if not row or row["status"] != "measured":
            return False
    return True


def holdout_cases(bank, root, model, images):
    cases = []
    for image in images:
        original = inside(bank, image["path"])
        if sha256(original) != image["sha256"]:
            raise EvidenceError("Holdout source hash differs from frozen manifest")
        with Image.open(original) as im:
            cut = np.asarray(center_crop(im, model["native_geometry"]["width"])).copy()
        for replicate in range(1, 6):
            stem = task_id(image, replicate)
            folder = root / model["id"] / "cases" / stem
            source = folder / "source.png"
            if source.exists():
                if not np.array_equal(load_rgb(source), cut):
                    raise EvidenceError("Holdout crop changed")
            else:
                save_rgb(source, cut)
            case = {"model": model["id"], "stem": stem, "image": image, "replicate": replicate,
                    "assignment": message(model, image["image_id"], replicate),
                    "seed": int(signature(model, image, replicate)[:8], 16),
                    "source_sha256": sha256(source), "cohort": "holdout"}
            freeze(folder / "case.json", case)
            cases.append(case)
    return cases


def context(project, settings):
    from wmbench_full_payload.storage import disjoint

    bank, root, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(bank, root, scratch)
    disjoint(project, bank, root)
    if any((root / n).exists() for n in ("bank.json", "diagnostic_identity.json")):
        raise EvidenceError("OUTPUT_ROOT points to another experiment; choose the new reference folder")
    if (root / "trial.json").exists() and read_json(root / "trial.json").get("policy", {}).get("schema") != POLICY["schema"]:
        raise EvidenceError("OUTPUT_ROOT contains a different trial")
    if (root / "plan.json").exists() and not (root / "trial.json").exists():
        raise EvidenceError("OUTPUT_ROOT contains an unidentified experiment plan")
    if settings.get("n_images") != 5:
        raise ValueError("This diagnostic has a fixed five-image replay and five-image holdout")
    trial = read_json(bank / "trial.json")
    if trial.get("policy", {}).get("schema") != "native-resolution-canary-v1":
        raise ValueError("BANK_ROOT must be the native-resolution canary, not the RGB512 bank")
    selected = settings.get("models", list(METHODS))
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - set(METHODS):
        raise ValueError("Select unique IDs from the five diagnostic methods")
    models = models_for_trial(project)
    data = read_json(bank / "_inputs/manifest.json")
    images = data["images"]
    if len(images) < 10 or len({x["image_id"] for x in images[:10]}) != 10:
        raise ValueError("Need the original first ten distinct RGB512 sources; do not replace them")
    contracts = read_json(project / "source_contracts.json")
    identity = {"policy": POLICY, "historical_trial": digest(trial), "contracts": contracts,
                "models": {n: models[n] for n in selected},
                "code": digest({p.relative_to(project).as_posix(): sha256(p)
                                for p in sorted((project / "src").rglob("*.py"))})}
    freeze(root / "trial.json", identity)
    cases = {}
    for name in selected:
        if trial["models"][name] != models[name]:
            raise EvidenceError(f"Historical model definition differs: {name}")
        print(f"Verify original cases: {name}", flush=True)
        # This copies evidence read-only; never writes into the historical bank.
        history = freeze_cases(bank, root, models[name], images[:5])
        cases[name] = [{**c, "cohort": "historical"} for c in history]
        if name in SERIALIZER_METHODS:
            cases[name] += holdout_cases(bank, root, models[name], images[5:10])
        for filename in ("model_identity.json", "dependency_identity.json"):
            copy_frozen(bank / name / "_meta" / filename, root / name / "_meta" / f"historical_{filename}")
        for filename in ("requirements.lock", "request.json"):
            copy_frozen(bank / name / "_meta/environment" / filename, root / name / "_meta/environment" / filename)
    plan = {"identity": digest(identity), "cases": cases,
            "holdout_ids": [x["image_id"] for x in images[5:10]], "policy": POLICY}
    freeze(root / "plan.json", plan)
    return bank, root, scratch, {n: models[n] for n in selected}, plan


def source_root(options, name):
    root = Path(options["repo_root"])
    suffix = {"hidden_benchmark": "submodules/HiDDeN", "rosteals": "submodules/RoSteALS"}.get(name)
    return root / suffix if suffix else root


def verify_sources(project, options, name):
    """Compare Git blob hashes, allowing only checkout CRLF normalization."""
    spec = read_json(project / "source_contracts.json")[name]
    root = source_root(options, name)
    checked = {}
    for relative, expected in spec["files"].items():
        raw = (root / relative).read_bytes()
        candidates = (raw, raw.replace(b"\r\n", b"\n"))
        hashes = [hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest() for b in candidates]
        if expected not in hashes:
            raise EvidenceError(f"Upstream source differs from reviewed revision: {name}/{relative}")
        checked[relative] = {"git_blob": expected, "local_sha256": hashlib.sha256(raw).hexdigest()}
    return {"repo": spec["repo"], "revision": spec["revision"], "files": checked}


def compare_arrays(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise EvidenceError("Invalid comparison tensors")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"exact": bool(np.array_equal(left, right)), "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "close": bool(np.allclose(left, right, atol=POLICY["float_atol"], rtol=POLICY["float_rtol"]))}
