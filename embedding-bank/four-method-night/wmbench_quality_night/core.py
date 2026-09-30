"""Frozen inputs, compatible replay imports, and independent additional cases."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from wmbench.io import atomic_bytes, digest, read_json, sha256
from wmbench_full_payload.storage import EvidenceError, cached, disjoint, freeze, inside, sealed
from wmbench_native_canary.contracts import models_for_trial
from wmbench_native_diagnostics.core import copy_frozen, freeze_cases, validate_case
from wmbench_author_reference.core import holdout_cases

METHODS = ("sepwater", "editguard", "rosteals", "distseal_posthoc_dcae")
CLEAN_SHA256 = "b967b91b94a464ab73f82db63a57ee818e6fc5d99d7ae6aa094a1777cac977d5"
PROFILES = {
    "sepwater": ("historical", "resize224"),
    "editguard": ("historical", "clean_checkpoint"),
    "rosteals": ("historical", "resize256"),
    "distseal_posthoc_dcae": ("historical",),
}
POLICY = {
    "schema": "four-method-quality-night-v2", "messages_per_image": 5,
    "profiles": PROFILES, "attacks": [], "matched_psnr": False, "training": False,
    "additional": "manifest entries 6..N, separate from the five historical images",
    "selection": "all predefined profiles, no best-message/checkpoint/serializer selection",
    "extra_controls": ["floor", "half_up", "float32_roundtrip", "repeat_message_1"],
}


def row_path(root, case, profile):
    return root / case["model"] / "cases" / case["stem"] / profile / "result.json"


def task_key(identity, case, profile):
    return digest({"run": identity, "case": case, "profile": profile})


def folder_for(root, case):
    if case["cohort"] == "historical":
        return validate_case(root, case)
    folder = root / case["model"] / "cases" / case["stem"]
    if sha256(folder / "source.png") != case["source_sha256"]:
        raise EvidenceError("Additional source crop changed")
    return folder


def result(root, case, profile, identity, *, verify=True):
    path = row_path(root, case, profile)
    if verify:
        folder_for(root, case)
        return cached(root, path, task_key(identity, case, profile))
    # Reporting reads small records only. Execution/resume verifies artifacts.
    if not path.is_file():
        return None
    row = read_json(path)
    if row.get("signature") != task_key(identity, case, profile):
        raise EvidenceError(f"Changed task signature: {path}")
    return row


def complete(root, case, profile, identity):
    return (result(root, case, profile, identity) or {}).get("status") == "measured"


def carrier(original, historical, name, profile):
    if original.shape != (512, 512, 3) or original.dtype != np.uint8:
        raise EvidenceError("Expected frozen RGB512 input")
    if profile not in PROFILES[name]:
        raise ValueError(f"Unsupported profile: {name}/{profile}")
    if profile in ("historical", "clean_checkpoint"):
        return historical.copy()
    size = 224 if name == "sepwater" else 256
    return np.asarray(Image.fromarray(original).resize((size, size), Image.Resampling.BILINEAR)).copy()


def context(project, settings):
    bank, root, scratch, previous = (Path(settings[k]).expanduser().resolve()
                                     for k in ("bank", "output", "scratch", "previous"))
    disjoint(bank, root, scratch, previous)
    disjoint(project, bank, root, previous)
    n = settings.get("n_images", 20)
    if type(n) is not int or not 5 <= n <= 100:
        raise ValueError("N_IMAGES must be an integer from 5 to 100")
    if settings.get("models") != list(METHODS):
        raise ValueError("Keep the four-method list and its fixed execution order")
    if not (root / "trial.json").exists() and root.exists():
        foreign = [p.name for p in root.iterdir() if p.name not in ("_release", "_reports")]
        if foreign:
            raise EvidenceError(f"Use the new NIGHT output folder, found {foreign[:5]}")
    history = read_json(bank / "trial.json")
    if history.get("policy", {}).get("schema") != "native-resolution-canary-v1":
        raise EvidenceError("BANK_ROOT must be the original native-resolution canary")
    manifest = read_json(bank / "_inputs/manifest.json")
    images = manifest["images"][:n]
    if len(images) != n or len({str(x["image_id"]) for x in images}) != n:
        raise EvidenceError(f"Need {n} distinct frozen originals; replacements are not downloaded")
    all_models = models_for_trial(project)
    models = {name: all_models[name] for name in METHODS}
    identity = json.loads(json.dumps({"policy": POLICY, "n_images": n, "models": models,
        "historical_trial": digest(history), "images": images,
        "code": digest({p.relative_to(project).as_posix(): sha256(p)
                        for p in sorted((project / "src").rglob("*.py"))}),
        "sources": read_json(project / "source_contracts.json")}))
    freeze(root / "trial.json", identity)
    # Original files are shared across messages; copy and verify each once.
    for image in images:
        origin, target = inside(bank, image["path"]), inside(root, image["path"])
        if sha256(origin) != image["sha256"]:
            raise EvidenceError(f"Original changed: {origin}")
        copy_frozen(origin, target)
    freeze(root / "_inputs/manifest.json", {"images": images})
    cases = {}
    for name, model in models.items():
        if history["models"][name] != model:
            raise EvidenceError(f"Historical model differs: {name}")
        replay = freeze_cases(bank, root, model, images[:5])
        for case in replay:
            case["cohort"] = "historical"
        extra = holdout_cases(bank, root, model, images[5:])
        for case in extra:
            case["cohort"] = "additional"
        cases[name] = replay + extra
        for file in ("model_identity.json", "dependency_identity.json"):
            copy_frozen(bank / name / "_meta" / file, root / name / "_meta" / f"historical_{file}")
        for file in ("requirements.lock", "request.json"):
            copy_frozen(bank / name / "_meta/environment" / file,
                        root / name / "_meta/environment" / file)
        copy_frozen(bank / name / "_meta/environment/installed.freeze.txt",
                    root / name / "_meta/historical_installed.freeze.txt")
        expected = read_json(root / name / "_meta/historical_dependency_identity.json")
        snapshot = root / name / "_meta/historical_installed.freeze.txt"
        if sha256(snapshot) != expected["installed_sha256"]:
            raise EvidenceError(f"Historical installed snapshot hash mismatch: {name}")
    plan = {"identity": digest(identity), "cases": cases, "profiles": PROFILES,
            "n_images": n, "previous": str(previous)}
    plan = json.loads(json.dumps(plan))
    freeze(root / "plan.json", plan)
    return bank, root, scratch, models, plan


def import_previous(project, root, previous, plan):
    """Copy only validated, unchanged v1 measurements; never touch the old run."""
    if not (previous / "plan.json").is_file():
        print("Previous quality run absent; all required cases will be measured.", flush=True)
        return 0
    old_plan, old_trial = read_json(previous / "plan.json"), read_json(previous / "trial.json")
    now = read_json(root / "trial.json")
    release = read_json(project / "NIGHT_BASE.json")
    if (old_plan["identity"] != digest(old_trial) or
            old_trial.get("code") != release["prior_code_sha256"] or
            old_trial.get("policy", {}).get("schema") != "four-method-quality-review-v1" or
            old_trial["historical_trial"] != now["historical_trial"] or old_trial["models"] != now["models"]):
        raise EvidenceError("Previous run is not the audited v1; no automatic migration")
    from wmbench_quality_review.core import result as old_result
    count = 0
    for name, cases in plan["cases"].items():
        prior_cases = {c["stem"]: c for c in old_plan["cases"][name]}
        for case in cases:
            if case["cohort"] != "historical":
                continue
            old_case = prior_cases[case["stem"]]
            if {k: v for k, v in case.items() if k != "cohort"} != old_case:
                raise EvidenceError("Prior source/message identity differs")
            for profile in plan["profiles"][name]:
                if row_path(root, case, profile).exists() or profile not in old_plan["profiles"][name]:
                    continue
                row = old_result(previous, old_case, profile, old_plan["identity"])
                if not row or row["status"] != "measured":
                    continue
                files = []
                for item in row["artifacts"]:
                    origin, target = inside(previous, item["path"]), inside(root, item["path"])
                    copy_frozen(origin, target)
                    files.append(target)
                receipt = row_path(root, case, profile).parent / "imported_v1.json"
                freeze(receipt, row)
                files.append(receipt)
                provenance = {"root": str(previous), "plan_identity": old_plan["identity"],
                              "record_sha256": sha256(row_path(previous, old_case, profile))}
                measured = {k: v for k, v in row.items() if k not in ("signature", "artifacts", "finished_unix")}
                sealed(root, row_path(root, case, profile), task_key(plan["identity"], case, profile),
                       {**measured, "imported_from": provenance, "extra_controls": "not_run_in_v1"}, files)
                runtime = row["execution"]["runtime"]
                freeze(root / name / "_meta/paired_runtime.json", runtime)
                count += 1
                if count % 10 == 0:
                    print(f"Imported {count} verified prior measurements (no inference)", flush=True)
    return count
