"""Fixed comparisons, historical inputs, and content-addressed resume."""

from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.io import digest, read_json, sha256
from wmbench_full_payload.storage import EvidenceError, cached, disjoint, freeze
from wmbench_native_canary.contracts import models_for_trial
from wmbench_native_diagnostics.core import copy_frozen, freeze_cases, validate_case

METHODS = ("rosteals", "sepwater", "distseal_posthoc_dcae", "editguard")
PROFILES = {
    "rosteals": ("historical", "resize256"),
    "sepwater": ("historical", "crop256", "resize256"),
    "distseal_posthoc_dcae": ("historical",),
    "editguard": ("historical", "clean_checkpoint"),
}
POLICY = {
    "schema": "four-method-quality-review-v1", "images": 5, "messages": 5,
    "profiles": PROFILES, "attacks": [], "matched_psnr": False,
    "training": False, "production_changes": False,
    "variants": "predefined diagnostic comparisons; no best-BER selection",
    "clean_checkpoint": "official EditGuard clean.pth, separate from degrade.pth",
}


def profiles(name):
    return PROFILES[name]


def row_path(root, case, profile):
    return root / case["model"] / "cases" / case["stem"] / profile / "result.json"


def task_key(identity, case, profile):
    return digest({"run": identity, "case": case, "profile": profile})


def result(root, case, profile, identity):
    validate_case(root, case)
    return cached(root, row_path(root, case, profile), task_key(identity, case, profile))


def complete(root, case, identity, profile=None):
    return all((result(root, case, p, identity) or {}).get("status") == "measured"
               for p in ((profile,) if profile else profiles(case["model"])))


def carrier(original, historical, profile):
    """Change only the predefined geometry; never select a crop by recovery."""
    if original.shape != (512, 512, 3) or original.dtype != np.uint8:
        raise EvidenceError("Expected the frozen RGB512 original")
    if profile in ("historical", "clean_checkpoint"):
        return historical.copy()
    if profile == "crop256":
        return original[128:384, 128:384].copy()
    if profile == "resize256":
        return np.asarray(Image.fromarray(original).resize((256, 256), Image.Resampling.BILINEAR)).copy()
    raise ValueError(profile)


def context(project, settings):
    bank, root, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(project, bank, root, scratch / "slots")
    disjoint(bank, root, scratch)
    if settings.get("n_images") != 5 or settings.get("models") != list(METHODS):
        raise ValueError("This diagnostic requires the four fixed methods and N_IMAGES=5")
    if not (root / "trial.json").exists() and root.exists():
        foreign = [p.name for p in root.iterdir() if p.name not in ("_release", "_reports")]
        if foreign:
            raise EvidenceError(f"Choose a new OUTPUT_ROOT, found: {foreign[:5]}")
    history = read_json(bank / "trial.json")
    if history.get("policy", {}).get("schema") != "native-resolution-canary-v1":
        raise EvidenceError("BANK_ROOT must be the completed native-resolution canary")
    models = models_for_trial(project)  # Frozen dependency ZIP, not today's live registry.
    models = {n: models[n] for n in METHODS}
    images = read_json(bank / "_inputs/manifest.json")["images"][:5]
    if len(images) != 5 or len({i["image_id"] for i in images}) != 5:
        raise EvidenceError("Need five distinct historical originals")
    identity = {"policy": POLICY, "models": models, "historical_trial": digest(history),
                "code": digest({p.relative_to(project).as_posix(): sha256(p)
                                for p in sorted((project / "src").rglob("*.py"))}),
                "sources": read_json(project / "source_contracts.json")}
    # JSON normalizes tuples to lists; compare the normalized form on resume.
    import json
    identity = json.loads(json.dumps(identity))
    freeze(root / "trial.json", identity)
    cases = {}
    for name, model in models.items():
        if history["models"][name] != model:
            raise EvidenceError(f"Frozen model differs from the historical bank: {name}")
        cases[name] = freeze_cases(bank, root, model, images)
        for filename in ("model_identity.json", "dependency_identity.json"):
            copy_frozen(bank / name / "_meta" / filename, root / name / "_meta" / f"historical_{filename}")
        for filename in ("requirements.lock", "request.json"):
            copy_frozen(bank / name / "_meta/environment" / filename, root / name / "_meta/environment" / filename)
    plan = {"identity": digest(identity), "cases": cases, "profiles": {n: list(profiles(n)) for n in METHODS}}
    freeze(root / "plan.json", plan)
    return bank, root, scratch, models, plan
