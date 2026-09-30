"""Reuse verified inputs and receipted caches, never import unmeasured success."""

from pathlib import Path

from authornight.storage import EvidenceError, digest, disjoint, freeze, inside, read
from authorrecovery.migration import copy_verified

PARENT_ID = "31e475a2773d73beafdcfcfc7704fcee8069a0a0782b67552bad37d4e85e1565"
PARENT_RELEASE = "42ccd248d1764e78cc4a662626b07dab63778d83c56037bf47f60dbcca11c8dd"


def make_plan(settings, code_hash):
    parent, root, scratch = [Path(settings[k]).resolve() for k in ("parent", "output", "scratch")]
    disjoint(parent, root, scratch)
    if not 0 < float(settings["max_hours"]) <= 24:
        raise ValueError("MAX_HOURS must be in (0, 24]")
    old = read(parent / "plan.json")
    if old.get("identity") != PARENT_ID or old.get("code_sha256") != PARENT_RELEASE:
        raise EvidenceError("Select the reviewed RoSteALS_SepWater_RECOVERY run")
    if digest({k: v for k, v in old.items() if k != "identity"}) != PARENT_ID:
        raise EvidenceError("Parent plan checksum mismatch")
    if old["sizes"]["rosteals"] != 256 or old["messages_per_image"] != 5:
        raise EvidenceError("Unexpected scientific settings")
    if root.exists() and not (root / "plan.json").exists():
        extra = {p.name for p in root.iterdir()} - {"_release", "_logs", "_reports"}
        if extra:
            raise EvidenceError(f"Use a new output folder, found: {sorted(extra)}")
    plan = {"schema": "rosteals-import-fix-v1", "parent_identity": PARENT_ID,
            "parent_root": str(parent), "code_sha256": code_hash,
            "images": old["images"], "messages_per_image": 5,
            "profiles": {"rosteals": old["profiles"]["rosteals"]},
            "sizes": {"rosteals": 256}, "sources": {k: old["sources"][k]
                for k in ("rosteals", "taming", "imagenetc")},
            "input": old["input"], "cli_checks": 5, "training": False,
            "matched_psnr": False, "full_paper_reproduction": False,
            "fix": "Use ImageNet-C/imagenet_c as package root in prepare and worker environments",
            "invariants": "Same weights, original JPEGs, seeds, 100-bit profiles, native author PNG export",
            "attacks": {"pool": "author test", "operators": 14, "severities": [1, 2, 3, 4, 5],
                        "input": "saved internal floor PNG", "profile": "raw100"},
            "excluded": ["sepwater", "distseal"],
            "paper_limits": ["COCO replaces the paper evaluation dataset",
                             "Fixed full-severity grid is not the paper's random severity sampling",
                             "BCH text uses seven hex ASCII characters, not 56 random useful bits"]}
    plan["identity"] = digest(plan)
    freeze(root / "plan.json", plan)
    return plan


def import_parent(settings, code_hash):
    plan = make_plan(settings, code_hash)
    parent, root = Path(settings["parent"]), Path(settings["output"])
    originals = read(parent / "_inputs/originals.json")
    if [x["image_id"] for x in originals] != [int(x["image_id"]) for x in plan["images"]]:
        raise EvidenceError("Source image IDs/order differ from the fixed plan")
    for index, item in enumerate(originals):
        copy_verified(inside(parent, item["path"]), inside(root, item["path"]), item["sha256"])
        if index % 20 == 0 or index + 1 == len(originals):
            print(f"Verified original {index+1}/{len(originals)}", flush=True)
    freeze(root / "_inputs/originals.json", originals)
    # RoSteALS produced no records in the reviewed run. Keep it as evidence;
    # do not invent/import results or carry failed preflight statuses forward.
    if list((parent / "rosteals/coco/records").glob("*.json")):
        raise EvidenceError("Parent contains new RoSteALS results; review before migrating")
    for name in ("execution.json", "environment_request.json", "installed_packages.json"):
        source = parent / "_meta/rosteals" / name
        if source.exists():
            freeze(root / "_meta/rosteals" / name, read(source))
    for method in ("rosteals", "taming", "imagenetc"):
        source = parent / "_meta/sources" / (method + ".json")
        if source.exists():
            freeze(root / "_meta/sources" / source.name, read(source))
    for directory in ("rosteals", "sources"):
        for receipt in sorted((parent / "_assets" / directory).glob("*.receipt.json")):
            name = receipt.name.removesuffix(".receipt.json")
            if directory == "sources" and not name.startswith(("rosteals-", "taming-", "imagenetc-")):
                continue
            metadata = read(receipt)
            target = inside(root, f"_assets/{directory}/{name}")
            print(f"Reusing verified download: {directory}/{name}", flush=True)
            copy_verified(receipt.with_name(name), target, metadata["sha256"])
            freeze(target.with_name(receipt.name), metadata)
    freeze(root / "_meta/fix/parent.json", {"identity": PARENT_ID, "code_sha256": PARENT_RELEASE,
            "parent_root": str(parent), "sepwater": "not run, previous evidence stays in parent"})
    print("Input/cache reuse complete. SepWater and DistSeal untouched.", flush=True)
    return plan
