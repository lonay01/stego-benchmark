"""Use existing carriers, full payloads and decoder provenance without mutation."""

from pathlib import Path
import zipfile

from wmbench.io import read_json, write_json, sha256, digest as bank_digest, download, atomic_bytes
from wmbench_full_payload.storage import EvidenceError, freeze, disjoint, inside
from wmbench_bank.contracts import POLICY as BANK_POLICY, models_for_run
from wmbench_digital.bank import collect
from .spec import POLICY, digest, ATTACKS, seed


def choose_captions(value, ids):
    result = {}
    for row in sorted(value["annotations"], key=lambda r: int(r["id"])):
        image = int(row["image_id"])
        if image in ids and str(image) not in result:
            text = row["caption"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty official caption")
            result[str(image)] = dict(text=text, annotation_id=int(row["id"]), sha256=digest(text))
    if {int(i) for i in result} != set(ids):
        raise ValueError("Official captions missing for one or more frozen COCO IDs")
    return result


def context(project, settings):
    bank, root, scratch = [Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch")]
    disjoint(bank, root, scratch)
    if (root / "digital-run.json").exists() or (root / "trial.json").exists():
        raise EvidenceError("Choose a NEW neural output directory, not the bank/digital run")
    n = settings["n_images"]
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("N_IMAGES must be an integer in [1,100]")
    if type(settings["gpu_batch_size"]) is not int or not 1 <= settings["gpu_batch_size"] <= 32:
        raise ValueError("GPU_BATCH_SIZE must be 1..32")
    trial, models, release = read_json(bank / "trial.json"), models_for_run(project), read_json(project / "release.json")
    if trial["policy"] != BANK_POLICY or trial["models"] != models or trial["release"] != bank_digest(release):
        raise EvidenceError("Bank is not the reviewed 29-method production release")
    for name, expected in release["files"].items():
        if sha256(inside(project, name)) != expected:
            raise EvidenceError(f"Production decoder changed: {name}")
    own = read_json(project.parent / "neural-release.json")
    for name, expected in own["files"].items():
        if sha256(inside(project.parent, name)) != expected:
            raise EvidenceError(f"Neural release changed: {name}")
    manifest = read_json(bank / "_inputs/manifest.json")
    if trial["dataset"] != manifest["content_hash"] or n > len(manifest["images"]):
        raise EvidenceError("Dataset identity/size mismatch")
    selected = list(models) if settings["models"] == "all" else settings["models"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected)-models.keys():
        raise ValueError("Select 'all' or unique registered method IDs")
    spec = dict(bank_trial=bank_digest(trial), policy=POLICY, neural_release=digest(own),
                gpu_batch_size=settings["gpu_batch_size"])
    freeze(root / "neural-run.json", spec)  # N is a prefix selection, not run identity.
    write_json(root / "selection.json", dict(n_images=n, models=selected))
    return dict(bank=bank, output=root, scratch=scratch, models=models, selected=selected,
                images=manifest["images"][:n], all_images=manifest["images"], identity=digest(spec))


def inputs(ctx, supplied=None):
    """Freeze ALL 100 captions and seed assignments before the canary."""
    root = ctx["output"] / "_inputs"
    path = root / "captions_val2017.json"
    if supplied:
        supplied = Path(supplied)
        if path.exists() and sha256(path) != sha256(supplied):
            raise EvidenceError("Supplied annotations differ from the saved file")
        if not path.exists():
            atomic_bytes(path, supplied.read_bytes())
    if not path.exists():
        archive = ctx["scratch"] / "inputs/annotations_trainval2017.zip"
        download("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as handle:
            atomic_bytes(path, handle.read("annotations/captions_val2017.json"))
    captions = choose_captions(read_json(path), {int(i["image_id"]) for i in ctx["all_images"]})
    freeze(root / "captions.json", dict(annotation_sha256=sha256(path), selection="minimum annotation ID", captions=captions))
    assignments = [{"image_id": i["image_id"], "replicate": r,
                    "seeds": {a["id"]: seed(dict(image_id=i["image_id"], replicate=r), a) for a in ATTACKS[1:]}}
                   for i in ctx["all_images"] for r in range(1, 6)]
    freeze(root / "seeds.json", assignments)
    return captions
