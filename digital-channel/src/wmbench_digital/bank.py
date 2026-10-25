"""Read-only validation and indexing of the production bank."""

from pathlib import Path
from PIL import Image
from wmbench.io import read_json, sha256, write_json, digest as bank_digest
from wmbench_full_payload.storage import EvidenceError, freeze, disjoint, inside
from wmbench_full_payload.worker import record_path, task_id
from wmbench_bank.contracts import MODES, POLICY as BANK_POLICY, models_for_run, message
from wmbench_bank.worker import signature
from .spec import POLICY, digest


def context(project, settings):
    bank, output, scratch = [Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch")]
    disjoint(bank, output, scratch)
    n = settings["n_images"]
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("N_IMAGES must be an integer between 1 and 100")
    trial = read_json(bank / "trial.json")
    models = models_for_run(project)
    release = read_json(project / "release.json")
    if trial["policy"] != BANK_POLICY or trial["models"] != models or trial["release"] != bank_digest(release):
        raise EvidenceError("Bank contract/release differs from the reviewed production release")
    for path, expected in release["files"].items():
        if sha256(inside(project, path)) != expected:
            raise EvidenceError(f"Production decoder release changed: {path}")
    manifest = read_json(bank / "_inputs/manifest.json")
    if trial["dataset"] != manifest["content_hash"] or n > len(manifest["images"]):
        raise EvidenceError("Frozen dataset identity/size mismatch")
    selected = list(models) if settings["models"] == "all" else settings["models"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected)-models.keys():
        raise ValueError("Select 'all' or a unique list of registered methods")
    own_release = read_json(project.parent / "digital-release.json")
    for path, expected in own_release["files"].items():
        if sha256(inside(project.parent, path)) != expected:
            raise EvidenceError(f"Digital release changed: {path}")
    spec = {"bank_trial": bank_digest(trial), "policy": POLICY, "digital_release": digest(own_release)}
    freeze(output / "digital-run.json", spec)  # N deliberately excluded: five extends to 100.
    write_json(output / "selection.json", {"n_images": n, "models": selected})
    return dict(bank=bank, output=output, scratch=scratch, models=models, selected=selected,
                images=manifest["images"][:n], identity=digest(spec))


def collect(ctx, model):
    bank, cases, coverage = ctx["bank"], [], []
    hashes = {}

    def verify(row):
        for artifact in row.get("artifacts", []):
            name = artifact["path"]
            if name not in hashes:
                hashes[name] = sha256(inside(bank, name))
            if hashes[name] != artifact["sha256"]:
                raise EvidenceError(f"Bank artifact changed: {name}")

    for image in ctx["images"]:
        for rep in range(1, 6):
            stem, sig = task_id(image, rep), signature(model, image, rep)
            native = read_json(record_path(bank, model, stem, "native"))
            if native["signature"] != sig or native["status"] != "saved" or native["assignment"] != message(model, image["image_id"], rep):
                raise EvidenceError(f"Native contract invalid: {model['id']} {stem}")
            verify(native)
            for mode in MODES:
                row = read_json(record_path(bank, model, stem, mode))
                if row["signature"] != sig or row["status"] not in {"saved", "unmatched"}:
                    raise EvidenceError(f"Unfinished bank record: {model['id']} {stem} {mode}")
                coverage.append(dict(method=model["id"], image_id=image["image_id"], replicate=rep,
                                     mode=mode, status=row["status"], matching=row.get("matching")))
                if row["status"] != "saved":
                    continue
                verify(row)
                clean = read_json(record_path(bank, model, stem, mode+"_decode"))
                if clean["signature"] != sig or clean["status"] not in {"measured", "rejected"}:
                    raise EvidenceError("Missing valid bank clean control")
                verify(clean)
                with Image.open(inside(bank, row["output"])) as png:
                    size = [model["native_geometry"]["width"], model["native_geometry"]["height"]]
                    if png.mode != "RGB" or list(png.size) != size:
                        raise EvidenceError("Unexpected saved carrier geometry")
                case = dict(id=f"{stem}_{mode}", method=model["id"], image_id=image["image_id"],
                            replicate=rep, mode=mode, signature=sig, output=row["output"],
                            carrier_sha256=hashes[row["output"]], source_crop=native["source_crop"],
                            decoder_metadata=native["decoder_metadata"], assignment=native["assignment"],
                            clean=clean, psnr=row["psnr"], matching=row.get("matching"),
                            embedding_seconds=native["seconds"], size=size)
                case["key"] = digest({"run": ctx["identity"], "case": case})
                cases.append(case)
        print(f"BANK VERIFIED {model['id']} image {image['image_id']}; eligible carriers so far: {len(cases)}", flush=True)
    write_json(ctx["output"] / model["id"] / "cases.json", cases)
    write_json(ctx["output"] / model["id"] / "coverage.json", coverage)
    return cases
