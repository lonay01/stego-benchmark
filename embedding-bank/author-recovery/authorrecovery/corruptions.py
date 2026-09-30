"""Same author attacks/seeds; commit each five-severity group and retry errors."""

from __future__ import annotations

import io
from pathlib import Path
import time
import zipfile

from authornight.storage import EvidenceError, cached, commit, freeze, read


def run(job):
    import torch
    from torchvision import transforms
    from PIL import Image
    from authornight.latent import AuthorLatent, set_seed
    from authornight.metrics import bits_score
    from authornight.spec import seed
    root, name = Path(job["output"]), job["method"]
    plan = read(root / "plan.json")
    reference = AuthorLatent(name, Path(job["source"]), root)
    if name == "rosteals":
        from tools.augment_imagenetc import RandomImagenetC
        noise = RandomImagenetC(1, 5, "test")
    else:
        from tools.noise_select import random_select
        noise = random_select(1, 5, "val")
    expected = [(noise.method_names[int(noise_id)], severity)
                for noise_id in noise.corrupt_ids for severity in range(1, 6)]
    freeze(root / "_meta/recovery" / (name + "_attack_grid.json"), [list(x) for x in expected])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    for path in sorted((root / name / "coco/records").glob("*_raw100.json")):
        if time.time() >= job["deadline"]:
            return
        relative = f"{name}/corruptions/{path.stem}.json"
        prior = cached(root, relative, plan["identity"], verify=False)
        scores = {(s["attack"], s["severity"]): s for s in prior.get("scores", [])} if prior else {}
        if prior and len(scores) != len(prior.get("scores", [])):
            raise EvidenceError("Duplicate attack observations")
        if set(scores) - set(expected):
            raise EvidenceError("Unexpected historical attack grid")
        if len(scores) == len(expected) and all(s["status"] == "measured" for s in scores.values()):
            continue
        row = cached(root, path.relative_to(root).as_posix(), plan["identity"])
        with zipfile.ZipFile(root / next(iter(row["artifacts"]))) as archive:
            image = Image.open(io.BytesIO(archive.read("internal.png"))).convert("RGB")

        def save():
            ordered = [scores[pair] for pair in expected if pair in scores]
            complete = len(ordered) == len(expected) and all(s["status"] == "measured" for s in ordered)
            body = {"method": name, "status": "measured" if complete else "partial_or_errors",
                    "profile": "all_five_severities", "source_record": path.relative_to(root).as_posix(),
                    "source_record_sha256": row["record_sha256"], "scores": ordered,
                    "note": "Same internal floor PNG + author pool + all severities; not paper random severity sampling"}
            if prior and "imported_from" in prior:
                body["imported_from"] = prior["imported_from"]
            commit(root, relative, plan["identity"], body)

        for noise_id in noise.corrupt_ids:
            for severity in range(1, 6):
                if time.time() >= job["deadline"]:
                    save()
                    return
                pair = (noise.method_names[int(noise_id)], severity)
                if scores.get(pair, {}).get("status") == "measured":
                    continue
                set_seed(seed(name, "corruption", path.stem, int(noise_id), severity))
                try:
                    perturbed = noise(image.copy(), int(noise_id), severity)
                    with torch.no_grad():
                        score = bits_score(reference.decode(transform(perturbed)[None].cuda()), row["payload"])
                    score.pop("bits")
                    score.pop("logits")
                    scores[pair] = {"attack": pair[0], "severity": severity, "status": "measured", **score}
                except Exception as error:
                    scores[pair] = {"attack": pair[0], "severity": severity, "status": "error", "error": repr(error)}
            save()
        print(f"SAVED {name} grid {path.stem}: {sum(s['status']=='measured' for s in scores.values())}/{len(expected)}", flush=True)
