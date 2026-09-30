"""Run the released DistSeal attack grid using pinned OmniSealBench operators."""

from __future__ import annotations
import ast
import importlib.util
import io
from pathlib import Path
import time
from typing import Any, List, Optional
import zipfile
import numpy as np
from PIL import Image

from .environment import checkout
from .latent import set_seed
from .spec import seed
from .storage import cached, commit, freeze, read, sha256


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operators(source, yaml_path):
    """Avoid importing unrelated OmniSealBench audio/video pipelines."""
    import torch
    import yaml
    base = Path(source) / "src/omnisealbench/attacks"
    geo = load_module(base / "geometric.py", "authornight_omni_geometry")
    value = load_module(base / "valuemetric.py", "authornight_omni_values")
    filename = base / "mixed_video.py"
    tree = ast.parse(filename.read_text("utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("Sequential", "JPEG_Crop_Brightness")]
    if len(nodes) != 2:
        raise ValueError("Unexpected mixed attack source")
    namespace = dict(torch=torch, nn=torch.nn, Crop=geo.Crop, Brightness=value.Brightness,
                     JPEG=value.JPEG, Optional=Optional, List=List, Any=Any)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(filename), "exec"), namespace)
    registry = {"geometric": vars(geo), "valuemetric": vars(value), "mixed_video": namespace}
    result = [("Identity", {}, geo.Identity())]
    for name, config in yaml.safe_load(Path(yaml_path).read_text("utf-8")).items():
        _, _, module, cls = config["func"].split(".")
        instance = registry[module][cls]()
        args = config.get("args", {})
        if not args:
            result.append((name, {}, instance))
        else:
            if len(args) != 1:
                raise ValueError("Review multidimensional attack parameters explicitly")
            parameter, values = next(iter(args.items()))
            result.extend((name, {parameter: v}, instance) for v in values)
    return result


def run(job):
    from .distseal import AuthorDistSeal
    from .metrics import bits_score
    root, source = Path(job["output"]), Path(job["source"])
    plan = read(root / "plan.json")
    model = AuthorDistSeal(source, root, need_ae=False)
    omni = checkout("omnisealbench", root, Path(job["scratch"]))
    grid = operators(omni, source / "scripts/attacks.yaml")
    freeze(root / "_meta/distseal/attack_grid.json", {
        "grid": [{"name": name, "args": args} for name, args, _ in grid],
        "yaml_sha256": sha256(source / "scripts/attacks.yaml"),
        "operator_sha256": {name: sha256(omni / "src/omnisealbench/attacks" / name)
                            for name in ("geometric.py", "valuemetric.py", "mixed_video.py")}})
    cohort = job.get("cohort", "generated")
    for path in sorted((root / "distseal" / cohort / "records").glob("*.json")):
        relative = f"distseal/{cohort}_attacks/{path.stem}.json"
        if cached(root, relative, plan["identity"]):
            continue
        if time.time() >= job["deadline"]:
            return
        row = cached(root, path.relative_to(root).as_posix(), plan["identity"])
        rows = row["samples"] if cohort == "generated" else [{**row, "name": path.stem}]
        scores = []
        with zipfile.ZipFile(root / next(iter(row["artifacts"]))) as archive:
            for sample in rows:
                member = sample["name"] + "_wm.png" if cohort == "generated" else "author.png"
                original = model.tensor(np.array(Image.open(io.BytesIO(archive.read(member))).convert("RGB")))
                for index, (name, args, operation) in enumerate(grid):
                    if time.time() >= job["deadline"]:
                        return
                    set_seed(seed("distseal-attacks", cohort, sample["name"], index))
                    try:
                        # JPEG mutates its argument; every attack gets fresh pixels.
                        attacked, _ = operation(original.clone(), None, **args)
                        score = bits_score(model.decode(attacked)[0], sample["payload"])
                        score.pop("logits")
                        score.pop("bits")
                        scores.append({"sample": sample["name"], "attack": name, "args": args, "status": "measured", **score})
                    except Exception as error:
                        scores.append({"sample": sample["name"], "attack": name, "args": args,
                                       "status": "error", "error": f"{type(error).__name__}: {error}"})
        commit(root, relative, plan["identity"], {"method": "distseal", "cohort": cohort,
            "status": "measured" if all(x["status"] == "measured" for x in scores) else "has_errors",
            "source_record": path.relative_to(root).as_posix(), "scores": scores})
        print(f"SAVED DistSeal {cohort} attacks {path.stem}: {len(scores)} tests", flush=True)
