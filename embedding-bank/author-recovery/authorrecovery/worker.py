"""Isolated CPU/GPU phases. A failed preflight blocks that model's long stages."""

from __future__ import annotations

import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

from authornight.storage import EvidenceError, cached, freeze, read, sha256, write


def prepare_ecc(job):
    root, scratch, project = [Path(job[k]) for k in ("output", "scratch", "project")]
    folder = scratch / "environments/ecc"
    python = folder / "bin/python"
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv==0.8.15 first")
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(scratch / "uv-cache"), UV_NO_CACHE="1")
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.10.18", str(folder)], check=True, env=env)
    subprocess.run([uv, "pip", "sync", "--python", str(python), str(project / "locks/ecc.txt")], check=True, env=env)
    write(root / "_meta/ecc/prepared.json", {"python": str(python), "lock_sha256": sha256(project / "locks/ecc.txt")})


def configure_gpu(job):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze(Path(job["output"]) / "_meta" / job["method"] / "execution.json", {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "cudnn": torch.backends.cudnn.version(),
        "tf32": False, "dtype": "float32", "cudnn_deterministic": True})


def preflight(job):
    import numpy as np
    from PIL import Image
    from authornight.latent import AuthorLatent, run_coco, set_seed
    root = Path(job["output"])
    plan = read(root / "plan.json")
    checks = []
    if job["method"] == "rosteals":
        # Full author config, model weights, metrics, PNG, and a fresh author CLI.
        run_coco({**job, "image_limit": 1})
        image_id = int(plan["images"][0]["image_id"])
        for repetition in range(5):
            for profile in ("raw100", "author_bch56"):
                relative = f"rosteals/coco/records/{image_id:012d}_m{repetition}_{profile}.json"
                row = cached(root, relative, plan["identity"])
                if not row or not row["checks"] or not all(row["checks"].values()):
                    raise EvidenceError(f"RoSteALS preflight missing/failed: {relative}")
                checks.append(relative)
        # Import the actual corruption pool too, not just the neural model.
        from tools.augment_imagenetc import RandomImagenetC
        if len(RandomImagenetC(1, 5, "test").corrupt_ids) != 14:
            raise EvidenceError("Unexpected RoSteALS attack pool")
    else:
        reference = AuthorLatent("sepwater", Path(job["source"]), root)
        for item in read(root / "_inputs/originals.json")[:5]:
            input_file = root / item["path"]
            if sha256(input_file) != item["sha256"]:
                raise EvidenceError("Input image changed")
            for profile in ("raw100", "author_bch56"):
                relative = f"sepwater/coco/records/{int(item['image_id']):012d}_m0_{profile}.json"
                row = cached(root, relative, plan["identity"])
                if row is None:
                    raise EvidenceError("The reviewed SepWater cohort is missing; restore the original run")
                set_seed(row["seed"])
                data, images = reference.measure(Image.open(input_file).convert("RGB"), row["payload"])
                with zipfile.ZipFile(root / next(iter(row["artifacts"]))) as archive:
                    expected = np.asarray(Image.open(io.BytesIO(archive.read("author.png"))))
                actual = np.asarray(Image.open(io.BytesIO(images["author.png"])))
                equal = np.array_equal(expected, actual) and all(
                    data["measurements"][stage]["bits"] == row["measurements"][stage]["bits"]
                    for stage in ("raw_internal", "author_png"))
                checks.append({"record": relative, "pixels_and_bits_equal": bool(equal)})
                if not equal:
                    write(root / "_meta/recovery/sepwater_replay_failure.json", checks)
                    raise EvidenceError("SepWater replay differs; do not combine new attacks with historical evidence")
        from tools.noise_select import random_select
        if len(random_select(1, 5, "val").corrupt_ids) != 9:
            raise EvidenceError("Unexpected SepWater attack pool")
    write(root / "_meta/recovery" / (job["method"] + "_preflight.json"), {"passed": True, "checks": checks})
    print(f"PREFLIGHT PASSED: {job['method']}; {len(checks)} cases", flush=True)


def main(job):
    phase = job["phase"]
    if phase == "import":
        from .migration import import_parent
        import_parent(job["settings"], job["code_hash"])
        return
    if phase == "prepare_ecc":
        prepare_ecc(job)
        return
    if phase == "ecc":
        from .ecc import run
        run(job["output"], job["deadline"])
        return
    if job["method"] not in ("rosteals", "sepwater"):
        raise ValueError("This recovery cannot run DistSeal or any other model")
    if phase == "prepare":
        from authornight.environment import REQUIREMENTS, prepare
        if "slackclient==2.9.3" not in REQUIREMENTS["rosteals"]:
            REQUIREMENTS["rosteals"].append("slackclient==2.9.3")
        prepare(job["method"], Path(job["project"]), Path(job["output"]), Path(job["scratch"]))
        return
    configure_gpu(job)
    if phase == "preflight":
        preflight(job)
    elif phase == "coco" and job["method"] == "rosteals":
        from authornight.latent import run_coco
        run_coco(job)
    elif phase == "corruptions":
        from .corruptions import run
        run(job)
    else:
        raise ValueError(f"Unsupported recovery phase: {phase}")


if __name__ == "__main__":
    main(read(sys.argv[1]))
