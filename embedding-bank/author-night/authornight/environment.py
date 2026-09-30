"""Isolated author environments and persistent, hash-checked downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from .spec import SOURCES
from .storage import EvidenceError, atomic, extract_safe, freeze, read, sha256, write

PYTHON = {"rosteals": "3.9.23", "sepwater": "3.10.18", "distseal": "3.10.18"}
COMMON = ["numpy==1.23.5", "Pillow==9.4.0", "omegaconf==2.3.0", "lpips==0.1.4",
          "einops==0.6.1", "scipy==1.9.3", "scikit-image==0.19.3", "matplotlib==3.6.3",
          "opencv-python-headless==4.8.1.78", "Wand==0.6.13", "requests==2.32.3",
          "gdown==5.2.0", "psutil==6.1.1", "setuptools==59.5.0", "wheel==0.45.1"]
REQUIREMENTS = {
    "rosteals": COMMON + ["torch==1.11.0+cu113", "torchvision==0.12.0+cu113", "pytorch-lightning==1.4.2",
        "torchmetrics==0.6.0", "kornia==0.6.0", "transformers==4.19.2", "open-clip-torch==2.0.2",
        "bchlib==0.14.0", "pandas==1.5.3", "protobuf==3.20.3", "tensorboard==2.11.2", "pyDeprecate==0.3.1",
        "huggingface-hub==0.16.4"],
    "sepwater": COMMON + ["torch==2.0.1", "torchvision==0.15.2", "pytorch-lightning==1.8.6",
        "torchmetrics==0.11.4", "kornia==0.6.12", "transformers==4.30.2", "open-clip-torch==2.20.0",
        "bchlib==2.1.3", "protobuf==3.20.3", "tensorboardx==2.6.2.2", "taming-transformers-rom1504==0.0.6",
        "timm==0.9.16", "huggingface-hub==0.16.4"],
    "distseal": ["torch==2.7.1", "torchvision==0.22.1", "numpy==1.26.4", "Pillow==10.4.0",
        "omegaconf==2.3.0", "einops==0.8.1", "timm==1.0.20", "diffusers==0.34.0",
        "transformers==4.49.0", "huggingface-hub==0.34.4", "accelerate==1.10.1",
        "safetensors==0.6.2", "scipy==1.15.3", "opencv-python-headless==4.11.0.86",
        "lpips==0.1.4", "pytorch-msssim==1.0.0", "torch-fidelity==0.3.0",
        "torchdiffeq==0.2.5", "torchmetrics==1.7.4", "torchinfo==1.8.0", "termcolor==3.1.0",
        "wandb==0.21.1", "tensorboard==2.20.0", "matplotlib==3.10.6", "pandas==2.2.3",
        "scikit-image==0.24.0", "requests==2.32.3", "setuptools==80.9.0", "psutil==6.1.1",
        "PyYAML==6.0.2", "onnx==1.17.0", "onnxsim==0.4.36", "onnxruntime==1.20.1"],
}


def command(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def download(url, destination, timeout=90):
    destination = Path(destination)
    receipt = destination.with_name(destination.name + ".receipt.json")
    if receipt.exists():
        prior = read(receipt)
        if prior["url"] != url or not destination.is_file() or sha256(destination) != prior["sha256"]:
            raise EvidenceError(f"Changed cached download: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download.partial")
    last = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "authornight/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
                size = 0
                length = response.headers.get("Content-Length")
                while True:
                    block = response.read(4 * 1024**2)
                    if not block:
                        break
                    out.write(block)
                    size += len(block)
                if not size or length is not None and size != int(length):
                    raise IOError("Incomplete download")
            os.replace(partial, destination)
            freeze(receipt, {"url": url, "sha256": sha256(destination), "bytes": destination.stat().st_size})
            return destination
        except Exception as error:
            last = error
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last}") from last


def checkout(name, root, scratch):
    repo, revision = SOURCES[name]
    archive = download(f"https://codeload.github.com/{repo}/zip/{revision}", root / "_assets/sources" / f"{name}-{revision}.zip")
    target = scratch / "sources" / f"{repo.split('/')[-1]}-{revision}"
    marker = target / ".authornight-extracted.json"
    if not marker.exists():
        extract_safe(archive, scratch / "sources")
        write(marker, {"archive": sha256(archive)})
    files = {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob('*'))
             if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json") and p != marker}
    freeze(root / "_meta/sources" / (name + ".json"), {"revision": revision, "files": files})
    return target


def runtime_env(project, root, scratch, source):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(project), str(source), str(source / "deps")]),
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(root / "_assets/huggingface"), TORCH_HOME=str(root / "_assets/torch"),
               HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="60",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               UV_CACHE_DIR=str(scratch / "uv-cache"), UV_NO_CACHE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def prepare(name, project, root, scratch):
    if os.name == "nt":
        raise RuntimeError("Author GPU reproduction requires Linux/Colab; CPU unit tests also work on Windows")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install controller uv==0.8.15 first")
    if shutil.disk_usage(scratch).free < 25 * 1024**3:
        raise RuntimeError("Need at least 25 GiB free temporary SSD before environment preparation")
    source = checkout(name, root, scratch)
    env = runtime_env(project, root, scratch, source)
    extras = []
    if name == "rosteals":
        taming = checkout("taming", root, scratch)
        imagenetc = checkout("imagenetc", root, scratch) / "ImageNet-C"
        extras = [str(taming), str(imagenetc)]
    elif name == "distseal":
        # Only attacks need this source. Install its runtime dependencies in the
        # separate attacks stage, not into the author's image-generation path.
        pass
    env["PYTHONPATH"] += os.pathsep + os.pathsep.join(extras)
    meta = root / "_meta" / name
    meta.mkdir(parents=True, exist_ok=True)
    folder = scratch / "environments" / name
    python = folder / "bin/python"
    request = {"python": PYTHON[name], "requirements": REQUIREMENTS[name], "source": SOURCES[name]}
    # Normalize the tuple for roundtripping through JSON.
    freeze(meta / "environment_request.json", json.loads(json.dumps(request)))
    if not python.is_file():
        command([uv, "venv", "--python", PYTHON[name], "--seed", folder], env=env)
    lock = meta / "requirements.lock"
    if not lock.exists():
        # Linux dependency resolution is performed and shipped with this release.
        # Never resolve newer transitive versions in the overnight GPU session.
        shipped = project / "locks" / (name + ".txt")
        atomic(lock, shipped.read_bytes())
    if sha256(lock) != sha256(project / "locks" / (name + ".txt")):
        raise EvidenceError("Stored environment lock differs from this release")
    receipt = folder / "authornight-installed.json"
    if not receipt.exists() or read(receipt).get("lock") != sha256(lock):
        indexes = ["--extra-index-url", "https://download.pytorch.org/whl/cu113", "--index-strategy", "unsafe-best-match"] if name == "rosteals" else []
        command([uv, "pip", "sync", "--python", python, lock, *indexes], env=env)
        write(receipt, {"lock": sha256(lock)})
    snapshot = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], env=env, text=True)
    normalized = sorted(line.strip() for line in snapshot.splitlines() if line.strip())
    freeze(meta / "installed_packages.json", normalized)
    write(meta / "prepared.json", {"python": str(python), "source": str(source), "lock_sha256": sha256(lock)})
    return python, source, env


def prepared_environment(name, project, root, scratch, source):
    env = runtime_env(project, root, scratch, source)
    if name == "rosteals":
        for extra, suffix in (("taming", ""), ("imagenetc", "ImageNet-C")):
            repo, revision = SOURCES[extra]
            env["PYTHONPATH"] += os.pathsep + str(scratch / "sources" / (repo.split('/')[-1] + '-' + revision) / suffix)
    return env


def setup_inputs(settings, root):
    from PIL import Image
    plan = read(root / "plan.json")
    inputs = []
    for item in plan["images"]:
        image_id = int(item["image_id"])
        name = f"{image_id:012d}.jpg"
        # The S3 HTTPS hostname has a valid certificate; never disable TLS.
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}"
        target = download(url, root / "_inputs/coco_original" / name)
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            size = list(image.size)
        inputs.append({"image_id": image_id, "path": target.relative_to(root).as_posix(),
                       "sha256": sha256(target), "size": size, "url": url})
        print(f"COCO original {len(inputs)}/{len(plan['images'])}: {name}", flush=True)
    freeze(root / "_inputs/originals.json", inputs)
    return inputs
