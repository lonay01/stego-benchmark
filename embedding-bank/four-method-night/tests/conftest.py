"""Test the actual source ZIP, not a different checkout import path."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src/wmbench_quality_night").is_dir():
    PROJECT = ROOT
else:
    spec = importlib.util.spec_from_file_location("night_builder", ROOT / "build_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.build()
    TEMP = tempfile.TemporaryDirectory(prefix="wm-night-tests-")
    PROJECT = Path(TEMP.name)
    with zipfile.ZipFile(ROOT / "dist" / builder.ARCHIVE) as archive:
        archive.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    import torch
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


@pytest.fixture
def project():
    return PROJECT


@pytest.fixture
def historical(tmp_path, project):
    import numpy as np
    from wmbench.io import sha256, write_json
    from wmbench.matching import save_rgb, load_rgb
    from wmbench_native_canary.contracts import models_for_trial, POLICY
    from wmbench_native_canary.worker import process_pair
    from wmbench_quality_night.core import METHODS
    from wmbench.io import read_json

    class PixelBits:
        def embed(self, source, output, bits, metadata):
            image = load_rgb(source).copy()
            image.flat[:len(bits)] = (image.flat[:len(bits)] & 254) | np.asarray(bits, dtype=np.uint8)
            save_rgb(output, image)
            write_json(metadata, {"expected_bits": bits})
        def decode(self, path, metadata):
            dec = read_json(metadata)
            assert not any(dec["expected_bits"])
            return {"bits": (load_rgb(path).flat[:len(dec["expected_bits"])] & 1).tolist()}

    bank, output, scratch, previous = (tmp_path / n for n in ("bank", "out", "scratch", "previous"))
    models = models_for_trial(project)
    images = []
    for i in range(6):
        path = bank / "_inputs/images" / f"{i:012d}.png"
        save_rgb(path, np.full((512, 512, 3), 50 + i, dtype=np.uint8))
        images.append({"image_id": i, "path": path.relative_to(bank).as_posix(), "sha256": sha256(path)})
    write_json(bank / "_inputs/manifest.json", {"images": images})
    write_json(bank / "trial.json", {"policy": POLICY, "models": models})
    for name in METHODS:
        for image in images[:5]:
            for replicate in range(1, 6):
                process_pair(PixelBits(), models[name], bank, image, replicate, {})
        meta = bank / name / "_meta"
        write_json(meta / "model_identity.json", {"test_only": True})
        write_json(meta / "environment/request.json", {"test_only": True})
        lock = meta / "environment/requirements.lock"
        lock.write_text("numpy==1.26.4\n", encoding="utf-8")
        snapshot = meta / "environment/installed.freeze.txt"
        snapshot.write_text("numpy==1.26.4\n", encoding="utf-8")
        write_json(meta / "dependency_identity.json", {"lock_sha256": sha256(lock), "installed_sha256": sha256(snapshot)})
    return {"bank": str(bank), "output": str(output), "scratch": str(scratch), "previous": str(previous),
            "models": list(METHODS), "n_images": 6, "cleanup": False}
