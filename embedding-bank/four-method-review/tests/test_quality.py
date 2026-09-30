"""CPU contracts and orchestration, explicitly not four pretrained GPU runs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from wmbench.adapters.legacy import EditGuardAdapter
from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError, sealed
from wmbench_native_canary.contracts import models_for_trial, POLICY as NATIVE_POLICY
from wmbench_native_canary.worker import process_pair
from wmbench_quality_review.core import METHODS, PROFILES, carrier, context, complete, row_path, task_key
from wmbench_quality_review.probes import decode, direct_latent, editguard_checkpoint, load_clean_checkpoint
from wmbench_quality_review.report import aggregate, report
from wmbench_quality_review.worker import measure


class PixelBits:
    """Synthetic LSB test double, not a watermarking method."""
    def embed(self, source, output, bits, metadata):
        image = load_rgb(source).copy()
        image.flat[:len(bits)] = (image.flat[:len(bits)] & 254) | np.asarray(bits, dtype=np.uint8)
        save_rgb(output, image)
        write_json(metadata, {"expected_bits": bits})

    def decode(self, path, metadata):
        dec = read_json(metadata)
        assert not any(dec["expected_bits"])
        return {"bits": (load_rgb(path).flat[:len(dec["expected_bits"])] & 1).tolist()}


@pytest.fixture
def historical(tmp_path, project):
    bank, root, scratch = tmp_path / "bank", tmp_path / "out", tmp_path / "scratch"
    models = models_for_trial(project)
    images = []
    for i in range(5):
        path = bank / "_inputs/images" / f"{i:012d}.png"
        save_rgb(path, np.full((512, 512, 3), 50 + i, dtype=np.uint8))
        images.append({"image_id": i, "path": path.relative_to(bank).as_posix(), "sha256": sha256(path)})
    write_json(bank / "_inputs/manifest.json", {"images": images})
    write_json(bank / "trial.json", {"policy": NATIVE_POLICY, "models": models})
    for name in METHODS:
        for image in images:
            for replicate in range(1, 6):
                process_pair(PixelBits(), models[name], bank, image, replicate, {})
        for file in ("model_identity.json", "dependency_identity.json"):
            write_json(bank / name / "_meta" / file, {"test_only": True})
        for file in ("requirements.lock", "request.json"):
            write_json(bank / name / "_meta/environment" / file, {"test_only": True})
    settings = {"bank": str(bank), "output": str(root), "scratch": str(scratch),
                "n_images": 5, "models": list(METHODS)}
    return settings


def test_fixed_matrix():
    assert sum(len(v) for v in PROFILES.values()) * 25 == 200
    assert set(METHODS) == {"rosteals", "sepwater", "distseal_posthoc_dcae", "editguard"}


def test_geometry():
    source = np.arange(512 * 512 * 3).reshape(512, 512, 3).astype(np.uint8)
    historic = source[144:368, 144:368]
    assert np.array_equal(carrier(source, historic, "historical"), historic)
    assert np.array_equal(carrier(source, historic, "crop256"), source[128:384, 128:384])
    expected = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.BILINEAR))
    assert np.array_equal(carrier(source, historic, "resize256"), expected)
    with pytest.raises(ValueError):
        carrier(source, historic, "choose_best")


def test_context_resume_readonly_and_full_payload(historical, project):
    bank = Path(historical["bank"])
    before = {str(p): sha256(p) for p in bank.rglob("*") if p.is_file()}
    _, root, _, _, plan = context(project, historical)
    assert context(project, historical)[-1] == plan
    assert before == {str(p): sha256(p) for p in bank.rglob("*") if p.is_file()}
    for name, cases in plan["cases"].items():
        assert len(cases) == 25
        assert len(cases[0]["assignment"]["payload"]) == (64 if name in ("editguard", "distseal_posthoc_dcae") else 100)
        assert not complete(root, cases[0], plan["identity"])
    report(root)
    assert read_json(root / "_reports/summary.json")["complete"] is False
    assert (root / "Four_Method_QUALITY_report.zip").is_file()


def test_resume_keeps_nonzero_ber_and_rejects_corruption(historical, project):
    _, root, _, _, plan = context(project, historical)
    case = plan["cases"]["editguard"][0]
    artifact = root / "test-artifact.json"
    write_json(artifact, {"value": 1})
    sealed(root, row_path(root, case, "historical"), task_key(plan["identity"], case, "historical"),
           {"status": "measured", "ber": 0.25}, [artifact])
    assert complete(root, case, plan["identity"], "historical")
    write_json(artifact, {"value": 2})
    with pytest.raises(EvidenceError, match="Damaged"):
        complete(root, case, plan["identity"], "historical")


class FakeEditNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("test_weight", torch.ones(1))

    def forward(self, x, x_h=None, message=None, rev=False):
        if rev:
            return x.reshape(1, -1)[:, :64] - 0.5
        value = x.clone()
        value.reshape(1, -1)[:, :64] = message + 0.5
        return x, value


def fake_edit():
    adapter = EditGuardAdapter.__new__(EditGuardAdapter)
    adapter.torch, adapter.device = torch, "cpu"
    adapter.net = FakeEditNet().eval()
    adapter.dwt = lambda x: x
    adapter.model = SimpleNamespace(Quantization=lambda x: torch.round(x * 255) / 255)
    return adapter


def test_editguard_real_adapter_boundary_smoke(historical, project):
    bank, root, _, models, plan = context(project, historical)
    case = plan["cases"]["editguard"][0]
    attempt = root / "smoke"
    attempt.mkdir()
    job = {"output": str(root), "bank": str(bank)}
    observed = measure(fake_edit(), job, case, "historical", attempt)
    assert observed["measurements"]["native_png_api"]["ber"] == 0
    assert observed["checks"]["api_direct_bits_equal"]
    assert observed["checks"]["memory_png_bits_equal"]
    assert not any(read_json(attempt / "decoder.json")["expected_bits"])
    assert observed["same_latent_ae"] is False


def test_editguard_checkpoint_strictness(tmp_path):
    adapter = fake_edit()
    path = tmp_path / "clean.pth"
    torch.save({"module.test_weight": torch.tensor([2.])}, path)
    assert load_clean_checkpoint(adapter, path)["all_equal"]
    adapter.net.test_weight.fill_(3)
    with pytest.raises(EvidenceError):
        editguard_checkpoint(adapter, path)


def test_rgb_bgr_conversion_is_symmetric():
    adapter = fake_edit()
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:, :, 2] = 1
    decoded = decode(adapter, "editguard", image)
    assert decoded["bits"] == [1] * 64  # Blue input is first native BGR channel.


def test_same_latent_control_and_normalization():
    class Network:
        def encode_first_stage(self, value):
            self.cover = value.clone()
            return value
        def __call__(self, latent, unused, message):
            return latent + 0.02, None
        def decode_first_stage(self, value):
            return value
    net = Network()
    adapter = SimpleNamespace(wrapper=SimpleNamespace(model=net), device="cpu")
    source = np.full((256, 256, 3), 128, dtype=np.uint8)
    observed = direct_latent(adapter, "rosteals", source, [0] * 100)
    assert np.allclose(observed["ae"], source.astype(np.float32) / 255)
    assert np.allclose(observed["raw"] - observed["ae"], .01, atol=1e-6)
    assert torch.allclose(net.cover, torch.full_like(net.cover, 128 / 255 * 2 - 1), atol=1e-7)


def test_aggregation_keeps_denominator_and_nonzero_errors():
    row = {"status": "measured", "measurements": {"raw": {"ber": .1, "exact": False}},
           "psnr_db": {"original_to_marked": 25.}, "checks": {"test": False}}
    summary = aggregate([row, None, {"status": "error"}])
    assert summary["expected"] == 3 and summary["measured"] == 1
    assert summary["stages"]["raw"]["ber_percent"] == 10
    assert summary["check_failures"]["test"] == 1
    assert not summary["complete"]


def test_notebook_and_frozen_dependencies(project):
    import nbformat
    root = Path(__file__).resolve().parents[1]
    if not (root / "build_notebook.py").is_file():
        pytest.skip("Notebook/base-archive build verification requires the repository checkout")
    nb = nbformat.read(root / "dist/Four_Method_QUALITY_CHECK.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 4 and "PAYLOAD_BASE64" not in "\n".join(code)
    for source in code:
        compile(source, "cell", "exec")
    build = read_json(root / "dist/build.json")
    assert sha256(root / "dist/Four_Method_QUALITY_CHECK_source.zip") == build["sha256"]
    assert build["sha256"] in code[1]
    import zipfile
    with zipfile.ZipFile(root.parent / "meta-recheck/dist/Meta_SepWater_RoSteALS_CHECK_source.zip") as old:
        for name in old.namelist():
            if name.startswith(("src/", "registry/", "locks/")):
                assert (project / name).read_bytes() == old.read(name), name
    for path in (project / "src").rglob("*.py"):
        ast.parse(path.read_text("utf-8-sig"))


def test_missing_clean_asset_does_not_block_baseline(tmp_path, monkeypatch):
    from wmbench_quality_review import runner
    import bank_diagnostics.assets as assets
    monkeypatch.setattr(runner, "prepare_frozen", lambda job: None)
    monkeypatch.setattr(assets, "fetch", lambda *args: (_ for _ in ()).throw(AssertionError("no download allowed")))
    runner.prepare({"model": {"id": "editguard"}, "output": str(tmp_path),
                    "settings": {"download_author_assets": False}})
    assert "Supply" in read_json(tmp_path / "_assets/editguard/clean_download_error.json")["reason"]


def test_production_registry_and_protocols_agree():
    root = Path(__file__).resolve().parents[2]
    registry = root / "registry/bank_models.json"
    if not registry.is_file():
        pytest.skip("Live-catalogue check requires the repository checkout")
    additions = read_json(root / "registry/bank_additions.json")
    rows = read_json(registry) + additions["models"]
    ids = {r["id"] for r in rows}
    assert len(ids) == 29
    assert {"omniguard", "fang_screen", "camera_shooting_instance"}.isdisjoint(ids)
    assert "meta_hidden_revisited" in ids and "hidden_benchmark" not in ids
    for channel in ("digital", "neural", "physical"):
        text = (root.parent / "docs" / f"{channel}-channel-protocol.md").read_text("utf-8")
        assert "| **Total** | **29** |" in text


def test_packaged_command_line_entrypoint(project):
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONPATH": str(project / "src")}
    proc = subprocess.run([sys.executable, "-m", "wmbench_quality_review.runner", "--help"],
                          env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "clean_checkpoint" in proc.stdout and "report" in proc.stdout
