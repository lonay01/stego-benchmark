"""Regression fixtures, CPU boundaries, orchestration and resume invariants."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wmbench.io import read_json, sha256, write_json, digest
from wmbench.matching import save_rgb, load_rgb
from wmbench_full_payload.storage import EvidenceError, sealed
from wmbench_quality_night.core import (METHODS, PROFILES, carrier, context, complete,
                                       import_previous, result, row_path, task_key)
from wmbench_quality_night.environment import dependency_diff, pins, constrain_build_dependencies, verify_dependencies
from wmbench_quality_night.report import report
from wmbench_quality_night.worker import measure


def test_exact_reported_editguard_regression():
    folder = Path(__file__).parent / "fixtures"
    old = (folder / "editguard_historical.freeze.txt").read_text()
    failed = (folder / "editguard_failed.freeze.txt").read_text()
    assert dependency_diff(old, failed) == {"platformdirs": {"expected": "4.11.12", "actual": "4.12.1"}}
    assert dependency_diff(old, old) == {}
    assert len(pins(old)) == 82


@pytest.mark.parametrize("value", ["", "torch>=2", "package @ https://example.com/x.whl", "NumPy==1\nnumpy==1"])
def test_snapshot_rejects_unpinned_or_duplicate(value):
    with pytest.raises(EvidenceError):
        pins(value)


def test_nested_constraints_keep_lock_and_full_snapshot(tmp_path, monkeypatch):
    from wmbench import environment
    from wmbench_full_payload.compat import constrained_builds
    calls = []
    original = lambda args, **kwargs: calls.append(list(map(str, args)))
    monkeypatch.setattr(environment, "command", original)
    lock, full = tmp_path / "requirements.lock", tmp_path / "installed.freeze.txt"
    lock.write_text("torch==2.5.1\n")
    full.write_text("torch==2.5.1\nplatformdirs==4.11.12\n")
    with constrain_build_dependencies(full, "3.10.18"):
        with constrained_builds(lock):
            environment.command(["uv", "venv", "--python", "3.10", "venv"])
            environment.command(["uv", "pip", "install", "--no-build-isolation", "basicsr==1.4.2"])
    assert environment.command is original
    assert calls[0][calls[0].index("--python") + 1] == "3.10.18"
    assert calls[1].count("--constraint") == 2
    assert str(lock) in calls[1] and str(full) in calls[1]


def test_dependency_gate_not_disabled(tmp_path):
    meta, env = tmp_path / "meta", tmp_path / "env"
    meta.mkdir(); env.mkdir()
    old, actual = meta / "historical_installed.freeze.txt", env / "installed.freeze.txt"
    old.write_text("platformdirs==4.11.12\n")
    actual.write_text("platformdirs==4.12.1\n")
    expected = {"lock_sha256": "same-lock", "installed_sha256": sha256(old)}
    write_json(meta / "historical_dependency_identity.json", expected)
    prepared = {"config": str(env / "worker_config.json"),
                "identity": {"lock_sha256": "same-lock", "installed_sha256": sha256(actual)}}
    with pytest.raises(EvidenceError, match="not restored exactly"):
        verify_dependencies(meta, prepared)
    assert read_json(meta / "dependency_audit.json")["package_diff"]["platformdirs"]["actual"] == "4.12.1"
    actual.write_bytes(old.read_bytes())
    prepared["identity"]["installed_sha256"] = sha256(actual)
    assert verify_dependencies(meta, prepared)["package_diff"] == {}
    assert len(list((meta / "dependency_audits").glob("*.json"))) == 2


def test_matrix_and_geometry():
    assert sum(map(len, PROFILES.values())) * 20 * 5 == 700
    source = np.arange(512 * 512 * 3).reshape(512, 512, 3).astype(np.uint8)
    from PIL import Image
    expected = np.asarray(Image.fromarray(source).resize((224, 224), Image.Resampling.BILINEAR))
    assert np.array_equal(carrier(source, source[144:368, 144:368], "sepwater", "resize224"), expected)
    for bad in ("crop256", "resize256"):
        with pytest.raises(ValueError):
            carrier(source, source, "sepwater", bad)


def test_context_full_payload_and_readonly(historical, project):
    bank = Path(historical["bank"])
    before = {str(p): sha256(p) for p in bank.rglob("*") if p.is_file()}
    _, root, _, _, plan = context(project, historical)
    assert context(project, historical)[-1] == plan
    assert before == {str(p): sha256(p) for p in bank.rglob("*") if p.is_file()}
    for name, cases in plan["cases"].items():
        assert len(cases) == 30
        assert sum(c["cohort"] == "historical" for c in cases) == 25
        assert sum(c["cohort"] == "additional" for c in cases) == 5
        assert len(cases[0]["assignment"]["payload"]) == (100 if name in ("rosteals", "sepwater") else 64)
    summary = report(root, archive=True)
    assert summary["expected"] == 210 and summary["measured"] == 0
    assert summary["checks_ok"] is None and not summary["complete"]
    assert (root / "Four_Method_NIGHT_report.zip").is_file()


def test_bad_n_or_overlapping_output(historical, project):
    with pytest.raises(ValueError):
        context(project, {**historical, "n_images": 101})
    with pytest.raises(ValueError):
        context(project, {**historical, "output": historical["bank"]})


def test_failed_setup_snapshot_does_not_replace_historical_or_block_resume(historical, project):
    _, root, _, _, plan = context(project, historical)
    meta = root / "editguard/_meta"
    snapshot = meta / "historical_installed.freeze.txt"
    historical_hash = sha256(snapshot)
    failed = meta / "environment/installed.freeze.txt"
    failed.write_text("numpy==0.0.0\n")
    assert context(project, historical)[-1] == plan
    assert sha256(snapshot) == historical_hash
    assert failed.read_text() == "numpy==0.0.0\n"
    prepared = {"config": str(failed.parent / "worker_config.json"),
                "identity": {**read_json(meta / "historical_dependency_identity.json"),
                             "installed_sha256": sha256(failed)}}
    with pytest.raises(EvidenceError, match="not restored exactly"):
        verify_dependencies(meta, prepared)


def test_artifact_corruption_is_not_regenerated(historical, project):
    _, root, _, _, plan = context(project, historical)
    case = plan["cases"]["editguard"][-1]
    artifact = root / "example.txt"
    artifact.write_text("evidence")
    sealed(root, row_path(root, case, "historical"), task_key(plan["identity"], case, "historical"),
           {"status": "measured", "ber": .1}, [artifact])
    assert complete(root, case, "historical", plan["identity"])
    artifact.write_text("changed")
    with pytest.raises(EvidenceError, match="Damaged"):
        complete(root, case, "historical", plan["identity"])


class FakeEditNet(torch.nn.Module):
    """Synthetic bit carrier, not the pretrained EditGuard model."""
    def forward(self, x, x_h=None, message=None, rev=False):
        if rev:
            return x.reshape(1, -1)[:, :64] - 0.5
        value = x.clone()
        value.reshape(1, -1)[:, :64] = message + .5
        return x, value


def test_actual_editguard_adapter_cpu_boundaries(historical, project, monkeypatch):
    from wmbench.adapters.legacy import EditGuardAdapter
    from wmbench_quality_night import worker
    decoder_calls = []
    original_decode_metadata = worker.decode_metadata
    def one_decoder_metadata(path):
        decoder_calls.append(path.name)
        return original_decode_metadata(path)
    monkeypatch.setattr(worker, "decode_metadata", one_decoder_metadata)
    _, root, _, _, plan = context(project, historical)
    adapter = EditGuardAdapter.__new__(EditGuardAdapter)
    adapter.torch, adapter.device, adapter.net = torch, "cpu", FakeEditNet().eval()
    adapter.dwt = lambda x: x
    adapter.model = SimpleNamespace(Quantization=lambda x: torch.round(x * 255) / 255)
    case = plan["cases"]["editguard"][-5]
    attempt = root / "cpu-smoke"
    attempt.mkdir()
    observed = measure(adapter, {"output": str(root)}, case, "historical", attempt)
    for key in ("repeat_raw_equal", "repeat_png_equal", "api_direct_bits_equal", "float_reload_bits_equal", "memory_png_bits_equal"):
        assert observed["checks"][key]
    assert observed["measurements"]["native_png_api"]["ber"] == 0
    assert not any(read_json(attempt / "decoder.json")["expected_bits"])
    assert decoder_calls == ["metadata.json"]
    assert "floor_png" in observed["measurements"] and "half_up_png" in observed["measurements"]


def test_sepwater_224_wrapper_not_256_bypass(historical, project, monkeypatch):
    from wmbench.adapters.physical import SepWater
    from wmbench_quality_night import worker

    class Network(torch.nn.Module):
        def encode_first_stage(self, x):
            assert x.shape[-2:] == (224, 224)
            return x
        def forward(self, latent, unused, payload):
            return latent + .01, None
        def decode_first_stage(self, x):
            return x
        def decoder(self, x):
            assert x.shape[-2:] == (224, 224)
            return torch.ones((1, 100))

    _, root, _, _, plan = context(project, historical)
    adapter = SepWater.__new__(SepWater)
    adapter.model_id = "sepwater"
    adapter.device, adapter.network, adapter.load_details = "cpu", Network().eval(), {}
    monkeypatch.setattr(worker, "direct_latent", lambda *a: (_ for _ in ()).throw(AssertionError("no bypass")))
    case = plan["cases"]["sepwater"][-5]
    attempt = root / "sepwater-cpu-smoke"
    attempt.mkdir()
    observed = measure(adapter, {"output": str(root)}, case, "resize224", attempt)
    assert observed["geometry"] == [224, 224] and observed["api_applicable"]
    assert observed["same_latent_ae"] and observed["checks"]["repeat_png_equal"]
    assert observed["checks"]["api_direct_bits_equal"]


def synthetic_prior(project, settings, new_plan, root):
    from wmbench_quality_review import core as old
    previous = Path(settings["previous"])
    old_settings = {**settings, "output": str(previous), "n_images": 5, "models": list(old.METHODS)}
    _, _, _, _, old_plan = old.context(project, old_settings)
    trial = read_json(previous / "trial.json")
    trial["code"] = read_json(project / "NIGHT_BASE.json")["prior_code_sha256"]
    write_json(previous / "trial.json", trial)
    old_plan["identity"] = digest(trial)
    write_json(previous / "plan.json", old_plan)
    case = old_plan["cases"]["rosteals"][0]
    artifact = previous / "rosteals/test.png"
    save_rgb(artifact, np.zeros((8, 8, 3), dtype=np.uint8))
    data = {"status": "measured", "measurements": {"raw": {"exact": False, "ber": .1},
            "native_png_direct": {"exact": False, "ber": .1}}, "checks": {"pixels": True},
            "psnr_db": {"original_to_marked": 25}, "execution": {"runtime": {"cpu_test": True}}}
    sealed(previous, old.row_path(previous, case, "historical"), old.task_key(old_plan["identity"], case, "historical"), data, [artifact])
    return previous, case


def test_import_preserves_nonzero_ber_and_never_edits_source(historical, project):
    _, root, _, _, plan = context(project, historical)
    previous, _ = synthetic_prior(project, historical, plan, root)
    before = {str(p): sha256(p) for p in previous.rglob("*") if p.is_file()}
    assert import_previous(project, root, previous, plan) == 1
    assert import_previous(project, root, previous, plan) == 0
    assert before == {str(p): sha256(p) for p in previous.rglob("*") if p.is_file()}
    case = plan["cases"]["rosteals"][0]
    row = result(root, case, "historical", plan["identity"])
    assert row["measurements"]["raw"]["ber"] == .1
    assert row["extra_controls"] == "not_run_in_v1" and row["imported_from"]
    summary = report(root)
    assert summary["imported"] == 1 and summary["measured"] == 1


def test_prior_changed_code_rejected(historical, project):
    _, root, _, _, plan = context(project, historical)
    previous, _ = synthetic_prior(project, historical, plan, root)
    trial = read_json(previous / "trial.json")
    trial["code"] = "changed"
    write_json(previous / "trial.json", trial)
    with pytest.raises(EvidenceError, match="audited v1"):
        import_previous(project, root, previous, plan)


@pytest.mark.parametrize("failed_phase,exit_code,expect_sibling", [
    ("historical", 1, True), ("historical", 2, False), ("prepare", 1, False)])
def test_profile_failure_continues_next_profile_and_next_method(historical, project, monkeypatch,
                                                               failed_phase, exit_code, expect_sibling):
    from wmbench_quality_night import runner
    from wmbench import colab_canary
    _, root, scratch, models, plan = context(project, historical)
    calls = []
    monkeypatch.setattr(runner, "context", lambda *a: (Path(historical["bank"]), root, scratch, models, plan))
    monkeypatch.setattr(runner, "import_previous", lambda *a: 0)
    monkeypatch.setattr(runner, "complete", lambda *a: False)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *a: SimpleNamespace(free=200 * 1024**3))
    @contextlib.contextmanager
    def lease(*a):
        yield lambda: None
    monkeypatch.setattr(colab_canary, "controller_lease", lease)
    def process(command, env, log, slot, *args, **kwargs):
        phase = command[command.index("wmbench_quality_night.runner") + 1]
        calls.append((slot.name, phase))
        log.parent.mkdir(parents=True, exist_ok=True); log.write_text("CPU orchestration test")
        if phase == "prepare":
            write_json(root / slot.name / "_meta/prepared.json", {"status": "ready", "python": "fake-python"})
        return {"exit_code": exit_code if (slot.name, phase) == ("sepwater", failed_phase) else 0, "reason": ""}
    monkeypatch.setattr(colab_canary, "monitored_process", process)
    observed = runner.run(project, historical)
    assert (("sepwater", "resize224") in calls) == expect_sibling
    assert ("editguard", "clean_checkpoint") in calls
    assert ("distseal_posthoc_dcae", "historical") in calls
    assert not observed["complete"]


def test_report_does_not_hash_large_artifacts(historical, project, monkeypatch):
    from wmbench_quality_night import core
    _, root, _, _, plan = context(project, historical)
    monkeypatch.setattr(core, "sha256", lambda *a: (_ for _ in ()).throw(AssertionError("unexpected large artifact scan")))
    summary = report(root)
    assert summary["expected"] == 210 and not summary["complete"]


def test_release_static_and_notebook(project):
    import ast, zipfile, nbformat
    root = Path(__file__).resolve().parents[1]
    if not (root / "build_notebook.py").exists():
        pytest.skip("Repository-only build checks")
    nb = nbformat.read(root / "dist/Four_Method_NIGHT_CHECK.ipynb", as_version=4)
    nbformat.validate(nb)
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(cells) == 4
    for cell in cells:
        compile(cell, "notebook-cell", "exec")
    assert "BASE64" not in "\n".join(cells)
    assert "N_IMAGES = 20" in cells[0]
    assert "wmbench_quality_night.runner" in cells[2]
    build = read_json(root / "dist/build.json")
    assert sha256(root / "dist/Four_Method_NIGHT_CHECK_source.zip") == build["sha256"]
    assert build["sha256"] in cells[1]
    with zipfile.ZipFile(root.parent / "four-method-review/dist/Four_Method_QUALITY_CHECK_source.zip") as old:
        for name in old.namelist():
            if name.startswith(("src/", "registry/", "locks/")):
                assert (project / name).read_bytes() == old.read(name), name
    for path in (project / "src").rglob("*.py"):
        ast.parse(path.read_text("utf-8-sig"))


def test_packaged_cli(project):
    import os, subprocess, sys
    proc = subprocess.run([sys.executable, "-m", "wmbench_quality_night.runner", "--help"],
        env={**os.environ, "PYTHONPATH": str(project / "src")}, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "resize224" in proc.stdout and "crop256" not in proc.stdout
