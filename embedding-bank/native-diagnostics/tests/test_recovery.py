"""Regressions for transient WIBE imports and evidence-preserving recovery."""

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wmbench.io import digest, read_json, sha256, write_json
from wmbench_full_payload.storage import EvidenceError
from wmbench_native_diagnostics import probes, recovery, runner


def transient_network(tmp_path):
    # Match WIBE's pattern: a live class whose defining module is no longer
    # registered. This produces the actual inspect.getsourcefile TypeError.
    path = tmp_path / "transient_model.py"
    path.write_text('''import torch
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = torch.nn.Linear(2, 2)
        self.control = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)
    def forward(self, x):
        return self.decoder(self.control(self.ae(x)))
''', encoding="utf-8")
    module = types.ModuleType("DIAG_TRANSIENT.network")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    assert module.__name__ not in sys.modules
    with pytest.raises(TypeError, match="built-in class"):
        inspect.getsourcefile(module.Network)
    return module.Network().eval(), path


def test_transient_hidden_class_is_audited_without_import_changes(tmp_path):
    network, path = transient_network(tmp_path)
    adapter = SimpleNamespace(wrapper=SimpleNamespace(encoder_decoder=network,
                                                     params=SimpleNamespace(message_length=30)))
    before = dict(sys.modules)
    result = probes.audit_model(adapter, "hidden_benchmark")
    assert before == dict(sys.modules)
    assert result["source_hashes"][str(path)] == sha256(path)
    assert any(r["strategy"] == "method_code" for r in result["source_resolution"])
    assert result["checkpoint_config"]["message_length"] == 30
    assert not result["training_modules"]


def test_transient_rosteals_still_checks_checkpoint_tensors(tmp_path):
    network, source = transient_network(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    torch.save({"state_dict": network.state_dict()}, checkpoint)
    adapter = SimpleNamespace(wrapper=SimpleNamespace(model=network,
        params=SimpleNamespace(weights_path=str(checkpoint))))
    result = probes.audit_model(adapter, "rosteals")
    assert result["source_hashes"][str(source)] == sha256(source)
    assert result["checkpoint_coverage"] == {"checked_keys": 6, "missing": [], "different": []}
    with torch.no_grad():
        network.control.weight.add_(1)
    with pytest.raises(EvidenceError, match="coverage failure"):
        probes.audit_model(adapter, "rosteals")


def test_unavailable_source_is_explicit_not_fake_provenance():
    files, resolution = probes.class_source_files(int)
    assert not files
    assert resolution["strategy"] == "unavailable"


def test_missing_checkpoint_keys_still_stop_rosteals(tmp_path):
    network, _ = transient_network(tmp_path)
    checkpoint = tmp_path / "incomplete.pt"
    state = network.state_dict()
    del state["decoder.weight"]
    torch.save(state, checkpoint)
    adapter = SimpleNamespace(wrapper=SimpleNamespace(model=network,
        params=SimpleNamespace(weights_path=str(checkpoint))))
    with pytest.raises(EvidenceError, match="coverage failure"):
        probes.audit_model(adapter, "rosteals")


def proposed(project):
    return {"policy": {"schema": "fixture"}, "historical_trial": "fixture-trial",
            "models": {"hidden_benchmark": {"native_bits": 30}},
            "source_code": digest(recovery.source_manifest(project))}


def test_recognized_recovery_preserves_identity_plan_and_records(tmp_path, project):
    current = proposed(project)
    old = {**current, "source_code": recovery.PREVIOUS_SOURCE_SHA256}
    identity = tmp_path / "diagnostic_identity.json"
    write_json(identity, old)
    write_json(tmp_path / "plan.json", {"identity": digest(old), "cases": {}})
    write_json(tmp_path / "sepmark/cases/fixture/stages/raw_float.json", {"original": True})
    paths = [p for p in tmp_path.rglob("*.json")]
    before = {p: p.read_bytes() for p in paths}
    assert recovery.freeze_run_identity(project, tmp_path, current) == old
    assert before == {p: p.read_bytes() for p in paths}
    receipt = read_json(next((tmp_path / "_meta/code_revisions").glob("*.json")))
    assert receipt["preserved_run_identity"] == digest(old)
    assert receipt["current_source_code"] == current["source_code"]
    assert recovery.freeze_run_identity(project, tmp_path, current) == old
    assert len(list((tmp_path / "_meta/code_revisions").glob("*.json"))) == 1


@pytest.mark.parametrize("change", ["source", "policy", "models", "historical_trial"])
def test_recovery_rejects_unknown_or_changed_contract(tmp_path, project, change):
    current = proposed(project)
    old = {**current, "source_code": recovery.PREVIOUS_SOURCE_SHA256}
    if change == "source":
        old["source_code"] = "unknown-release"
    else:
        old[change] = "changed"
    write_json(tmp_path / "diagnostic_identity.json", old)
    with pytest.raises(EvidenceError):
        recovery.freeze_run_identity(project, tmp_path, current)
    assert not (tmp_path / "_meta/code_revisions").exists()
    assert read_json(tmp_path / "diagnostic_identity.json") == old


def test_recovery_rejects_changed_inference_code(tmp_path, project, monkeypatch):
    files = recovery.source_manifest(project)
    files["wmbench_native_diagnostics/core.py"] = "changed-scorer"
    current = proposed(project)
    current["source_code"] = digest(files)
    write_json(tmp_path / "diagnostic_identity.json", {**current,
               "source_code": recovery.PREVIOUS_SOURCE_SHA256})
    monkeypatch.setattr(recovery, "source_manifest", lambda _: files)
    with pytest.raises(EvidenceError, match="outside"):
        recovery.freeze_run_identity(project, tmp_path, current)


def test_fresh_run_remains_content_addressed(tmp_path, project):
    current = proposed(project)
    assert recovery.freeze_run_identity(project, tmp_path, current) == current
    assert read_json(tmp_path / "diagnostic_identity.json") == current
    assert not (tmp_path / "_meta/code_revisions").exists()


def test_recovery_controller_runs_only_the_two_pending_models(tmp_path, monkeypatch):
    from wmbench import colab_canary
    bank, root, scratch, project = [tmp_path / n for n in ("bank", "root", "scratch", "project")]
    for path in (bank, root, scratch, project):
        path.mkdir()
    names = ["hidden_benchmark", "rosteals", "sepwater", "sepmark", "distseal_posthoc_dcae"]
    models = {n: {"id": n} for n in names}
    cases = {n: [{"model": n}] for n in names}
    done = set(names[2:])
    old_status = {"status": "error", "reason": "inspect TypeError", "session": "previous"}
    for n in names:
        write_json(root / n / "_meta/status.json", old_status if n not in done else
                   {"status": "diagnostics_complete", "session": "previous"})
    completed_before = {p: p.read_bytes() for n in done for p in (root / n).rglob("*.json")}
    monkeypatch.setattr(runner, "context", lambda p, s: (bank, root, scratch, models,
        {"identity": "preserved-run", "cases": cases}))
    monkeypatch.setattr(runner, "source_manifest", lambda p: {"source": "recovery"})
    monkeypatch.setattr(runner, "report", lambda *a: [])
    monkeypatch.setattr(runner, "complete", lambda r, c, k: c["model"] in done)
    monkeypatch.setattr(runner, "worker_environment", lambda *a: {})
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 1024**3))
    calls = []
    def process(command, env, log, slot, *args, **kwargs):
        job = read_json(Path(command[-1]))
        stage, name = command[-3], job["model"]["id"]
        assert job["identity"] == "preserved-run" and job["source_code"]
        calls.append((name, stage))
        if stage == "prepare":
            write_json(root / name / "_meta/prepared.json", {"status": "ready", "python": sys.executable})
        else:
            done.add(name)
        return {"exit_code": 0, "reason": ""}
    monkeypatch.setattr(colab_canary, "monitored_process", process)
    runner.run(project, {"bank": str(bank), "output": str(root), "scratch": str(scratch), "cleanup": False})
    assert calls == [(n, s) for n in names[:2] for s in ("prepare", "work")]
    assert completed_before == {p: p.read_bytes() for p in completed_before}
    for n in names[:2]:
        assert read_json(next((root / n / "_meta/status_history").glob("*.json"))) == old_status
        assert read_json(root / n / "_meta/status.json")["status"] == "diagnostics_complete"
