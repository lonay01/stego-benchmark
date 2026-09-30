"""CPU contracts: no fake GPU measurements and no edits to parent evidence."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from authornight.storage import EvidenceError, cached, commit, digest, write
from authorrecovery import migration
from authorrecovery.ecc import CorrectedBCH, evaluate, selftest
from authorrecovery.report import report, bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    parent, output, scratch = [tmp_path / name for name in ("parent", "output", "scratch")]
    plan = {"code_sha256": migration.PARENT_RELEASE, "images": [{"image_id": i} for i in range(5)],
            "profiles": {m: ["raw100", "author_bch56"] for m in migration.METHODS}, "messages_per_image": 5}
    plan["identity"] = digest(plan)
    monkeypatch.setattr(migration, "PARENT_ID", plan["identity"])
    write(parent / "plan.json", plan)
    settings = dict(parent=str(parent), output=str(output), scratch=str(scratch), max_hours=1)
    return parent, output, settings, plan


def sample_row():
    codec = CorrectedBCH()
    data = codec.bits(b"secrets")
    payload = codec.encode(data)
    predicted = list(payload)
    predicted[5] ^= 1
    score = {"bits": predicted, "ber": .01, "exact": False, "author_ecc_text_exact": False,
             "nbits": 100, "errors": 1, "word_accuracy_threshold_0_2": True}
    return {"method": "sepwater", "profile": "author_bch56", "image_id": 0, "repetition": 0,
            "payload": payload, "measurements": {"author_png": score, "raw_internal": score}, "quality": {},
            "checks": {"png_pixels_equal": True}}


def test_bch_all_guaranteed_cases():
    assert selftest()["total_trials"] == 1300


def test_encoder_exact_author_packet():
    import bchlib
    codec = CorrectedBCH()
    text = b"secrets"
    bch = bchlib.BCH(5, 137)
    assert codec.encode(codec.bits(text)) == codec.bits(text + bch.encode(text)) + [0]*4


@pytest.mark.parametrize("bad", [[], [0]*99, [0]*101, [2]*100])
def test_bad_received_packet_rejected(bad):
    with pytest.raises(ValueError):
        CorrectedBCH().decode(bad)


def test_ecc_does_not_mutate_predictions_or_payload():
    row = sample_row()
    row["record_sha256"] = "test"
    before = json.dumps(row, sort_keys=True)
    result = evaluate(row, CorrectedBCH())
    assert result["stages"]["author_png"]["data_exact"]
    assert json.dumps(row, sort_keys=True) == before
    assert result["stages"]["author_png"]["raw_payload_ber"] == .01


def test_wrong_encoder_is_detected():
    row = sample_row()
    row["record_sha256"] = "test"
    row["payload"][90] ^= 1
    with pytest.raises(EvidenceError):
        evaluate(row, CorrectedBCH())


def test_plan_bound_to_parent_and_code(experiment):
    parent, output, settings, prior = experiment
    plan = migration.make_plan(settings, "newrelease")
    assert plan["parent_identity"] == prior["identity"]
    assert set(plan["profiles"]) == {"rosteals", "sepwater"}
    assert migration.make_plan({**settings, "max_hours": 8}, "newrelease") == plan
    with pytest.raises(EvidenceError):
        migration.make_plan(settings, "differentrelease")


def test_parent_is_not_output(experiment):
    parent, output, settings, prior = experiment
    with pytest.raises(EvidenceError):
        migration.make_plan({**settings, "output": str(parent)}, "new")


def test_parent_plan_tamper_detected(experiment):
    parent, output, settings, prior = experiment
    prior["messages_per_image"] = 3
    write(parent / "plan.json", prior)
    with pytest.raises(EvidenceError):
        migration.make_plan(settings, "new")


def test_import_retains_parent_and_provenance(experiment):
    parent, output, settings, prior = experiment
    plan = migration.make_plan(settings, "new")
    artifact = parent / "sepwater/coco/artifacts/test.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"saved-image-bytes")
    relative = "sepwater/coco/records/test_author_bch56.json"
    commit(parent, relative, prior["identity"], sample_row(), [artifact])
    before = {p.relative_to(parent): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert migration.import_record(parent, output, relative, plan["identity"])
    imported = cached(output, relative, plan["identity"])
    assert imported["imported_from"]["identity"] == prior["identity"]
    assert not migration.import_record(parent, output, relative, plan["identity"])
    assert before == {p.relative_to(parent): p.read_bytes() for p in parent.rglob('*') if p.is_file()}


def test_bad_artifact_does_not_commit_record(experiment):
    parent, output, settings, prior = experiment
    plan = migration.make_plan(settings, "new")
    artifact = parent / "artifact.zip"
    artifact.write_bytes(b"original")
    relative = "sepwater/coco/records/test_author_bch56.json"
    commit(parent, relative, prior["identity"], sample_row(), [artifact])
    artifact.write_bytes(b"changed")
    with pytest.raises(EvidenceError):
        migration.import_record(parent, output, relative, plan["identity"])
    assert not (output / relative).exists()
    assert not list(output.glob(".import-*"))


def test_resumed_attack_grid_is_not_overwritten(experiment):
    parent, output, settings, prior = experiment
    plan = migration.make_plan(settings, "new")
    relative = "sepwater/corruptions/test.json"
    commit(parent, relative, prior["identity"], {"scores": []})
    migration.import_record(parent, output, relative, plan["identity"])
    imported = cached(output, relative, plan["identity"])
    imported["scores"] = [{"attack": "test", "severity": 1}]
    body = {k: v for k, v in imported.items() if k not in ("record_sha256", "saved_unix", "identity", "artifacts")}
    commit(output, relative, plan["identity"], body)
    migration.import_record(parent, output, relative, plan["identity"])
    assert len(cached(output, relative, plan["identity"])["scores"]) == 1


def test_partial_report_never_claims_completion(experiment):
    parent, output, settings, prior = experiment
    plan = migration.make_plan(settings, "new")
    commit(output, "sepwater/coco/records/test_author_bch56.json", plan["identity"], sample_row())
    scores = [{"attack": "noise", "severity": 1, "status": "measured", "errors": 1, "nbits": 100}]
    commit(output, "sepwater/corruptions/test.json", plan["identity"], {"scores": scores})
    summary = report(output)
    assert not summary["complete"]
    assert summary["attacks"]["sepwater"]["complete_grids"] == 0
    assert summary["attacks"]["sepwater"]["measured_tests"] == 1
    assert "distseal" not in summary["models"]
    assert summary["models"]["rosteals"]["raw100"]["measurements"] == {}
    with zipfile.ZipFile(bundle(output)) as archive:
        assert "_reports/summary.json" in archive.namelist()


def test_copy_can_resume_and_rejects_changed_target(tmp_path):
    source, target = tmp_path / "a.bin", tmp_path / "b.bin"
    source.write_bytes(b"abc")
    checksum = hashlib.sha256(b"abc").hexdigest()
    migration.copy_verified(source, target, checksum)
    migration.copy_verified(source, target, checksum)
    target.write_bytes(b"xyz")
    with pytest.raises(EvidenceError):
        migration.copy_verified(source, target, checksum)


def test_worker_rejects_distseal():
    from authorrecovery.worker import main
    with pytest.raises(ValueError):
        main({"method": "distseal", "phase": "generation"})


def test_distributed_contracts():
    if not (ROOT.parent / "author-night/dist/Three_Method_AUTHOR_NIGHT_source.zip").exists():
        pytest.skip("Source rebuild contract requires the frozen parent release")
    from build_notebook import inherited_files
    files = inherited_files()
    assert b'job.get("worker_module", "authornight.worker")' in files["authornight/runner.py"]
    assert b'originals[:job.get("image_limit", len(originals))]' in files["authornight/latent.py"]
    old = {line for line in files["locks/rosteals.txt"].decode().splitlines() if "==" in line}
    new = {line for line in (ROOT / "locks/rosteals.txt").read_text().splitlines() if "==" in line}
    assert new - old == {"slackclient==2.9.3"}
    assert old.issubset(new)
