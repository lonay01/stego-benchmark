"""Offline contracts and failure/restart tests. Stubs do not validate model quality."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pytest

from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.contracts import POLICY, capacity, message, models_for_trial, score
from wmbench_full_payload.runner import cleanup, complete, context, prepare, report
from wmbench_full_payload.storage import EvidenceError, cached, disjoint, freeze, inside, sealed
from wmbench_full_payload.worker import FullStages, process_pair, record_path, signature, task_id


def model(bits=64, reserved=None):
    return {"id": "stub", "native_bits": bits, "reserved_bits": reserved or {}}


class PixelAdapter:
    """The reference is not available to decode; bits really round-trip via PNG."""
    def __init__(self, bits=64, unchanged=False, failures=0):
        self.bits, self.unchanged, self.failures = bits, unchanged, failures
        self.embeds = self.decodes = 0

    def embed(self, source, target, payload, metadata):
        self.embeds += 1
        image = load_rgb(source)
        if not self.unchanged:
            # Distributed, varying residual makes all three scalar targets attainable.
            residual = (np.arange(image.size).reshape(image.shape) % 201 - 100)
            image = np.clip(image.astype(np.int16) + residual, 0, 255).astype(np.uint8)
            flat = image.reshape(-1)
            flat[:self.bits] = (flat[:self.bits] // 2) * 2 + payload
        save_rgb(target, image)
        write_json(metadata, {"expected_bits": payload, "native_bits": self.bits})

    def decode(self, source, metadata):
        self.decodes += 1
        assert read_json(metadata)["expected_bits"] == [0] * self.bits
        if self.failures:
            self.failures -= 1
            raise RuntimeError("simulated decoder failure")
        bits = (load_rgb(source).reshape(-1)[:self.bits] % 2).tolist()
        return {"bits": bits, "details": None}


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / "_inputs/images/000000000139.png"
    save_rgb(source, np.full((512, 512, 3), 128, dtype=np.uint8))
    return tmp_path, {"image_id": 139, "path": source.relative_to(tmp_path).as_posix(), "sha256": sha256(source)}


def test_all_catalogue_contracts(project):
    models = models_for_trial(project)
    assert len(models) == 32
    assert capacity(models["fnns_omniseal"])["information_bits"] == 32
    assert capacity(models["invismark"])["information_bits"] == 94
    assert capacity(models["dwt_dct_svd"])["information_bits"] == 4096
    assert capacity(models["ssl_watermarking"])["information_bits"] == 2048
    assert capacity(models["fang_screen"])["coded_bits"] == 64
    for entry in models.values():
        spec = capacity(entry)
        if spec["information_bits"]:
            assignment = message(entry, 139, 1)
            assert score({"bits": assignment["payload"]}, assignment)["exact"]


@pytest.mark.parametrize("bits", [30, 32, 48, 64, 94, 100, 256, 2048, 4096])
def test_full_message_not_repeated_30(bits):
    value = message(model(bits), 139, 1)
    assert len(value["information"]) == bits
    assert value == message(model(bits), 139, 1)
    assert value["information"] != message(model(bits), 139, 2)["information"]
    assert value["information"] == message(model(4096), 139, 1)["information"][:bits]
    if bits >= 64:
        assert value["information"][:30] != value["information"][30:60]


def test_reserved_and_all_information_scoring():
    assigned = message(model(100, {"0": 1, "99": 0}), 139, 1)
    decoded = assigned["payload"].copy()
    for index in (0, 1, 50, 98):
        decoded[index] ^= 1
    result = score({"bits": decoded, "details": None}, assigned)
    assert result["information_errors"] == 3
    assert result["ber"] == 3 / 98
    assert result["reserved_errors"] == [0]
    assert not result["exact"]


@pytest.mark.parametrize("bits", [[], [0] * 63, [0] * 65, [2] * 64, [float("nan")] * 64])
def test_bad_decode_not_reported_as_ber(bits):
    with pytest.raises(ValueError):
        score({"bits": bits}, message(model(), 139, 1))


def test_ecc_rejection_is_not_success_or_fabricated_ber():
    assigned = message(model(), 139, 1)
    result = score({"bits": assigned["payload"], "details": {"ecc_valid": False}}, assigned)
    assert result["status"] == "rejected" and result["ber"] is None and result["exact"] is False


def test_pair_png_roundtrip_matching_resume(sample):
    root, image = sample
    adapter, entry = PixelAdapter(), model()
    rows = process_pair(adapter, entry, root, image, 1, {"session": "a"})
    assert len(rows) == 4 and all(row["status"] == "measured" for row in rows)
    assert rows[0]["exact"]
    assert adapter.embeds == 1 and adapter.decodes == 4
    record = record_path(root, entry, task_id(image, 1), "native")
    before = record.read_bytes()
    assert read_json(record)["pre_serialization"]["pre_png"]["status"] == "unavailable"
    for target in (39, 42, 45):
        item = read_json(record_path(root, entry, task_id(image, 1), f"matched_{target}"))
        assert abs(item["psnr"] - target) <= POLICY["tolerance"]
    process_pair(adapter, entry, root, image, 1, {"session": "b"})
    assert (adapter.embeds, adapter.decodes) == (1, 4)
    assert record.read_bytes() == before


def test_decode_failure_retries_only_that_stage(sample):
    root, image = sample
    adapter, entry = PixelAdapter(failures=1), model()
    rows = process_pair(adapter, entry, root, image, 1, {})
    assert rows[0]["status"] == "error" and rows[0]["ber"] is None
    rows = process_pair(adapter, entry, root, image, 1, {})
    assert rows[0]["exact"] and (adapter.embeds, adapter.decodes) == (1, 5)
    assert len(list((root / "stub/_meta/records/technical_error_history").glob("*.json"))) == 1


def test_nonzero_ber_not_retried_and_unreachable_not_published(sample):
    root, image = sample
    adapter, entry = PixelAdapter(unchanged=True), model()
    rows = process_pair(adapter, entry, root, image, 1, {})
    assert rows[0]["ber"] > 0
    assert all(row["status"] == "unmatched" for row in rows[1:])
    assert not list((root / "stub").glob("matched_*/*.png"))
    process_pair(adapter, entry, root, image, 1, {})
    assert (adapter.embeds, adapter.decodes) == (1, 1)


def test_corrupt_saved_image_never_silently_reencoded(sample):
    root, image = sample
    adapter, entry = PixelAdapter(), model()
    process_pair(adapter, entry, root, image, 1, {})
    png = next((root / "stub/native").glob("*.png"))
    png.write_bytes(b"corrupt")
    with pytest.raises(EvidenceError, match="Damaged"):
        process_pair(adapter, entry, root, image, 1, {})
    assert adapter.embeds == 1 and png.read_bytes() == b"corrupt"


def test_changed_task_identity_does_not_resume(sample):
    root, image = sample
    adapter = PixelAdapter()
    process_pair(adapter, model(), root, image, 1, {})
    with pytest.raises(EvidenceError, match="Changed task identity"):
        process_pair(adapter, model(100), root, image, 1, {})


def test_freeze_paths_and_seals(tmp_path):
    freeze(tmp_path / "one.json", {"a": 1})
    with pytest.raises(EvidenceError):
        freeze(tmp_path / "one.json", {"a": 2})
    with pytest.raises(EvidenceError):
        inside(tmp_path, "../outside")
    with pytest.raises(ValueError):
        disjoint(tmp_path, tmp_path / "nested")
    sealed(tmp_path, tmp_path / "seal.json", "abc", {"status": "measured"}, [tmp_path / "one.json"])
    assert cached(tmp_path, tmp_path / "seal.json", "abc")
    with pytest.raises(EvidenceError):
        cached(tmp_path, tmp_path / "seal.json", "changed")


def test_unsafe_cleanup_rejected(tmp_path):
    with pytest.raises(EvidenceError):
        cleanup(tmp_path, tmp_path, "owner")


def test_owned_cleanup_preserves_results_and_slot_logs(tmp_path):
    scratch = tmp_path / "scratch"
    slot = scratch / "slots/stub"
    freeze(slot / ".full-payload-owner.json", {"owner": "ok"})
    (slot / "assets").mkdir()
    (slot / "assets/cache.bin").write_bytes(b"regenerable")
    (slot / "keep.log").write_text("log")
    results = tmp_path / "results.png"
    results.write_bytes(b"durable")
    cleanup(slot, scratch, "ok")
    assert not (slot / "assets").exists()
    assert (slot / "keep.log").exists() and results.read_bytes() == b"durable"


def test_context_copies_inputs_without_changing_old_bank(tmp_path, project, monkeypatch):
    from wmbench_full_payload import runner
    bank, output, scratch = [tmp_path / name for name in ("bank", "out", "scratch")]
    image = bank / "_inputs/images/139.png"
    save_rgb(image, np.full((512, 512, 3), 128, dtype=np.uint8))
    data = {"content_hash": "stub", "images": [{"image_id": 139, "path": "_inputs/images/139.png", "sha256": sha256(image)}]}
    write_json(bank / "_inputs/manifest.json", data)
    (bank / "_inputs/legacy_manifest.csv").write_text("stub manifest")
    snapshot = {p.relative_to(bank).as_posix(): p.read_bytes() for p in bank.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "freeze_inputs", lambda *a: data)
    settings = {"bank": str(bank), "output": str(output), "scratch": str(scratch), "n_images": 5, "models": ["fin"]}
    ctx = context(project, settings)
    assert ctx["selected"] == ["fin"] and len(ctx["models"]) == 32
    assert sha256(output / data["images"][0]["path"]) == sha256(image)
    assert context(project, {**settings, "n_images": 100})["identity"] == ctx["identity"]
    assert snapshot == {p.relative_to(bank).as_posix(): p.read_bytes() for p in bank.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        context(project, {**settings, "output": str(bank)})


def test_prepare_restores_lock_and_records_evidence(tmp_path, monkeypatch):
    from wmbench_full_payload import runner
    entry = model()
    output, slot = tmp_path / "output", tmp_path / "slot"
    old = output / "stub/_meta/environment"
    old.mkdir(parents=True)
    (old / "requirements.lock").write_text("numpy==1.26.4\n")
    write_json(old / "request.json", {"sha256": "fixture"})
    monkeypatch.setattr(runner, "ensure_manual_assets", lambda job: None)
    monkeypatch.setattr(runner, "prepare_model", lambda *args: {"python": "python", "identity": {"a": 1}})
    prepare({"model": entry, "output": str(output), "slot": str(slot), "project": str(tmp_path),
             "settings": {"device": "cpu"}, "session": "fixture"})
    assert (slot / "locks/stub/requirements.lock").read_bytes() == (old / "requirements.lock").read_bytes()
    assert read_json(output / "stub/_meta/prepared.json")["status"] == "ready"


def test_prepare_blocker_does_not_initialize_model(tmp_path, monkeypatch):
    from wmbench_full_payload import runner
    monkeypatch.setattr(runner, "ensure_manual_assets", lambda job: "Missing matching checkpoint")
    monkeypatch.setattr(runner, "prepare_model", lambda *args: pytest.fail("Blocked prerequisite"))
    prepare({"model": model(), "output": str(tmp_path), "slot": str(tmp_path / "slot"),
             "project": str(tmp_path), "settings": {}, "session": "fixture"})
    assert read_json(tmp_path / "stub/_meta/prepared.json")["status"] == "blocked"


def test_full_stages_preserves_diagnostics(tmp_path):
    assigned = message(model(), 139, 1)
    stages = FullStages(tmp_path, assigned, "abc")
    row = stages.measure("float", lambda: {"bits": assigned["payload"], "native_hw": [128, 128], "logits": [0.5] * 64})
    assert row["exact"] and row["information_bits"] == 64 and row["native_hw"] == [128, 128]
    assert len(row["logits"]) == 64
    again = stages.measure("float", lambda: pytest.fail("Completed observation must not be recomputed"))
    assert row == again


def test_report_denominators_and_completion(sample):
    root, image = sample
    entry = model()
    write_json(root / "trial.json", {"models": {"stub": entry}})
    write_json(root / "_inputs/manifest.json", {"images": [image]})
    adapter = PixelAdapter(unchanged=True)
    for replicate in range(1, 6):
        process_pair(adapter, entry, root, image, replicate, {})
    assert complete(root, entry, [image])
    result = report(root, {"n_images": 1, "models": "all"})
    assert result[0]["native_expected"] == 5
    assert result[0]["matched_unavailable"] == 15
    assert result[0]["terminal_modes"] == 20
    assert result[0]["status"] == "completed_with_native_bit_errors_or_rejections"
    assert (root / "_reports/modes.csv").is_file()


def test_notebook_and_archive_are_reproducible(project):
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root / "dist/build.json")
    archive_path = root / "dist" / manifest["archive"]
    assert sha256(archive_path) == manifest["sha256"]
    notebook = nbformat.read(root / "dist" / manifest["notebook"], as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert len(cells) == 4
    assert manifest["sha256"] in cells[1].source
    for cell in cells:
        ast.parse(cell.source)
        assert "PAYLOAD_BASE64" not in cell.source
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == manifest["files"][name]
            if name.endswith(".py"):
                ast.parse(data.decode("utf-8-sig"))
            if name.startswith("src/wmbench/"):
                assert data == (root.parent / name).read_bytes()
    assert (project / "src/bank_diagnostics/worker.py").is_file()


def test_controller_imports_no_heavy_model_runtime(project):
    code = ("import sys; from pathlib import Path; import wmbench_full_payload.runner; "
            "from wmbench.colab_canary import controller_lease, monitored_process; "
            "from wmbench_full_payload.contracts import models_for_trial; "
            "assert len(models_for_trial(Path(sys.argv[1]))) == 32; "
            "assert not {'torch', 'tensorflow', 'diffusers', 'onnxruntime'} & set(sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    subprocess.run([sys.executable, "-c", code, str(project)], env=env, check=True, timeout=30)
