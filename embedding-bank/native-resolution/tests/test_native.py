"""Offline contracts and round-trip orchestration, not GPU model validation."""
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wmbench.io import read_json, write_json, sha256
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError, disjoint
from wmbench_native_canary.contracts import models_for_trial, capacity, message, score, center_crop, EXCLUDED
from wmbench_native_canary.worker import process_pair, complete, freeze_crop
from wmbench_native_canary.runner import report


def input_image(root):
    path = root / "_inputs/images/000000000001.png"
    path.parent.mkdir(parents=True)
    y, x = np.indices((512, 512))
    pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return {"image_id": 1, "path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


class TinyBits:
    """Test double: store bits in pixel LSBs; not a benchmark method."""
    def __init__(self, bad=False, fail=False):
        self.embeds, self.decodes, self.bad, self.fail = 0, 0, bad, fail

    def embed(self, source, output, payload, metadata):
        self.embeds += 1
        if self.fail:
            raise RuntimeError("test-only technical failure")
        pixels = load_rgb(source).copy()
        flat = pixels.reshape(-1)
        flat[:len(payload)] = (flat[:len(payload)] & 254) | np.asarray(payload, dtype=np.uint8)
        save_rgb(output, pixels)
        write_json(metadata, {"expected_bits": payload})

    def decode(self, source, metadata):
        self.decodes += 1
        meta = read_json(metadata)
        assert not any(meta["expected_bits"]), "Decoder received the answer"
        bits = (load_rgb(source).reshape(-1)[:len(meta["expected_bits"])] & 1).tolist()
        if self.bad:
            bits[0] ^= 1
        return {"bits": bits}


def test_catalogue_and_payloads(project):
    models = models_for_trial(project)
    assert len(models) == 32
    assert {k for k, m in models.items() if m.get("canary_exclusion")} == set(EXCLUDED)
    assert sum(not m.get("canary_exclusion") for m in models.values()) == 29
    assert capacity(models["dwt_dct_svd"])["information_bits"] == 32
    assert capacity(models["ssl_watermarking"])["information_bits"] == 30
    assert capacity(models["fnns_omniseal"])["information_bits"] == 32
    assert capacity(models["invismark"])["information_bits"] == 94
    for name, model in models.items():
        if model.get("canary_exclusion"):
            continue
        spec = capacity(model)
        assert spec["bits_per_pixel"] == spec["information_bits"] / model["native_geometry"]["width"]**2
        assignment = message(model, 139, 1)
        assert len(assignment["payload"]) == spec["adapter_bits"]
        assert assignment == message(model, 139, 1)
        assert assignment["information"] != message(model, 139, 2)["information"]
        assert score({"bits": assignment["payload"]}, assignment)["ber"] == 0
        for index, bit in spec["reserved_bits"].items():
            assert assignment["payload"][int(index)] == bit


@pytest.mark.parametrize("size", [128, 224, 256, 400, 512])
def test_crop_is_exact_pixels(size):
    pixels = np.arange(512 * 512 * 3, dtype=np.uint32).reshape(512, 512, 3).astype(np.uint8)
    crop = center_crop(Image.fromarray(pixels), size)
    low = (512 - size) // 2
    assert np.array_equal(np.asarray(crop), pixels[low:low+size, low:low+size])


@pytest.mark.parametrize("bad", [True, False])
def test_roundtrip_and_resume(tmp_path, project, bad):
    image = input_image(tmp_path)
    model = models_for_trial(project)["hidden_benchmark"]
    adapter = TinyBits(bad=bad)
    first = process_pair(adapter, model, tmp_path, image, 1, {"kind": "test-double"})
    second = process_pair(adapter, model, tmp_path, image, 1, {"kind": "test-double"})
    assert first == second
    assert adapter.embeds == adapter.decodes == 1
    assert first["exact"] is (not bad)
    assert not complete(tmp_path, model, [image])  # Only one of five messages.
    record = read_json(tmp_path / model["id"] / "_meta/records/000000000001_m01_native.json")
    assert record["size"] == 128
    assert record["pre_serialization"]["pre_png"]["status"] == "unavailable"
    for rep in range(2, 6):
        process_pair(adapter, model, tmp_path, image, rep, {})
    assert complete(tmp_path, model, [image])  # Bit errors are terminal measurements too.


def test_technical_retry_preserves_history(tmp_path, project):
    image, model = input_image(tmp_path), models_for_trial(project)["dwt_dct_svd"]
    adapter = TinyBits(fail=True)
    with pytest.raises(RuntimeError, match="test-only"):
        process_pair(adapter, model, tmp_path, image, 1, {})
    adapter.fail = False
    assert process_pair(adapter, model, tmp_path, image, 1, {})["exact"]
    assert list(tmp_path.rglob("technical_error_history/*.json"))


def test_corruption_stops_resume(tmp_path, project):
    image, model = input_image(tmp_path), models_for_trial(project)["dwt_dct_svd"]
    process_pair(TinyBits(), model, tmp_path, image, 1, {})
    path = tmp_path / "dwt_dct_svd/native/000000000001_m01.png"
    path.write_bytes(b"corrupt-test")
    with pytest.raises(EvidenceError):
        process_pair(TinyBits(), model, tmp_path, image, 1, {})


def test_changed_source_stops_crop(tmp_path, project):
    image, model = input_image(tmp_path), models_for_trial(project)["fin"]
    freeze_crop(tmp_path, model, image)
    (tmp_path / image["path"]).write_bytes(b"test-corruption")
    with pytest.raises(EvidenceError):
        freeze_crop(tmp_path, model, image)


def test_incorrect_full_decode_rejected(project):
    assignment = message(models_for_trial(project)["invismark"], 139, 1)
    with pytest.raises(ValueError):
        score({"bits": assignment["payload"][:30]}, assignment)


def test_no_nested_storage(tmp_path):
    with pytest.raises((ValueError, EvidenceError)):
        disjoint(tmp_path, tmp_path / "nested")


def test_partial_report_no_fake_pass(tmp_path, project):
    image = input_image(tmp_path)
    models = models_for_trial(project)
    write_json(tmp_path / "trial.json", {"models": models})
    write_json(tmp_path / "_inputs/manifest.json", {"images": [image]})
    process_pair(TinyBits(), models["hidden_benchmark"], tmp_path, image, 1, {})
    summary = report(tmp_path, {"n_images": 1})
    by_id = {r["model"]: r for r in summary}
    assert by_id["hidden_benchmark"]["png_exact"] == 1
    assert by_id["hidden_benchmark"]["expected"] == 5
    assert by_id["hidden_benchmark"]["status"] == "pending"
    assert by_id["omniguard"]["status"] == "excluded"
    assert by_id["omniguard"]["expected"] == 0
    assert not read_json(tmp_path / "_reports/summary.json")["all_selected_png_exact"]


def test_notebook_cells_and_schema():
    import nbformat
    path = Path(__file__).resolve().parents[1] / "dist/Embedding_Bank_NATIVE_RESOLUTION_CANARY.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert "PAYLOAD_BASE64" not in cell.source
            count += 1
    assert count == 4


def test_runtime_geometry_mismatch(project):
    from types import SimpleNamespace
    from wmbench_native_canary.adapters import validate_geometry
    model = models_for_trial(project)["fin"]
    with pytest.raises(ValueError, match="resolution"):
        validate_geometry(SimpleNamespace(resolution=512), model)
    assert validate_geometry(SimpleNamespace(resolution=128), model)["selected_carrier"] == [128, 128]


def test_infinite_psnr_reporting():
    from wmbench_native_canary.runner import mean
    assert mean(["Infinity", 42.0, None]) == "Infinity"


def test_tensor_observer_preserves_rng():
    import random
    import torch
    from wmbench_native_canary.worker import observe_boundary
    class TensorDouble:
        threshold = 0.5
        def _save(self, tensor, path):
            return tensor
        def _extract_logits(self, tensor):
            random.random()
            np.random.rand()
            torch.rand(3)
            return tensor
    adapter = TensorDouble()
    adapter.torch = torch
    assignment = {"payload": [1, 0], "capacity": {"information_bits": 2,
        "information_indices": [0, 1], "reserved_bits": {}, "interface": "raw_message"}}
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    observations = {}
    with observe_boundary(adapter, assignment, observations):
        adapter._save(torch.tensor([1., 0.]), "unused")
    assert observations["pre_png"]["exact"] is True
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert not hasattr(adapter, "before_png")
