"""CPU fixtures test orchestration, not pretrained watermark quality."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError, freeze, sealed
from wmbench_native_canary.contracts import message, models_for_trial
from wmbench_author_reference import core, report, upstream, worker


def assignment(bits):
    return {"payload": bits, "capacity": {"information_bits": len(bits), "information_indices": list(range(len(bits))),
            "reserved_bits": {}, "interface": "raw_message"}}


def make_case(root, name="hidden_benchmark", bits=None, stem="test"):
    bits = [0, 1, 0, 1] if bits is None else bits
    folder = root / name / "cases" / stem
    save_rgb(folder / "source.png", np.full((8, 8, 3), 100, dtype=np.uint8))
    return {"model": name, "stem": stem, "cohort": "holdout", "seed": 8,
            "source_sha256": sha256(folder / "source.png"), "assignment": assignment(bits)}


def test_release_static_and_pinned_base(project):
    root = Path(__file__).resolve().parents[1]
    for p in (project / "src").rglob("*.py"):
        compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")
    if not (root / "dist/build.json").exists():
        assert read_json(project / "REFERENCE_BASE_RELEASE.json")["sha256"] == "b250d3bfcac3c0c1c9b27a5053a97dcb66b5a71ce1f6b34c8205be71d3d439ed"
        return
    meta = read_json(root / "dist/build.json")
    archive = root / "dist" / meta["archive"]
    assert sha256(archive) == meta["sha256"]
    assert meta["base_sha256"] == "b250d3bfcac3c0c1c9b27a5053a97dcb66b5a71ce1f6b34c8205be71d3d439ed"
    with zipfile.ZipFile(root.parent / "native-diagnostics/dist/Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY_source.zip") as z:
        for name in z.namelist():
            if name.startswith(("src/", "locks/", "registry/")):
                assert (project / name).read_bytes() == z.read(name), name
    nb = nbformat.read(root / "dist" / meta["notebook"], as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 4
    for cell in code:
        compile(cell, "notebook", "exec")
        assert "PAYLOAD_BASE64" not in cell
    assert "watermarking_author_reference_v1" in code[0]
    assert "wmbench_author_reference.runner" in code[2]
    assert "CACHED_ARCHIVE" in code[1]
    assert meta["sha256"] in code[1]


def test_reference_does_not_import_or_call_adapter(project):
    text = (project / "src/wmbench_author_reference/upstream.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("adapter" in module for module in imports)
    assert "adapter_instance" not in text


@pytest.mark.parametrize("name,size,bits", [
    ("hidden_benchmark",128,30), ("rosteals",256,100), ("sepwater",224,100),
    ("sepmark",256,128), ("distseal_posthoc_dcae",512,64)])
def test_fixed_holdout_contract(tmp_path, project, name, size, bits):
    bank, root = tmp_path / "bank", tmp_path / "new"
    original = bank / "images/10.png"
    array = np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    save_rgb(original, array)
    image = {"path": "images/10.png", "sha256": sha256(original), "image_id": "10"}
    model = models_for_trial(project)[name]
    cases = core.holdout_cases(bank, root, model, [image])
    assert len(cases) == 5
    assert len({tuple(c["assignment"]["payload"]) for c in cases}) == 5
    cut = load_rgb(core.folder_for(root, cases[0]) / "source.png")
    assert cut.shape == (size,size,3)
    offset = (512-size)//2
    assert np.array_equal(cut, array[offset:offset+size,offset:offset+size])
    assert all(len(c["assignment"]["payload"]) == bits for c in cases)
    assert core.holdout_cases(bank, root, model, [image]) == cases
    assert sha256(original) == image["sha256"]


def test_source_audit_fails_closed(tmp_path):
    data = b"x = 1\n"
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    write_json(tmp_path / "source_contracts.json", {"sepmark": {"repo": "fixture", "revision": "a"*40, "files": {"x.py": blob}}})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_bytes(data.replace(b"\n",b"\r\n"))
    assert core.verify_sources(tmp_path, {"repo_root":str(repo)}, "sepmark")["files"]["x.py"]["git_blob"] == blob
    (repo / "x.py").write_text("x = 2\n")
    with pytest.raises(EvidenceError):
        core.verify_sources(tmp_path, {"repo_root":str(repo)}, "sepmark")


def test_full_inference_weight_verification():
    net = torch.nn.Module()
    net.ae = torch.nn.Linear(3, 2)
    net.control = torch.nn.Linear(2, 2)
    net.decoder = torch.nn.Linear(2, 4)
    state = copy.deepcopy(net.state_dict())
    assert upstream.require_inference_weights(net, state)["checked"] == 6
    before = upstream.tensor_digest(net, ("ae.","control.","decoder."))
    state["decoder.bias"][0] += 1
    with pytest.raises(EvidenceError):
        upstream.require_inference_weights(net, state)
    net.load_state_dict(state)
    assert before != upstream.tensor_digest(net, ("ae.","control.","decoder."))


def test_nonzero_ber_is_complete_not_retry(tmp_path):
    case = make_case(tmp_path, name="sepmark")
    row = worker.measured({"bits":[1,1,0,1]}, case)
    assert row["ber"] == .25
    sealed(tmp_path, core.row_path(tmp_path,case,"adapter"), core.key_for("run",case,"adapter"),
           {"status":"measured","png_score":row})
    assert core.complete(tmp_path,case,"run")
    with pytest.raises(EvidenceError):
        core.get_row(tmp_path,case,"adapter","other-run")


def test_tamper_detected_before_reuse(tmp_path):
    case = make_case(tmp_path,name="sepmark")
    folder = core.folder_for(tmp_path,case)
    artifact = folder / "raw.npy"
    np.save(artifact,np.zeros((2,2)))
    sealed(tmp_path,core.row_path(tmp_path,case,"adapter"),core.key_for("run",case,"adapter"),{"status":"measured"},[artifact])
    assert core.complete(tmp_path,case,"run")
    artifact.write_bytes(b"damaged")
    with pytest.raises(EvidenceError):
        core.complete(tmp_path,case,"run")


def test_core_agreement_independent_of_success():
    comparison = {"same_bits":True,"logits":{"close":True}}
    adapter = {"status":"measured","file_api_matches_direct":True,"png_score":{"ber":.12}}
    ref = {"status":"measured","encoder_float":{"close":True},
           "decoder_on_adapter_raw":comparison,"decoder_on_adapter_png":comparison}
    cross = {"status":"measured","file_api_matches_direct":True,
             "decoder_on_reference_raw":comparison,"decoder_on_reference_png":comparison}
    assert report.agreement(adapter,ref,cross,True) == "core_agreement"
    assert report.agreement(adapter,ref,cross,False) == "review_difference"
    assert report.agreement(adapter,{},cross,True) == "incomplete"


def test_exact_and_close_are_distinct():
    x = np.ones((2,2),dtype=np.float32)
    result = core.compare_arrays(x,x+1e-7)
    assert result["close"] and not result["exact"]
    with pytest.raises(EvidenceError):
        core.compare_arrays(x,np.zeros(2))
    with pytest.raises(EvidenceError):
        core.compare_arrays(x,x*np.nan)


def test_selected_function_ignores_unrelated_cli(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("raise RuntimeError('CLI must not run')\ndef export(x):\n    return x+1\n")
    assert upstream.selected_function(path,"export",{})(2) == 3


def test_cross_decode_and_report_with_cpu_tensors(tmp_path, monkeypatch):
    """Three sealed phases, different decoder objects, deliberate one-bit error."""
    case = make_case(tmp_path)
    folder = core.folder_for(tmp_path,case)
    arr = np.full((8,8,3),.25,dtype=np.float32)
    expected = [0,1,0,1]
    # Both decoders make the same one-bit error. This must still be core agreement.
    decoded = [1,1,0,1]
    def extract(x):
        return {"bits":decoded,"logits":[.8,.8,.2,.8]}
    raw = folder / "raw.npy"
    np.save(raw,arr)
    png = folder / "base.png"
    save_rgb(png,(arr*255).astype(np.uint8))
    blind = folder / "blind.json"
    write_json(blind,{"expected_bits":[0]*4})
    result = worker.measured(extract(arr),case)
    base = {"status":"measured","raw":raw.relative_to(tmp_path).as_posix(),
            "png":png.relative_to(tmp_path).as_posix(),"blind_metadata":blind.relative_to(tmp_path).as_posix(),
            "raw_score":result,"png_score":result,"png_direct":result,"file_api_matches_direct":True,"seconds":.1}
    sealed(tmp_path,core.row_path(tmp_path,case,"adapter"),core.key_for("run",case,"adapter"),base,[raw,png,blind])
    class Ref:
        repo = folder
        def embed(self, source, payload):
            assert payload == expected
            return torch.tensor(arr).permute(2,0,1)[None]*2-1, torch.zeros(1,3,8,8)
        def decode(self, rgb):
            return extract(rgb)
    def export(name,repo,raw,destination,**kwargs):
        save_rgb(destination,np.full((8,8,3),63,dtype=np.uint8))
    monkeypatch.setattr(upstream,"serialize_author",export)
    attempt = worker.fresh_attempt(folder,"reference")
    ref = worker.reference_case(Ref(),tmp_path,case,attempt,"run")
    sealed(tmp_path,core.row_path(tmp_path,case,"reference"),core.key_for("run",case,"reference"),
           {**ref,"status":"measured","seconds":.1},list(attempt.iterdir()))
    class Adapter:
        def decode(self,path,metadata):
            assert not any(read_json(metadata)["expected_bits"])
            return extract(load_rgb(path))
    monkeypatch.setattr(worker,"decode_float",lambda adapter,name,array: extract(array))
    cross = worker.cross_case(Adapter(),tmp_path,case,"run")
    sealed(tmp_path,core.row_path(tmp_path,case,"cross"),core.key_for("run",case,"cross"),
           {**cross,"status":"measured","seconds":.1})
    assert core.complete(tmp_path,case,"run")
    write_json(tmp_path / "plan.json",{"identity":"run","cases":{"hidden_benchmark":[case]},"policy":core.POLICY,"holdout_ids":[]})
    write_json(tmp_path / "hidden_benchmark/_meta/weight_agreement.json",{"identical_inference_tensors":True})
    report.report(tmp_path)
    row = read_json(tmp_path / "_reports/summary.json")["rows"][0]
    assert row["core_agreement"] == 1 and row["adapter"]["ber"] == .25
    assert "not zero BER" in (tmp_path / "_reports/REPORT.md").read_text()


def test_pending_report_not_false_success(tmp_path):
    case = make_case(tmp_path,name="sepmark")
    write_json(tmp_path / "plan.json",{"identity":"run","cases":{"sepmark":[case]},"policy":core.POLICY,"holdout_ids":[1]})
    report.report(tmp_path)
    row = read_json(tmp_path / "_reports/summary.json")["rows"][0]
    assert row["complete"] == 0 and row["adapter"]["ber"] is None


def test_cli_report_in_fresh_process(tmp_path, project):
    case = make_case(tmp_path,name="sepmark")
    write_json(tmp_path / "plan.json",{"identity":"run","cases":{"sepmark":[case]},"policy":core.POLICY,"holdout_ids":[]})
    config = tmp_path / "settings.json"
    write_json(config,{"output":str(tmp_path)})
    env = {**os.environ,"PYTHONPATH":str(project / "src")}
    completed = subprocess.run([sys.executable,"-m","wmbench_author_reference.runner","report","--settings",str(config)],
                               env=env,capture_output=True,text=True)
    assert completed.returncode == 0, completed.stderr
