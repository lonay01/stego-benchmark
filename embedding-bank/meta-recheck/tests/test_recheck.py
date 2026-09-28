"""CPU orchestration tests; pretrained smoke tests are separate and optional."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import nbformat
import numpy as np
from PIL import Image
import pytest

from wmbench.io import digest, read_json, sha256, write_json
from wmbench.matching import save_rgb, load_rgb
from wmbench_full_payload.storage import EvidenceError, sealed
from wmbench_full_payload.contracts import message as payload_message
from wmbench_native_canary.contracts import POLICY, models_for_trial, message, score
from wmbench_native_canary.worker import record_path, signature, task_id
from wmbench_author_reference import core as reference_core
from wmbench_author_reference.report import report as reference_report
from wmbench_meta_recheck import meta, runner, report


def synthetic_bank(bank, project):
    models = models_for_trial(project)
    images = []
    for index in range(10):
        pixels = np.random.default_rng(index).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        path = bank / f"_inputs/images/{index+1:012d}.png"
        save_rgb(path, pixels)
        images.append({"image_id": str(index+1), "path": path.relative_to(bank).as_posix(), "sha256": sha256(path)})
    write_json(bank / "_inputs/manifest.json", {"images": images})
    write_json(bank / "trial.json", {"policy": POLICY, "models": models})
    for name in ("rosteals", "sepwater"):
        model = models[name]
        for filename in ("model_identity.json", "dependency_identity.json", "environment/request.json"):
            write_json(bank / name / "_meta" / filename, {"fixture": True})
        (bank / name / "_meta/environment/requirements.lock").write_text("# fixture\n")
        size = model["native_geometry"]["width"]
        for image in images[:5]:
            array = load_rgb(bank / image["path"])
            offset = (512-size)//2
            path = bank / name / f"crop_{image['image_id']}.png"
            save_rgb(path, array[offset:offset+size, offset:offset+size])
            for rep in range(1, 6):
                stem, key = task_id(image,rep), signature(model,image,rep)
                a = message(model,image["image_id"],rep)
                decoder = bank / name / f"{stem}.json"
                write_json(decoder, {"expected_bits": [0]*100})
                enc = {"status": "saved", "assignment": a, "source_crop": path.relative_to(bank).as_posix(),
                       "output": path.relative_to(bank).as_posix(), "decoder_metadata": decoder.relative_to(bank).as_posix()}
                sealed(bank, record_path(bank,model,stem,"native"), key, enc, [path,decoder])
                sealed(bank, record_path(bank,model,stem,"native_decode"), key, score({"bits":a["payload"]},a))


def settings(bank,root,scratch):
    return {"bank":str(bank),"output":str(root),"scratch":str(scratch),"n_images":5,
            "models":["meta_hidden_revisited","rosteals","sepwater"], "device":"cpu", "max_hours":6}


def test_preflight_all_100_cases_readonly_resume(tmp_path,project):
    bank,root,scratch = (tmp_path / x for x in ("bank","new","scratch"))
    synthetic_bank(bank,project)
    before = {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob("*") if p.is_file()}
    config = settings(bank,root,scratch)
    runner.paths(project,config)
    plan = meta.context(project,bank,root / "meta_hidden")
    assert len(plan["cases"]) == 50
    ref = {**config,"output":str(root / "references"),"models":["rosteals","sepwater"]}
    rplan = reference_core.context(project,ref)[-1]
    assert sum(map(len,rplan["cases"].values())) == 50
    assert all(len(c["assignment"]["payload"]) == 100 for cs in rplan["cases"].values() for c in cs)
    assert meta.context(project,bank,root / "meta_hidden") == plan
    assert reference_core.context(project,ref)[-1] == rplan
    for case in plan["cases"]:
        a = case["assignment"]
        assert len(a["payload"]) == 48
        assert a["payload"] == payload_message({"id":"meta_hidden_revisited","native_bits":48},case["image_id"],case["replicate"])["payload"]
        size = meta.PROFILES[case["profile"]]["size"]
        assert a["capacity"]["bits_per_pixel"] == 48 / size**2
        saved = load_rgb(meta.case_folder(root / "meta_hidden",case) / "source.png")
        original = load_rgb(bank / f"_inputs/images/{int(case['image_id']):012d}.png")
        offset = (512-size)//2
        assert np.array_equal(saved, original[offset:offset+size,offset:offset+size])
    assert before == {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob("*") if p.is_file()}
    reference_report(root / "references")
    report.report(root)
    summary = read_json(root / "_reports/summary.json")
    assert sum(r["complete"] for r in summary["meta"]) == 0
    assert sum(r["complete"] for r in summary["references"]["rows"]) == 0
    archive = report.review_zip(root)
    with zipfile.ZipFile(archive) as z:
        assert all(not n.endswith((".png",".npy",".pth")) for n in z.namelist())
    config_path = tmp_path / "settings.json"
    write_json(config_path,config)
    child = subprocess.run([sys.executable,"-m","wmbench_meta_recheck.runner","report","--settings",str(config_path)],
        env={**os.environ,"PYTHONPATH":str(project / "src")},capture_output=True,text=True)
    assert child.returncode == 0,child.stderr


def test_static_notebook_and_base_preserved(project):
    for p in (project / "src").rglob("*.py"):
        compile(p.read_text(encoding="utf-8-sig"),str(p),"exec")
    root = Path(__file__).resolve().parents[1]
    if not (root / "dist/build.json").exists():
        return
    release = read_json(root / "dist/build.json")
    assert sha256(root / "dist" / release["archive"]) == release["sha256"]
    with zipfile.ZipFile(root.parent / "author-reference/dist/Embedding_Bank_AUTHOR_REFERENCE_source.zip") as z:
        for n in z.namelist():
            if n.startswith(("src/","locks/","registry/")):
                assert (project / n).read_bytes() == z.read(n)
    nb = nbformat.read(root / "dist" / release["notebook"],as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 4
    for c in code:
        compile(c,"notebook","exec")
        assert "PAYLOAD_BASE64" not in c
    assert release["sha256"] in code[1]
    assert release["base_sha256"] not in code[1]
    assert "wmbench_meta_recheck.runner" in code[2]
    assert "watermarking_meta_sepwater_rosteals_v1" in code[0]


def test_wrong_output_or_scope_rejected(tmp_path,project):
    root = tmp_path / "old"
    write_json(root / "trial.json",{"old":True})
    conf = settings(tmp_path / "bank",root,tmp_path / "scratch")
    with pytest.raises(EvidenceError):
        runner.paths(project,conf)
    assert read_json(root / "trial.json") == {"old":True}
    conf["output"] = str(tmp_path / "new")
    conf["n_images"] = 100
    with pytest.raises(ValueError):
        runner.paths(project,conf)
    conf["n_images"] = 5
    conf["output"] = conf["bank"]
    with pytest.raises(ValueError):
        runner.paths(project,conf)


def test_nonzero_ber_is_complete_and_corruption_stops_resume(tmp_path):
    case = {"profile":"checkpoint_256","stem":"x"}
    path = tmp_path / "checkpoint_256/x/source.png"
    save_rgb(path,np.ones((4,4,3),dtype=np.uint8))
    case["source_sha256"] = sha256(path)
    artifact = path.parent / "raw.npy"
    np.save(artifact,np.zeros((4,4,3)))
    sealed(tmp_path,path.parent / "measurement.json",digest({"trial":"run","case":case}),
           {"status":"measured","ber":.5},[artifact])
    assert meta.complete(tmp_path,case,"run")
    with pytest.raises(EvidenceError):
        meta.row(tmp_path,case,"other")
    artifact.write_bytes(b"damaged")
    with pytest.raises(EvidenceError):
        meta.complete(tmp_path,case,"run")


def test_rosteals_real_import_shape_in_fresh_process(tmp_path,project):
    """Regress the exact RoSteALS.* namespace failure, not GPU/model execution."""
    repo = tmp_path / "sources/wibe"
    package = repo / "submodules/RoSteALS"
    package.mkdir(parents=True)
    (repo / "submodules/taming-transformers").mkdir()
    (package / "__init__.py").write_text("")
    (package / "fixture.py").write_text("class Model:\n    value = 100\n")
    code = ("from wmbench_meta_recheck.reference import rosteals_import_paths; import importlib; "
            f"rosteals_import_paths({{'repo_root':{str(repo)!r}}}); "
            "assert importlib.import_module('RoSteALS.fixture').Model.value == 100")
    child = subprocess.run([sys.executable,"-c",code],env={**os.environ,"PYTHONPATH":str(project / "src")},
                           capture_output=True,text=True)
    assert child.returncode == 0,child.stderr
    dispatch = (project / "src/wmbench_meta_recheck/reference_runner.py").read_text()
    assert "wmbench_meta_recheck.reference_runner" in dispatch
    assert "from .reference import work" in dispatch


def test_meta_asset_hash_guard(tmp_path):
    raw = b"reviewed source\n"
    path = tmp_path / "source.py"
    path.write_bytes(raw)
    contract = {"files":{"source.py":{"size":len(raw),"git_blob":meta.blob(raw)}}}
    meta.verify_asset(path,"source.py",contract)
    path.write_bytes(b"different source\n")
    with pytest.raises(EvidenceError):
        meta.verify_asset(path,"source.py",contract)


def test_official_profiles_not_data_selected():
    assert meta.PROFILES == {"checkpoint_256":{"size":256,"jnd":False,"scaling_w":.3},
                             "demo_512":{"size":512,"jnd":True,"scaling_w":1.5}}
    assert meta.MODEL["native_bits"] == 48
    assert not meta.POLICY["attacks"] and not meta.POLICY["training"]


def test_worker_retry_and_resume_keeps_measured_errors(tmp_path,project,monkeypatch):
    root = tmp_path / "results"
    source = root / "demo_512/one/source.png"
    save_rgb(source,np.full((512,512,3),100,dtype=np.uint8))
    assignment = payload_message(meta.MODEL,"1",1)
    case = {"profile":"demo_512","stem":"one","assignment":assignment,"source_sha256":sha256(source)}
    plan = {"identity":"fixture","cases":[case]}
    write_json(root / "_meta/prepared.json",{"device":"cpu","assets":str(tmp_path / "assets")})
    monkeypatch.setattr(meta,"verify_asset",lambda *args: None)
    monkeypatch.setattr(meta,"runtime_info",lambda: {"device":"cpu-fixture"})
    calls = []
    class Fixture:
        weights = {"fixture":True}
        training_config = {"fixture":True}
        def __init__(self,*args):
            pass
        def measure(self,source,payload,profile,attempt):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("synthetic interrupted measurement")
            source.save(attempt / "fixture.png")
            bits = list(payload)
            bits[0] ^= 1
            value = {"bits":bits,"logits":[1.0 if b else -1.0 for b in bits]}
            return {"decoded":{s:value for s in ("raw","clipped","round8_memory","png","clean_control")},
                    "psnr_original_to_png":30.0}
    monkeypatch.setattr(meta,"MetaHidden",Fixture)
    with pytest.raises(RuntimeError,match="technical errors"):
        meta.work(project,root,plan)
    assert not meta.complete(root,case,"fixture")
    meta.work(project,root,plan)
    assert meta.complete(root,case,"fixture")
    value = meta.row(root,case,"fixture")
    assert value["stages"]["png"]["ber"] == 1/48
    assert not value["stages"]["png"]["exact"]
    meta.work(project,root,plan)
    assert len(calls) == 2
    assert len(list((source.parent / "technical_error_history").glob("*.json"))) == 1
