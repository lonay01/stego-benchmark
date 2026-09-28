"""Freeze a synthetic historical bank, ensure no writes to the source bank."""
from pathlib import Path

import numpy as np
import pytest

from wmbench.io import read_json, sha256, write_json
from wmbench.matching import save_rgb
from wmbench_full_payload.storage import EvidenceError, sealed
from wmbench_native_canary.contracts import POLICY as NATIVE_POLICY, message, models_for_trial, score
from wmbench_native_canary.worker import record_path, signature, task_id
from wmbench_author_reference import core, report


def synthetic_bank(bank,project):
    models = models_for_trial(project)
    images = []
    for i in range(1,11):
        path = bank / f"_inputs/images/{i:012d}.png"
        save_rgb(path,np.full((512,512,3),i*10,dtype=np.uint8))
        images.append({"image_id":str(i),"path":path.relative_to(bank).as_posix(),"sha256":sha256(path)})
    write_json(bank / "trial.json",{"policy":NATIVE_POLICY,"models":models})
    write_json(bank / "_inputs/manifest.json",{"images":images})
    for name in core.METHODS:
        model = models[name]
        for filename in ("model_identity.json","dependency_identity.json","environment/request.json"):
            write_json(bank / name / "_meta" / filename,{"fixture":True})
        lock = bank / name / "_meta/environment/requirements.lock"
        lock.write_text("# synthetic test lock\n")
        for image in images[:5]:
            size = model["native_geometry"]["width"]
            crop = bank / name / f"crop_{image['image_id']}.png"
            save_rgb(crop,np.full((size,size,3),int(image["image_id"])*10,dtype=np.uint8))
            for rep in range(1,6):
                stem,key = task_id(image,rep),signature(model,image,rep)
                a = message(model,image["image_id"],rep)
                decoder = bank / name / f"{stem}_decoder.json"
                write_json(decoder,{"expected_bits":[0]*len(a["payload"])})
                enc = {"status":"saved","assignment":a,"source_crop":crop.relative_to(bank).as_posix(),
                       "output":crop.relative_to(bank).as_posix(),"decoder_metadata":decoder.relative_to(bank).as_posix()}
                sealed(bank,record_path(bank,model,stem,"native"),key,enc,[crop,decoder])
                dec = score({"bits":a["payload"]},a)
                sealed(bank,record_path(bank,model,stem,"native_decode"),key,dec)
    return models


def test_full_preflight_resume_175_cases_readonly(tmp_path,project):
    bank,root,scratch = tmp_path / "old",tmp_path / "new",tmp_path / "scratch"
    synthetic_bank(bank,project)
    before = {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob("*") if p.is_file()}
    settings = {"bank":str(bank),"output":str(root),"scratch":str(scratch),"n_images":5,"models":list(core.METHODS)}
    _,_,_,models,plan = core.context(project,settings)
    assert sum(map(len,plan["cases"].values())) == 175
    assert plan["holdout_ids"] == ["6","7","8","9","10"]
    assert list(models) == list(core.METHODS)
    again = core.context(project,settings)[-1]
    assert again == plan
    assert before == {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob("*") if p.is_file()}
    report.report(root)
    assert sum(r["complete"] for r in read_json(root / "_reports/summary.json")["rows"]) == 0
    settings["n_images"] = 100
    with pytest.raises(ValueError):
        core.context(project,settings)


def test_output_cannot_be_previous_diagnostic(tmp_path,project):
    root = tmp_path / "old_diagnostic"
    write_json(root / "diagnostic_identity.json",{"historical":True})
    settings = {"bank":str(tmp_path / "bank"),"output":str(root),"scratch":str(tmp_path / "scratch"),"n_images":5}
    with pytest.raises(EvidenceError):
        core.context(project,settings)
    assert not (root / "trial.json").exists()


def test_report_does_not_replace_other_protocol(tmp_path):
    write_json(tmp_path / "plan.json",{"policy":{"schema":"some-other-experiment"}})
    with pytest.raises(ValueError):
        report.report(tmp_path)
    assert not (tmp_path / "_reports/REPORT.md").exists()
