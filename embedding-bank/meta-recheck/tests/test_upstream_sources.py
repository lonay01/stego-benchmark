"""Source-level CPU smoke for the two historical methods; no pretrained claims."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from wmbench.matching import load_rgb
from wmbench_author_reference.upstream import serialize_author


@pytest.fixture
def sources():
    path = os.environ.get("WMBENCH_UPSTREAM_FIXTURES")
    if not path:
        pytest.skip("Set WMBENCH_UPSTREAM_FIXTURES to the reviewed upstream source folder")
    return Path(path)


@pytest.mark.parametrize("name",["rosteals","sepwater"])
def test_author_residual_export(tmp_path,sources,name):
    raw = torch.linspace(-.1,1.1,3*8*8).reshape(1,3,8,8)
    source = np.full((8,8,3),100,dtype=np.uint8)
    cover = torch.from_numpy(source.copy()).permute(2,0,1)[None].float()/127.5-1
    signed = raw*2-1
    target = tmp_path / "out.png"
    serialize_author(name,sources / name,raw,target,signed=signed,cover=cover,source=source)
    residual = (signed.clamp(-1,1)-cover)[0].permute(1,2,0).numpy()
    expected = (np.clip(residual+source/127.5-1.,-1,1)*127.5+127.5).astype(np.uint8)
    assert np.array_equal(load_rgb(target),expected)


def test_rosteals_factory_after_import_repair(tmp_path,sources,project):
    repo = tmp_path / "wibe"
    package = repo / "submodules/RoSteALS"
    (package / "ldm").mkdir(parents=True)
    (repo / "submodules/taming-transformers").mkdir()
    shutil.copyfile(sources / "rosteals/ldm/util.py",package / "ldm/util.py")
    # Leave RoSteALS and ldm as namespace packages, as in the actual upstream tree.
    (package / "fixture.py").write_text("class Model:\n    def __init__(self, secret_len):\n        self.secret_len = secret_len\n")
    code = ("from wmbench_meta_recheck.reference import rosteals_import_paths; "
            f"rosteals_import_paths({{'repo_root':{str(repo)!r}}}); "
            "from ldm.util import instantiate_from_config; "
            "n=instantiate_from_config({'target':'RoSteALS.fixture.Model','params':{'secret_len':100}}); "
            "assert n.secret_len == 100")
    child = subprocess.run([sys.executable,"-c",code],env={**os.environ,"PYTHONPATH":str(project / "src")},
                           capture_output=True,text=True)
    assert child.returncode == 0,child.stderr
