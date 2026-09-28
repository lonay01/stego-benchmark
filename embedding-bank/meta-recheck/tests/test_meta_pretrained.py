"""Optional real pretrained CPU smoke. No inference about COCO canary quality."""
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from wmbench.io import read_json, write_json
from wmbench_meta_recheck.meta import MetaHidden, verify_asset, blob


def test_real_checkpoint_and_official_demo(tmp_path,project):
    supplied = os.environ.get("WMBENCH_META_FIXTURES")
    if not supplied:
        pytest.skip("Set WMBENCH_META_FIXTURES to the downloaded official Meta fixture folder")
    fixture = Path(supplied)
    assert blob((fixture / "demo.png").read_bytes()) == "e4e1e91a476f0dacf73e0c2c95dfe33fa140750d"
    assets = tmp_path / "assets"
    for relative,filename in (("hidden/models.py","models.py"),("hidden/attenuations.py","attenuations.py"),
                              ("hidden/ckpts/hidden_replicate.pth","hidden_replicate.pth")):
        target = assets / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(fixture / filename,target)
        verify_asset(target,relative,read_json(project / "meta_contract.json"))
    torch.set_num_threads(4)
    model = MetaHidden(assets,"cpu")
    payload = [int(b) for b in "111010110101000001010111010011010100010000100111"]
    with Image.open(fixture / "demo.png") as image:
        image = image.convert("RGB").resize((512,512),Image.Resampling.BICUBIC)
    observations = {}
    for profile,size in (("checkpoint_256",256),("demo_512",512)):
        offset = (512-size)//2
        source = image.crop((offset,offset,offset+size,offset+size))
        attempt = tmp_path / profile
        attempt.mkdir()
        result = model.measure(source,payload,profile,attempt)
        assert result["png_file_matches_pil"]
        assert result["round8_vs_png_same_bits"]
        assert np.load(attempt / "raw_rgb.npy").shape == (size,size,3)
        for stage,value in result["decoded"].items():
            assert len(value["bits"]) == 48 and len(value["logits"]) == 48
            assert np.isfinite(value["logits"]).all()
        observations[profile] = {"psnr":result["psnr_original_to_png"],
            "errors":{s:sum(a != b for a,b in zip(v["bits"],payload)) for s,v in result["decoded"].items()}}
    print("REAL_META_SMOKE",observations)
    # Record quality, but no test requires zero BER or chooses the better profile.
    if os.environ.get("WMBENCH_META_SMOKE_OUTPUT"):
        write_json(Path(os.environ["WMBENCH_META_SMOKE_OUTPUT"]),observations)
