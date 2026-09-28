"""Optional CPU checks using independently fetched pinned upstream source files.

Set WMBENCH_UPSTREAM_FIXTURES to a directory containing hidden/, rosteals/,
sepmark/, sepwater/ and distseal/. No pretrained weights are required. HiDDeN
uses a synthetic state dict to exercise the real upstream loader and network.
"""
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pytest
import torch

from wmbench.matching import load_rgb
from wmbench_author_reference import upstream


@pytest.fixture
def sources():
    value = os.environ.get("WMBENCH_UPSTREAM_FIXTURES")
    if not value:
        pytest.skip("Optional upstream sources were not supplied")
    return Path(value)


@pytest.mark.parametrize("name,directory", [
    ("hidden_benchmark","hidden"), ("rosteals","rosteals"), ("sepwater","sepwater"),
    ("sepmark","sepmark"), ("distseal_posthoc_dcae","distseal")])
def test_real_upstream_export(tmp_path,sources,name,directory):
    repo = sources / directory
    raw = torch.linspace(-.1,1.1,3*8*8).reshape(1,3,8,8)
    source = np.full((8,8,3),100,dtype=np.uint8)
    cover = torch.from_numpy(source.copy()).permute(2,0,1)[None].float()/127.5-1
    signed = raw*2-1
    target = tmp_path / "out.png"
    upstream.serialize_author(name,repo,raw,target,signed=signed,cover=cover,source=source)
    result = load_rgb(target)
    assert result.shape == source.shape and result.dtype == np.uint8
    if name == "sepmark":
        expected = ((signed[0].clamp(-1,1).permute(1,2,0)+1)/2*255+.5).clamp(0,255).byte().numpy()
    elif name == "distseal_posthoc_dcae":
        expected = (raw[0].clamp(0,1)*255).byte().permute(1,2,0).numpy()
    elif name == "hidden_benchmark":
        expected = np.clip((signed[0].permute(1,2,0).numpy()+1)*127.5,0,255).astype(np.uint8)
    else:
        res = (signed.clamp(-1,1)-cover)[0].permute(1,2,0).numpy()
        expected = (np.clip(res+source/127.5-1.,-1,1)*127.5+127.5).astype(np.uint8)
    assert np.array_equal(result,expected)


def test_real_hidden_core_fresh_loader(tmp_path,sources,monkeypatch):
    """Actual upstream classes, generated weights: a wiring test, not BER evidence."""
    root = tmp_path / "wibe"
    repo = root / "submodules/HiDDeN"
    shutil.copytree(sources / "hidden",repo)
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.chdir(tmp_path)
    # Keep imported upstream names out of other tests.
    before = set(sys.modules)
    try:
        from options import HiDDenConfiguration
        from model.encoder_decoder import EncoderDecoder
        cfg = HiDDenConfiguration(128,128,30,2,4,2,4,False,False,2,4,1.,1.,0.)
        net = EncoderDecoder(cfg,None).eval()
        folder = repo / "experiments/combined-noise"
        (folder / "checkpoints").mkdir(parents=True)
        with (folder / "options-and-config.pickle").open("wb") as f:
            for value in ({},[],cfg):
                pickle.dump(value,f)
        torch.save({"enc-dec-model":net.state_dict()},folder / "checkpoints/epoch-1.pyt")
        ref = upstream.Reference("hidden_benchmark",{"repo_root":str(root),"device":"cpu"})
        assert ref.network is not net
        assert ref.weights == upstream.tensor_digest(net)
        image = np.full((128,128,3),100,dtype=np.uint8)
        payload = [0,1]*15
        signed,cover = ref.embed(image,payload)
        with torch.no_grad():
            direct = net.encoder(cover,torch.tensor([payload]).float())
        assert torch.equal(signed,direct)
        rgb = ((signed+1)/2)[0].permute(1,2,0).numpy()
        decoded = ref.decode(rgb)
        assert len(decoded["bits"]) == 30
        assert decoded["threshold_rule"] == "numpy_round_clip"
    finally:
        for name in set(sys.modules)-before:
            if name in ("options","utils","vgg_loss") or name.startswith(("model.","noise_layers.")) or name in ("model","noise_layers"):
                sys.modules.pop(name,None)

