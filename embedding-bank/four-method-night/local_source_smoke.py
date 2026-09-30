"""Optional CPU source/checkpoint audit, not a four-model GPU validation.

SepWater: execute the pinned author's actual secret-control class in isolation
from training-only imports, and reproduce 56-vs-64 latent geometry.
EditGuard: instantiate the author's complete network and strictly load every
tensor of the known degrade checkpoint. Full image forward is GPU-only upstream
because IWT allocates CUDA tensors; this script does not patch that operation.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
import yaml


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sepwater", type=Path, required=True)
    parser.add_argument("--editguard", type=Path, required=True)
    parser.add_argument("--degrade", type=Path, required=True)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = args.sepwater / "control_Latent_Diffusion_Model/autoencoder3.py"
    raw = source.read_bytes().replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != "797b7b24b4813e16db353b029409978da5f8fd5d":
        raise ValueError("SepWater source is not the reviewed revision")
    sys.path.insert(0, str(args.sepwater))
    from Latent_Diffusion_Model.modules.diffusionmodules.util import conv_nd, zero_module
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("View", "SecretEncoder")]
    assert len(nodes) == 2
    namespace = {"nn": torch.nn, "np": np, "torch": torch, "conv_nd": conv_nd, "zero_module": zero_module}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    params = yaml.safe_load((args.sepwater / "pre_trained_models/inference.yaml").read_text())["model"]["params"]
    control = namespace["SecretEncoder"](**params["control_config"]["params"]).eval()
    with torch.no_grad():
        residual, _ = control(torch.zeros(1, 3, 56, 56), torch.zeros(1, 100))
        compatible = torch.zeros(1, 3, 56, 56) + residual
        try:
            _ = torch.zeros(1, 3, 64, 64) + residual
        except RuntimeError as error:
            mismatch = str(error)
        else:
            raise AssertionError("Expected the reported 256px mismatch")
    assert tuple(compatible.shape) == (1, 3, 56, 56)

    # This is the same compatibility alias used by the packaged adapter.
    import torchvision.transforms._functional_tensor as functional
    alias = types.ModuleType("torchvision.transforms.functional_tensor")
    alias.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[alias.__name__] = alias
    sys.path.insert(0, str(args.editguard / "code"))
    from models.networks import define_G_v2
    from options import options as author_options
    # The author's parser propagates the top-level scale into network_G. Reading
    # only raw YAML would instantiate half the invertible blocks by mistake.
    options = author_options.dict_to_nonedict(author_options.parse(
        str(args.editguard / "code/options/test_editguard.yml"), is_train=False))
    network = define_G_v2(options).eval()
    expected = "771d0418aae3926c953782a61c064fa6b49b52dba839ed1b7fd3583ffb467983"
    if checksum(args.degrade) != expected:
        raise ValueError("Not the previously audited author degrade checkpoint")
    state = torch.load(args.degrade, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    assert all(torch.equal(value, state[key]) for key, value in network.state_dict().items())
    result = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "sepwater": {"source_git_blob": blob, "input_resolution": 224,
            "control_output_shape": list(residual.shape), "reproduced_256_error": mismatch,
            "scope": "actual author control-class geometry; no full AE/decoder or pretrained SepWater inference"},
        "editguard": {"checkpoint_sha256": expected, "strict_state_tensors": len(state),
            "all_state_tensors_equal": True, "scope": "complete author network construction and pretrained weight loading; no CUDA-only image forward"}}
    if args.clean:
        clean_hash = "b967b91b94a464ab73f82db63a57ee818e6fc5d99d7ae6aa094a1777cac977d5"
        assert checksum(args.clean) == clean_hash
        clean = torch.load(args.clean, map_location="cpu", weights_only=True)
        clean = {key.removeprefix("module."): value for key, value in clean.items()}
        network.load_state_dict(clean, strict=True)
        assert all(torch.equal(value, clean[key]) for key, value in network.state_dict().items())
        result["editguard_clean"] = {"checkpoint_sha256": clean_hash,
            "strict_state_tensors": len(clean), "all_state_tensors_equal": True,
            "different_from_degrade_tensors": sum(not torch.equal(clean[key], state[key]) for key in clean)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
