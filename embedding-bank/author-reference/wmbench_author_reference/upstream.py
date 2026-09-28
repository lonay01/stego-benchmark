"""Fresh upstream networks and reviewed export expressions; no benchmark adapters.

This is a clean-image/raw-payload reference harness, not a claim to execute the
authors' complete training, ECC, noise or paper-evaluation programs.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench_full_payload.storage import EvidenceError
from .core import source_root


def tensor_digest(network, prefixes=()):
    """Hash every selected weight and buffer, not only parameter names."""
    result = {}
    for name, tensor in sorted(network.state_dict().items()):
        if prefixes and not name.startswith(prefixes):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        result[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if not result:
        raise EvidenceError("Empty inference weight inventory")
    return result


def require_inference_weights(network, state):
    active = network.state_dict()
    keys = [k for k in active if k.startswith(("ae.", "control.", "decoder."))]
    if not keys:
        raise EvidenceError("No inference weights found")
    import torch
    wrong = [k for k in keys if k not in state or active[k].shape != state[k].shape or
             not torch.equal(active[k].detach().cpu(), state[k].detach().cpu())]
    if wrong:
        raise EvidenceError(f"Reference checkpoint has unfilled inference weights: {wrong[:8]}")
    return {"checked": len(keys), "missing_or_different": wrong}


def selected_function(path, name, namespace):
    """Execute one unmodified upstream function without unrelated CLI imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise EvidenceError(f"Expected exactly one upstream function: {path}:{name}")
    env = dict(namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), env)
    return env[name]


def residual_export(repo, script, signed, cover, source):
    """Run the exact residual-to-uint8 statements in RoSteALS/SepWater CLI."""
    import torch
    tree = ast.parse((repo / script).read_text(encoding="utf-8-sig"))
    statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("res", "stego_uint8"):
                statements.append(node)
    statements.sort(key=lambda n: n.lineno)
    if len(statements) != 5:
        raise EvidenceError("Reviewed residual export no longer has five assignments")
    env = {"np": np, "torch": torch, "stego": signed, "cover": cover,
           "cover_org": Image.fromarray(source), "h": source.shape[0], "w": source.shape[1]}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(repo / script), "exec"), env)
    return env["stego_uint8"]


def serialize_author(name, repo, raw, destination, *, signed=None, cover=None, source=None):
    """Rules fixed from code before holdout evaluation; never decode-and-select."""
    import torch
    import torchvision
    destination.parent.mkdir(parents=True, exist_ok=True)
    if name in ("rosteals", "sepwater"):
        pixels = residual_export(repo, "inference.py" if name == "rosteals" else "embed.py", signed, cover, source)
    elif name == "hidden_benchmark":
        convert = selected_function(repo / "utils.py", "tensor_to_image", {"np": np})
        pixels = convert(signed)[0]
    elif name == "sepmark":
        # Use the first export in test_Dual_Mark.py, before any noise layer.
        path = repo / "test_Dual_Mark.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == "single_image" for t in n.targets)),
                       key=lambda n: n.lineno)
        if len(nodes) != 2:
            raise EvidenceError("Unexpected SepMark test serializer structure")
        env = {"encoded_images": signed, "index": 0, "torch": torch}
        exec(compile(ast.Module(body=[nodes[0]], type_ignores=[]), str(path), "exec"), env)
        pixels = env["single_image"]
    elif name == "distseal_posthoc_dcae":
        save = selected_function(repo / "distseal/utils/display.py", "save_img",
                                 {"torch": torch, "torchvision": torchvision, "Tensor": torch.Tensor})
        save(raw[0], str(destination))
        return
    else:
        raise ValueError(name)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise EvidenceError("Invalid upstream export")
    Image.fromarray(pixels).save(destination, format="PNG")


class Reference:
    """Reload selected inference weights without importing any wmbench adapter."""

    def __init__(self, name, options):
        import torch
        self.name, self.device = name, options["device"]
        self.repo = source_root(options, name).resolve()
        sys.path.insert(0, str(self.repo))
        os.chdir(self.repo)
        self.details = {"kind": "upstream_core_clean_raw_payload", "adaptations": []}
        if name == "hidden_benchmark":
            from model.encoder_decoder import EncoderDecoder
            import utils
            run = options.get("kwargs", {}).get("params", {}).get("run_name", "combined-noise")
            folder = self.repo / "experiments" / run
            _, config, _ = utils.load_options(folder / "options-and-config.pickle")
            paths = sorted((folder / "checkpoints").iterdir(),
                           key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)])
            checkpoint = paths[-1]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["enc-dec-model"]
            self.network = EncoderDecoder(config, None)
            self.network.load_state_dict(state, strict=True)
            self.length = int(config.message_length)
            self.size = int(config.H)
            if config.H != config.W or self.size != 128 or self.length != 30:
                raise EvidenceError("Unexpected HiDDeN checkpoint configuration")
            self.details.update(checkpoint=str(checkpoint), configuration=vars(config),
                provenance="ando-khachatryan PyTorch implementation, NOT original Torch7 authors",
                adaptations=["Fixed crop and payload replace demo randomness", "Direct clean encoder/decoder bypasses Noiser",
                             "tensor_to_image helper selected; no resized display montage"])
        elif name in ("rosteals", "sepwater"):
            from omegaconf import OmegaConf
            if name == "rosteals":
                sys.path.insert(0, str(Path(options["repo_root"]) / "submodules/taming-transformers"))
                base = Path(options["asset_root"]) / "model_files/rosteals"
                params = options.get("kwargs", {}).get("params", {})
                config_path = Path(params.get("config_path", base / "VQ4_mir_inference.yaml"))
                checkpoint = Path(params.get("weights_path", base / "epoch=000017-step=000449999.ckpt"))
                factory = importlib.import_module("ldm.util").instantiate_from_config
                self.size = 256
            else:
                config_path = Path(options.get("config", self.repo / "pre_trained_models/inference.yaml"))
                weights = options["weights"]
                checkpoint = Path(weights["checkpoint"] if isinstance(weights, dict) else weights)
                factory = importlib.import_module("Latent_Diffusion_Model.util").instantiate_from_config
                self.size = 224
            config = OmegaConf.load(config_path).model
            self.length = int(config.params.control_config.params.secret_len)
            if self.length != 100:
                raise EvidenceError("Expected selected 100-bit checkpoint")
            config.params.decoder_config.params.secret_len = self.length
            # Training losses/noise are not called by clean inference. Their removal
            # avoids downloading unrelated perceptual/deepfake models. Disclose it.
            self.details["original_config"] = OmegaConf.to_container(config, resolve=True)
            if name == "sepwater":
                config.params.noise_config = "__none__"
                config.params.loss_config = {"target": "torch.nn.Identity"}
            self.details.update(inference_config=OmegaConf.to_container(config, resolve=True),
                                checkpoint=str(checkpoint), config_path=str(config_path),
                                adaptations=["Fixed raw100 replaces text+BCH helper",
                                    "Clean core only; paper dataset/noise/evaluation CLI not executed"])
            if name == "sepwater":
                self.details["adaptations"].append("Unused training loss/noise omitted; all ae/control/decoder tensors verified")
            self.network = factory(config)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = {k.removeprefix("module."): v for k, v in loaded.get("state_dict", loaded).items()}
            missing, unexpected = self.network.load_state_dict(state, strict=False)
            self.details["load"] = {"missing": missing, "unexpected": unexpected,
                                    **require_inference_weights(self.network, state)}
        else:
            raise ValueError("Independent network reference is restricted to the first three methods")
        self.network.to(self.device).eval().requires_grad_(False)
        if any(m.training for m in self.network.modules()):
            raise EvidenceError("Reference has a training-mode submodule")
        self.weights = tensor_digest(self.network, () if name == "hidden_benchmark" else ("ae.", "control.", "decoder."))

    def embed(self, source, payload):
        import torch
        from torchvision import transforms
        if source.shape != (self.size, self.size, 3) or len(payload) != self.length:
            raise EvidenceError("Wrong reference geometry or payload")
        transform = transforms.Compose([transforms.Resize((self.size, self.size)), transforms.ToTensor(),
                                        transforms.Normalize([.5] * 3, [.5] * 3)])
        cover = transform(Image.fromarray(source)).unsqueeze(0).to(self.device)
        secret = torch.tensor([payload], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if self.name == "hidden_benchmark":
                signed = self.network.encoder(cover, secret)
            else:
                latent = self.network.encode_first_stage(cover)
                marked, _ = self.network(latent, None, secret)
                signed = self.network.decode_first_stage(marked)
        return signed, cover

    def decode(self, rgb):
        import torch
        # No expected payload is accepted here.
        tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            logits = self.network.decoder((tensor - .5) / .5).detach().float().cpu().numpy().reshape(-1)
        if len(logits) != self.length or not np.isfinite(logits).all():
            raise EvidenceError("Invalid upstream decoder output")
        bits = (np.rint(logits).clip(0, 1) if self.name == "hidden_benchmark" else logits > 0).astype(int)
        return {"bits": bits.tolist(), "logits": logits.tolist(),
                "threshold_rule": "numpy_round_clip" if self.name == "hidden_benchmark" else "greater_than_zero"}
