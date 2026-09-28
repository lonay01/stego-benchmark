"""Pinned Meta Revisited HiDDeN inference; not the original 2018 implementation.

Load the unchanged author classes after checking source hashes. Training-only
ViT/DVMark imports are not executed. No dependency on diagnostic notebooks.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.io import sha256, write_json
from wmbench.matching import save_rgb

MODEL_ID = "meta_hidden_revisited"
REVISION = "c91217c06e953a4eaf270ef9de4a3939bcea3366"
ASSET_HASHES = {
    "hidden/models.py": "b0eceb802f1703f9e7f97446128a6e30a2a5779ac1c3c3904418b3c058a3bcc6",
    "hidden/attenuations.py": "893809a7a3f2c846ab355a361d3eca27daa3421c5d020eee7ce9d231cf9bca4c",
    "hidden/ckpts/hidden_replicate.pth":
        "5a83d4d5c85f6d58abef7d1083d559dae1eb9171bf63ca0e0c13016fd4741443",
}
CHECKPOINT_SETTINGS = {
    "num_bits": 48, "redundancy": 1, "img_size": 256, "encoder": "hidden",
    "encoder_depth": 4, "encoder_channels": 64, "use_tanh": True,
    "decoder_depth": 8, "decoder_channels": 64, "scaling_i": 1.0,
    "scaling_w": 0.3, "scale_channels": False, "attenuation": None,
}


def verify_assets(root: Path) -> None:
    for relative, expected in ASSET_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Meta source/checkpoint checksum mismatch: {relative}")


def load_classes(root: Path):
    """Execute only four reviewed class definitions, without rewriting their AST."""
    import torch

    verify_assets(root)
    spec = importlib.util.spec_from_file_location(
        "wmbench_meta_attenuations", root / "hidden/attenuations.py"
    )
    attenuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attenuation)
    wanted = {"ConvBNRelu", "HiddenEncoder", "HiddenDecoder", "EncoderWithJND"}
    source = root / "hidden/models.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    if len(nodes) != 4 or {n.name for n in nodes} != wanted:
        raise ValueError("Unexpected Meta inference classes")
    namespace = {"__name__": "wmbench_meta_classes", "torch": torch,
                 "nn": torch.nn, "attenuations": attenuation}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class MetaHiddenAdapter:
    """48-bit checkpoint profile, 256-pixel RGB carrier, blind extraction."""

    def __init__(self, options: dict):
        import torch
        from torchvision import transforms

        if options.get("profile", "checkpoint_256") != "checkpoint_256":
            raise ValueError("Only the fixed checkpoint_256 Meta profile is registered")
        self.device = options.get("device", "cuda")
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Meta requires the requested CUDA device; no silent CPU fallback")
        self.torch = torch
        root = Path(options["asset_root"])
        classes = load_classes(root)
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(root / "hidden/ckpts/hidden_replicate.pth",
                                    map_location="cpu", weights_only=True)
        settings = vars(checkpoint["params"])
        if any(settings.get(k) != v for k, v in CHECKPOINT_SETTINGS.items()):
            raise ValueError("Unexpected Meta checkpoint inference configuration")
        encoder = classes["HiddenEncoder"](num_blocks=4, num_bits=48, channels=64)
        self.decoder = classes["HiddenDecoder"](num_blocks=8, num_bits=48, channels=64)
        state = {k.removeprefix("module."): v for k, v in checkpoint["encoder_decoder"].items()}
        if any(not k.startswith(("encoder.", "decoder.")) for k in state):
            raise ValueError("Unexpected Meta checkpoint state prefix")
        for prefix, model in (("encoder.", encoder), ("decoder.", self.decoder)):
            model.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()
                                   if k.startswith(prefix)}, strict=True)
            model.to(self.device).eval().requires_grad_(False)
        self.encoder = classes["EncoderWithJND"](
            encoder, None, False, 1.0, 0.3
        ).to(self.device).eval()
        self.normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        self.unnormalize = transforms.Normalize(
            [-.485/.229, -.456/.224, -.406/.225], [1/.229, 1/.224, 1/.225]
        )
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def _input(self, path: str, *, embedding: bool):
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError("Meta expects RGB input; conversion must be recorded upstream")
            if embedding and image.size != (256, 256):
                raise ValueError("Meta checkpoint_256 requires a 256x256 carrier; crop upstream")
            # Attacked images may have a different size. Never resize secretly.
            tensor = self.normalize(self.to_tensor(image)).unsqueeze(0)
        return tensor.to(self.device)

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        if len(payload) != 48 or any(type(b) is not int or b not in (0, 1) for b in payload):
            raise ValueError("Meta Revisited HiDDeN requires exactly 48 binary integers")
        torch = self.torch
        x = self._input(input_path, embedding=True)
        bipolar = 2 * torch.tensor([payload], dtype=torch.float32, device=self.device) - 1
        with torch.inference_mode():
            rgb = self.unnormalize(self.encoder(x, bipolar))
            if not torch.isfinite(rgb).all():
                raise ValueError("Meta encoder returned non-finite pixels")
            rounded = torch.round(255 * rgb.clamp(0, 1)) / 255
            png = self.to_pil(rounded.squeeze(0).cpu())
        save_rgb(output_path, np.array(png, dtype=np.uint8))
        write_json(metadata_path, {
            "model_id": MODEL_ID, "profile": "checkpoint_256", "source_revision": REVISION,
            "asset_sha256": ASSET_HASHES, "settings": CHECKPOINT_SETTINGS,
            "expected_bits": payload, "ecc": "none", "dtype": "float32",
            "serialization": "Meta_demo_clip_round_then_ToPILImage",
            "carrier_size": [256, 256],
        })

    def decode(self, input_path: str, metadata_path: str) -> dict:
        # metadata_path is deliberately unread: reference bits are not decoder input.
        with self.torch.inference_mode():
            logits = self.decoder(self._input(input_path, embedding=False))
        logits = logits.detach().float().cpu().numpy().reshape(-1)
        if logits.shape != (48,) or not np.isfinite(logits).all():
            raise ValueError("Meta decoder returned invalid 48-bit logits")
        return {"bits": (logits > 0).astype(int).tolist(),
                "logits": logits.tolist(), "ecc": "none"}
