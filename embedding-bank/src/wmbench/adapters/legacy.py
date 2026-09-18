"""The 15 legacy configurations, with explicit IO fixes and no benchmark CLI magic.

WIBE is imported at a pinned commit. It is not installed with all algorithm extras.
Weights retain upstream provenance; RivaGAN and HiDDeN remain benchmark substitutes.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from wmbench.io import download, read_json, safe_extract, write_json
from wmbench.matching import load_rgb, save_rgb


class InvisibleWatermarkAdapter:
    """CPU DWT-DCT-SVD or the package's 32-bit ONNX RivaGAN implementation."""

    def __init__(self, options: dict):
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        self.options = options
        self.method = options.get("method", "dwtDctSvd")
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder("bits", 32)
        if self.method == "rivaGan":
            self.encoder.loadModel()
            self.decoder.loadModel()

    def _arguments(self):
        return {} if self.method == "rivaGan" else {"scales": [0, 36, 0], "block": 4}

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        if len(payload) != 32:
            raise ValueError("This legacy configuration requires exactly 32 raw bits")
        image = load_rgb(input_path)
        self.encoder.set_watermark("bits", payload)
        result = self.encoder.encode(image[:, :, ::-1].copy(), self.method, **self._arguments())
        save_rgb(output_path, result[:, :, ::-1].astype(np.uint8))
        write_json(
            metadata_path,
            {"expected_bits": payload, "method": self.method, "ecc": "none", "carrier": "image"},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict:
        bits = self.decoder.decode(
            load_rgb(input_path)[:, :, ::-1].copy(), self.method, **self._arguments()
        )
        return {"bits": np.asarray(bits).astype(int).reshape(-1).tolist(), "ecc": "none"}


class WibeAdapter:
    """One wrapper instance stays resident for an entire worker task list."""

    def __init__(self, options: dict):
        import torch

        self.torch = torch
        self.options = options
        self.device = options.get("device", "cuda")
        self.model_id = options["model_id"]
        source = Path(options["repo_root"])
        assets = Path(options["asset_root"])
        sys.path.insert(0, str(source / "src"))
        # WIBE author wrappers have resource-relative paths. Link code, not user results.
        assets.mkdir(parents=True, exist_ok=True)
        for name in ("resources", "submodules", "src"):
            target = assets / name
            if not target.exists():
                target.symlink_to(source / name, target_is_directory=True)
        os.chdir(assets)
        import wibench.download as downloader

        def fetch_archive(url: str, name: str):
            archive = assets / "downloads" / f"{name}.zip"
            download(url.rstrip("/") + "/download", archive)
            safe_extract(archive, assets / "model_files")

        downloader.download_folder = fetch_archive
        module_name, class_name = options["wrapper"].split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        kwargs = json.loads(json.dumps(options.get("kwargs", {})))
        if "params" in kwargs:
            kwargs["params"]["device"] = self.device
        else:
            kwargs["device"] = self.device
        if self.model_id in ("videoseal", "pixelseal", "chunkyseal"):
            # Skip implicit torch.compile: compilation is not part of the scientific protocol.
            # Direct setup also ensures tensors and the model are on the same device.
            from videoseal.utils.cfg import setup_model_from_model_card

            name = self.model_id.upper()
            required = getattr(module, f"REQUIRED_FILES_{name}")
            if not all((assets / "model_files" / self.model_id / p).exists() for p in required):
                fetch_archive(getattr(module, f"URL_{name}"), self.model_id)
            print(
                f"WM_DIAG {self.model_id}: building model/loading checkpoint on CPU (FP32)",
                flush=True,
            )
            self.seal = setup_model_from_model_card(Path(options["model_card"]))
            print(f"WM_DIAG {self.model_id}: moving model to {self.device}", flush=True)
            self.seal.eval().to(self.device)
            self.wrapper = None
        else:
            self.seal = None
            if self.model_id == "vine_r":
                # Pinned WIBE decorates __init__ and downloads only VINE-B.
                # R weights were downloaded by prepare_model at pinned revisions.
                self.wrapper = cls.__new__(cls)
                cls.__init__.__wrapped__(self.wrapper, **kwargs)
            else:
                self.wrapper = cls(**kwargs)
            # Some upstream wrappers do not explicitly switch their modules to inference mode.
            for value in vars(self.wrapper).values():
                if isinstance(value, torch.nn.Module):
                    value.eval()
        from wibench.watermark_data import TorchBitWatermarkData

        self.bit_data = TorchBitWatermarkData

    def _image(self, path: str):
        return self.torch.from_numpy(load_rgb(path).copy()).permute(2, 0, 1).float() / 255

    def _payload(self, bits):
        return self.bit_data(self.torch.tensor([bits], dtype=self.torch.int64))

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        torch = self.torch
        image = self._image(input_path)
        watermark = self._payload(payload)
        with torch.no_grad():
            if self.seal is not None:
                result = self.seal.embed(
                    image.unsqueeze(0).to(self.device),
                    msgs=watermark.watermark.to(self.device),
                    is_video=False,
                    lowres_attenuation=True,
                )["imgs_w"].squeeze(0)
            elif self.model_id == "invismark":
                from wibench.utils import normalize_image, denormalize_image

                inner = self.wrapper.invismark
                result = inner.model._encode(
                    # Author _encode explicitly moves its residual back to CPU
                    # before adding it to inputs. The cover MUST remain on CPU.
                    normalize_image(image).cpu(),
                    watermark.watermark.float().to(self.device),
                )[0]
                result = denormalize_image(result)
            else:
                result = self.wrapper.embed(image, watermark)
        result = result.detach().squeeze().cpu()
        if tuple(result.shape) != tuple(image.shape) or not torch.isfinite(result).all():
            raise ValueError(f"Invalid encoder output: {tuple(result.shape)}")
        output = torch.round(result.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        save_rgb(output_path, output)
        write_json(
            metadata_path,
            {
                "expected_bits": payload,
                "ecc": "none",
                "model_id": self.model_id,
                "dtype": "float32",
                "serialization": "clip_round_u8",
                "wrapper": self.options["wrapper"],
            },
        )

    def decode(self, input_path: str, metadata_path: str) -> dict:
        torch = self.torch
        image = self._image(input_path)
        metadata = read_json(metadata_path)
        with torch.no_grad():
            if self.seal is not None:
                result = self.seal.detect(image.unsqueeze(0).to(self.device), is_video=False)
                bits = (result["preds"][0, 1:] > 0).int().cpu().numpy()
            else:
                # These blind wrappers accept a watermark object only for the common API.
                # Never expose the expected answer to an extractor.
                bits = self.wrapper.extract(
                    image, self._payload([0] * len(metadata["expected_bits"]))
                )
                if isinstance(bits, torch.Tensor):
                    bits = bits.detach().cpu().numpy()
        bits = np.asarray(bits).reshape(-1)
        if not np.isfinite(bits).all():
            raise ValueError("Non-finite decoder output")
        return {"bits": np.clip(np.rint(bits), 0, 1).astype(int).tolist(), "ecc": "none"}


class EditGuardAdapter:
    """Copyright path of EditGuard, not its localization/forgery-detection result."""

    def __init__(self, options: dict):
        import torch

        source = Path(options["repo_root"]) / "code"
        sys.path.insert(0, str(source))
        os.chdir(source)
        from options import options as option
        from models import create_model
        from models.modules.common import DWT

        config = option.parse(str(source / "options/test_editguard.yml"), is_train=False)
        for key in (
            "sdinpaint",
            "controlnetinpaint",
            "sdxl",
            "repaint",
            "addnoise",
            "addjpeg",
            "addpossion",
            "degrade_shuffle",
        ):
            config[key] = False
        self.device = options.get("device", "cuda")
        config["gpu_ids"] = [0] if self.device.startswith("cuda") else None
        config["message_length"] = 64
        self.model = create_model(option.dict_to_nonedict(config))
        self.model.load_test(options["weights"])
        self.net = self.model.netG.eval()
        self.dwt = DWT()
        self.torch = torch

    def _image(self, path):
        return (
            self.torch.from_numpy(load_rgb(path).copy())
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)[:, [2, 1, 0]]
            / 255
        )

    def embed(self, input_path, output_path, payload, metadata_path):
        torch = self.torch
        cover = self._image(input_path)
        local = torch.zeros_like(cover)
        local[:, 2] = 1
        message = torch.tensor([payload], device=self.device).float() - 0.5
        with torch.no_grad():
            _, encoded = self.net(x=self.dwt(cover), x_h=[self.dwt(local)], message=message)
        output = torch.round(encoded[0, [2, 1, 0]].clamp(0, 1) * 255).byte().cpu()
        save_rgb(output_path, output.permute(1, 2, 0).numpy())
        write_json(
            metadata_path, {"expected_bits": payload, "component": "copyright", "ecc": "none"}
        )

    def decode(self, input_path, metadata_path):
        with self.torch.no_grad():
            value = self.net(x=self.model.Quantization(self._image(input_path)), rev=True)
        value = value[-1] if isinstance(value, (tuple, list)) else value
        if value.numel() != 64:
            raise ValueError("EditGuard did not return the 64-bit copyright component")
        return {"bits": (value.reshape(-1) > 0).int().cpu().tolist(), "ecc": "none"}
