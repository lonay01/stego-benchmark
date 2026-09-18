"""Source-backed adapters for additional post-hoc watermark methods.

The orchestrator imports this module without importing PyTorch. Each adapter runs
in its own process and environment; author repositories often reuse module names
such as ``model`` and ``utils`` and must never be imported into the same process.

Inference contracts were reviewed against the commits recorded in
``registry/extended_models.json``. Published weights still need the real-model
preflight before results can be treated as an experiment.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from functools import partial
from typing import Any, Iterator


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(target)


def _validate_payload(payload: list[int], length: int) -> list[int]:
    if len(payload) != length or any(
        type(bit) not in (int, bool) or bit not in (0, 1) for bit in payload
    ):
        raise ValueError(
            f"Expected exactly {length} binary payload values; received {len(payload)}."
        )
    return [int(bit) for bit in payload]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _repository(options: dict[str, Any], extra: str = "") -> Path:
    repo = Path(options["repo_root"]).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Prepare the pinned source repository first: {repo}")
    import_root = repo / extra if extra else repo
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    return repo


def _asset(options: dict[str, Any], key: str = "weights") -> Path:
    value = options.get(key)
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise FileNotFoundError(
            f"Set options[{key!r}] to the matching published checkpoint/configuration."
        )
    path = Path(value)
    if not path.is_absolute():
        path = Path(options.get("asset_root", options["repo_root"])) / path
    if not path.is_file():
        raise FileNotFoundError(f"Required asset is absent: {path}")
    return path.resolve()


def _state_dict(path: Path, torch: Any) -> dict[str, Any]:
    # Restrict unpickling to tensor state dictionaries. No user-supplied Python
    # pickles or optimizer objects are required by these inference adapters.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a tensor state dictionary in {path}")
    return state


def _strip_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"Checkpoint has no weights with prefix {prefix!r}")
    return selected


class _TensorAdapter:
    """Common RGB PNG boundary; networks internally consume BCHW float32."""

    native_bits: int
    resolution: int | None = None
    threshold: float = 0.5

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = dict(options)
        self.repo = _repository(self.options)
        import torch

        self.torch = torch
        self.device = torch.device(self.options.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this worker environment.")
        self.model_id = self.options.get("model_id", type(self).__name__)
        self.weight_hashes: dict[str, str] = {}

    def initialize(self) -> None:
        """Worker initialization hook.

        These adapters load sources, checkpoints and auxiliary networks eagerly
        in their constructors. Returning here means all external initialization
        has already completed and the worker can fingerprint the assets.
        """

    def _checkpoint(self, key: str = "weights") -> Path:
        path = _asset(self.options, key)
        self.weight_hashes[key] = _sha256(path)
        return path

    def _image(self, path: str) -> Any:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        return self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _resize(self, image: Any, size: tuple[int, int]) -> Any:
        return self.torch.nn.functional.interpolate(
            image, size=size, mode="bilinear", align_corners=False, antialias=True
        )

    def _native_image(self, image: Any) -> Any:
        if self.resolution is None:
            return image
        return self._resize(image, (self.resolution, self.resolution))

    def _save(self, tensor: Any, path: str) -> None:
        from PIL import Image

        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(f"Expected one RGB image; got tensor shape {tuple(tensor.shape)}")
        if not self.torch.isfinite(tensor).all():
            raise ValueError("Model produced non-finite pixels.")
        pixels = (tensor[0].detach().float().clamp(0, 1) * 255).round().to(self.torch.uint8)
        array = pixels.permute(1, 2, 0).cpu().numpy()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(target, format="PNG")

    def _metadata(self, path: str, payload: list[int], **details: Any) -> None:
        _write_json(
            path,
            {
                "schema_version": 1,
                "model_id": self.model_id,
                "native_bits": self.native_bits,
                "full_payload": payload,
                "expected_bits": payload,
                "payload_indices": list(range(len(payload))),
                "checkpoint_sha256": self.weight_hashes,
                "native_resolution": self.resolution,
                "serialization": "RGB uint8 PNG; round-to-nearest and clip",
                "preprocessing": "bilinear antialias resize to native resolution; residual back to carrier resolution",
                **details,
            },
        )

    def _extract_logits(self, image: Any) -> Any:
        return self.decoder(self._native_image(image) * 2 - 1)

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        metadata = _read_json(metadata_path)
        if metadata["native_bits"] != self.native_bits:
            raise ValueError("Decoder/checkpoint payload length differs from embedding metadata.")
        with self.torch.inference_mode():
            logits = self._extract_logits(self._image(input_path)).reshape(-1)
        if logits.numel() != self.native_bits or not self.torch.isfinite(logits).all():
            raise ValueError("Decoder returned invalid or non-finite output.")
        return {
            "bits": logits.gt(self.threshold).to(self.torch.int64).cpu().tolist(),
            "score": None,
            "detection": None,
            "details": {"threshold": self.threshold, "logits": logits.cpu().tolist()},
        }

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        native = self._native_image(image)
        message = self.torch.tensor([bits], dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            marked = self.encoder(native * 2 - 1, message)
            residual = (marked - (native * 2 - 1)) / 2
            result = image + self._resize(residual, tuple(image.shape[-2:]))
        self._save(result, output_path)
        self._metadata(metadata_path, bits)


class ARWGANAdapter(_TensorAdapter):
    """Author 30-bit, 128x128 checkpoint; encoder/decoder only, no training graph."""

    native_bits = 30
    resolution = 128

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        encoder_type = importlib.import_module("model.encoder").Encoder
        decoder_type = importlib.import_module("model.decoder").Decoder
        # Values verified by pickletools disassembly of the published
        # options-and-config.pickle, without executing its training/noise objects.
        config = SimpleNamespace(
            H=128,
            W=128,
            message_length=30,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
        )
        state = _state_dict(self._checkpoint(), self.torch)["enc-dec-model"]
        self.encoder = encoder_type(config).to(self.device)
        self.decoder = decoder_type(config).to(self.device)
        self.encoder.load_state_dict(_strip_prefix(state, "encoder."), strict=True)
        self.decoder.load_state_dict(_strip_prefix(state, "decoder."), strict=True)
        self.encoder.eval()
        self.decoder.eval()


class RobustWideAdapter(_TensorAdapter):
    """Author 64-bit Robust-Wide network and author YAML configuration."""

    native_bits = 64
    resolution = 512

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        import yaml

        model_type = importlib.import_module("model").WatermarkModel
        config_path = self.repo / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.native_bits = int(config["wm_enc_config"]["message_length"])
        self.resolution = int(config["wm_enc_config"]["image_size"])
        self.model = model_type(**config)
        self.model.load_state_dict(_state_dict(self._checkpoint(), self.torch), strict=True)
        self.model.to(self.device).eval()
        self.encoder, self.decoder = self.model.encoder, self.model.decoder


class MaskWMAdapter(_TensorAdapter):
    """Global D_32bits checkpoint; no VAE-finetuned weights are substituted."""

    native_bits = 32
    resolution = 256

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        import yaml

        author = importlib.import_module("models.Mask_Model")
        config_path = self.repo / "configs/model/D_32bits.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Instantiate only the learned parts. WatermarkModel's training Noise
        # constructor evaluates a string; the benchmark supplies attacks itself.
        self.encoder = author.Encoder(**config["wm_enc_config"]).to(self.device)
        self.decoder = author.Decoder(**config["wm_dec_config"]).to(self.device)
        state = _state_dict(self._checkpoint(), self.torch)
        self.encoder.load_state_dict(_strip_prefix(state, "encoder."), strict=True)
        self.decoder.load_state_dict(_strip_prefix(state, "decoder."), strict=True)
        self.encoder.eval()
        self.decoder.eval()

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        native = self._native_image(image) * 2 - 1
        message = self.torch.tensor([bits], dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            marked = self.encoder(native, message, use_jnd=True, jnd_factor=1.3, blue=True)
            result = image + self._resize((marked - native) / 2, tuple(image.shape[-2:]))
        self._save(result, output_path)
        self._metadata(
            metadata_path,
            bits,
            variant="D_32bits",
            use_jnd=True,
            jnd_factor=1.3,
            blue=True,
            embedding_mask="full-image",
        )

    def _extract_logits(self, image: Any) -> Any:
        bits, _mask = self.decoder(self._native_image(image) * 2 - 1)
        return bits


class SepMarkAdapter(_TensorAdapter):
    """SepMark robust source-tracing decoder_C; fragile decoder_RF is recorded too."""

    native_bits = 128
    resolution = 256
    threshold = 0.0

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        encoder_type = importlib.import_module("network.Encoder_U").DW_Encoder
        decoder_type = importlib.import_module("network.Decoder_U").DW_Decoder
        state = _state_dict(self._checkpoint(), self.torch)
        if all(key.startswith("module.") for key in state):
            state = _strip_prefix(state, "module.")
        self.encoder = encoder_type(self.native_bits, attention="se").to(self.device)
        self.decoder = decoder_type(self.native_bits, attention="se").to(self.device)
        self.fragile_decoder = decoder_type(self.native_bits, attention="se").to(self.device)
        for prefix, module in (
            ("encoder.", self.encoder),
            ("decoder_C.", self.decoder),
            ("decoder_RF.", self.fragile_decoder),
        ):
            module.load_state_dict(_strip_prefix(state, prefix), strict=True)
            module.eval()

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        native = self._native_image(image) * 2 - 1
        # Training messages are {-0.1,+0.1}; passing {0,1} is an API error.
        message = (
            self.torch.tensor([bits], dtype=self.torch.float32, device=self.device) * 2 - 1
        ) * 0.1
        with self.torch.inference_mode():
            marked = self.encoder(native, message)
            result = image + self._resize((marked - native) / 2, tuple(image.shape[-2:]))
        self._save(result, output_path)
        self._metadata(
            metadata_path,
            bits,
            message_range=[-0.1, 0.1],
            primary_decoder="decoder_C",
            secondary_decoder="decoder_RF",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        result = super().decode(input_path, metadata_path)
        with self.torch.inference_mode():
            fragile = self.fragile_decoder(
                self._native_image(self._image(input_path)) * 2 - 1
            ).reshape(-1)
        if fragile.numel() != self.native_bits or not self.torch.isfinite(fragile).all():
            raise ValueError("SepMark fragile decoder returned invalid values.")
        result["details"]["fragile_bits"] = fragile.gt(0).to(self.torch.int64).cpu().tolist()
        result["details"]["primary_decoder"] = "decoder_C"
        return result


class SSLWatermarkingAdapter(_TensorAdapter):
    """Author multi-bit DINO feature optimization with a fixed saved carrier key."""

    native_bits = 30
    resolution = None

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        import numpy as np
        from torchvision import models

        self.author_utils = importlib.import_module("utils")
        self.author_images = importlib.import_module("utils_img")
        self.author_encode = importlib.import_module("encode")
        self.author_decode = importlib.import_module("decode")
        self.data_aug = importlib.import_module("data_augmentation").All()
        # Author modules use a global device; align it explicitly with the worker.
        for module in (
            self.author_utils,
            self.author_images,
            self.author_encode,
            self.author_decode,
        ):
            module.device = self.device
        self.author_images.image_mean = self.author_images.image_mean.to(self.device)
        self.author_images.image_std = self.author_images.image_std.to(self.device)

        backbone = models.resnet50(weights=None)
        backbone.head = self.torch.nn.Identity()
        backbone.fc = self.torch.nn.Identity()
        from wmbench.checkpoints import load_tensor_checkpoint

        state = load_tensor_checkpoint(self._checkpoint(), numpy_metadata=True)
        for key in ("state_dict", "model_state_dict", "teacher"):
            if key in state:
                state = state[key]
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        report = backbone.load_state_dict(state, strict=False)
        if report.missing_keys:
            raise ValueError(f"DINO backbone checkpoint is incomplete: {report.missing_keys}")
        norm = self.author_utils.load_normalization_layer(str(self._checkpoint("normalization")))
        self.model = self.author_utils.NormLayerWrapper(backbone, norm).to(self.device).eval()
        self.model.requires_grad_(False)
        self.native_bits = int(self.options.get("native_bits", 30))

        key_seed = int(self.options.get("key_seed", 0))
        with self.torch.no_grad():
            dimension = self.model(self.torch.zeros(1, 3, 224, 224, device=self.device)).shape[-1]
        # Preserve the caller's RNG state: a decoding key must not depend on the
        # message or on how many preceding tasks happened to run in the worker.
        np_state = np.random.get_state()
        try:
            np.random.seed(key_seed)
            with self.torch.random.fork_rng():
                self.torch.manual_seed(key_seed)
                self.carrier = self.author_utils.generate_carriers(self.native_bits, dimension).to(
                    self.device
                )
        finally:
            np.random.set_state(np_state)
        self.carrier_hash = hashlib.sha256(
            self.carrier.detach().cpu().numpy().tobytes()
        ).hexdigest()
        self.params = SimpleNamespace(
            batch_size=1,
            epochs=int(self.options.get("epochs", 100)),
            optimizer="Adam,lr=0.01",
            scheduler=None,
            lambda_w=float(self.options.get("lambda_w", 5e4)),
            lambda_i=1.0,
            target_psnr=float(self.options.get("native_psnr", 42.0)),
            verbose=0,
        )

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        normalized = self.author_images.normalize_img(image[0])
        message = self.torch.tensor([bits], dtype=self.torch.bool, device=self.device)

        class SingleImageLoader:
            dataset = [0]

            def __iter__(loader_self):
                yield [normalized], None

        # The image is the optimized variable, so inference_mode is invalid here.
        with self.torch.enable_grad():
            output = self.author_encode.watermark_multibit(
                SingleImageLoader(),
                message,
                self.carrier,
                self.model,
                self.data_aug,
                self.params,
            )[0]
        result = self.author_images.unnormalize_img(output).unsqueeze(0)
        self._save(result, output_path)
        key_path = str(Path(metadata_path).with_suffix(".carrier.npy"))
        import numpy as np

        np.save(key_path, self.carrier.detach().cpu().numpy(), allow_pickle=False)
        self._metadata(
            metadata_path,
            bits,
            key_seed=int(self.options.get("key_seed", 0)),
            carrier_file=Path(key_path).name,
            carrier_sha256=self.carrier_hash,
            native_psnr=self.params.target_psnr,
            optimization_epochs=self.params.epochs,
            optimization_lambda_w=self.params.lambda_w,
            preprocessing="author ImageNet normalization; original carrier resolution",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        metadata = _read_json(metadata_path)
        key_path = Path(metadata_path).parent / metadata["carrier_file"]
        carrier_array = np.load(key_path, allow_pickle=False)
        if hashlib.sha256(carrier_array.tobytes()).hexdigest() != metadata["carrier_sha256"]:
            raise ValueError("SSL carrier key does not match embedding metadata.")
        carrier = self.torch.from_numpy(carrier_array).to(self.device)
        with Image.open(input_path) as image, self.torch.inference_mode():
            output = self.author_decode.decode_multibit(
                [image.convert("RGB")], carrier, self.model
            )[0]["msg"]
        return {
            "bits": output.to(self.torch.int64).reshape(-1).tolist(),
            "score": None,
            "detection": None,
            "details": {"carrier_sha256": metadata["carrier_sha256"]},
        }


class OmniGuardCopyrightAdapter(_TensorAdapter):
    """100-bit copyright component, conditional on a verified copyright checkpoint.

    The public ``checkpoint.zip`` is not assumed to be this component: the demo
    loads anti-manipulation/localization models. A matching copyright state dict
    must be supplied, and strict loading rejects localization-only checkpoints.
    """

    native_bits = 100
    resolution = 256
    threshold = 0.0

    def __init__(self, options: dict[str, Any]) -> None:
        # Fail before importing a heavyweight framework if asset is unresolved.
        _asset(options)
        super().__init__(options)
        _repository(options, "omniguard_cop")
        author = importlib.import_module("omniguard_cop.models")
        self.encoder = author.Encoder(resolution=256, secret_len=100)
        self.decoder = author.Decoder(secret_len=100, pretrained=False, resize=224)
        self.model = author.StegModel(self.encoder, self.decoder)
        state = _state_dict(self._checkpoint(), self.torch)
        if all(key.startswith("module.") for key in state):
            state = _strip_prefix(state, "module.")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()


class ZoDiacAdapter(_TensorAdapter):
    """Author zero-bit diffusion-latent optimization and Tree-Ring p-value score.

    This is presence detection, not a 30-bit decoder. Only 512x512 carriers are
    accepted because the released watermark geometry is fixed to 4x64x64.
    """

    native_bits = 0
    resolution = 512

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        from diffusers import DDIMScheduler
        from torch.utils.checkpoint import checkpoint

        author = importlib.import_module("main.wmdiffusion")
        # Upstream checkpoint returns UNet2DConditionOutput rather than a bare
        # tensor. Non-reentrant checkpointing preserves nested-output gradients.
        # This explicit compatibility adaptation changes memory scheduling, not
        # model weights or forward values.
        author.checkpoint = partial(checkpoint, use_reentrant=False)
        self.key_type = importlib.import_module("main.wmpatch").GTWatermark
        self.ssim = importlib.import_module("loss.pytorch_ssim").ssim
        self.loss_type = importlib.import_module("loss.loss").LossProvider
        model_path = self.options.get("diffusion_model", "Manojb/stable-diffusion-2-1-base")
        revision = self.options.get(
            "diffusion_revision", "0094d483a120f3f33dafbd187ea4aa60d10de75c"
        )
        scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler", revision=revision
        )
        self.pipe = author.WMDetectStableDiffusionPipeline.from_pretrained(
            model_path,
            revision=revision,
            scheduler=scheduler,
            torch_dtype=self.torch.float32,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        for module in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            module.requires_grad_(False)
            module.eval()
        self.text_embeddings = self.pipe.get_text_embedding("")
        self.key_seed = int(self.options.get("key_seed", 10))
        self.key = self.key_type(
            self.device,
            w_channel=3,
            w_radius=10,
            generator=self.torch.Generator(device=self.device).manual_seed(self.key_seed),
        )
        self.steps = int(self.options.get("diffusion_steps", 50))
        self.iterations = int(self.options.get("iterations", 100))
        self.save_every = int(self.options.get("save_every", 5))
        self.ssim_threshold = float(self.options.get("native_ssim", 0.92))
        with _working_directory(self.repo):
            self.loss = self.loss_type([10.0, 0.1, 1.0, 0.0], self.device)

    def _carrier(self, input_path: str) -> Any:
        image = self._image(input_path)
        if tuple(image.shape[-2:]) != (512, 512):
            raise ValueError(
                "ZoDiac released configuration requires a 512x512 carrier; prepare the dataset once upstream."
            )
        return image

    def _invert(self, image: Any) -> Any:
        return self.pipe.forward_diffusion(
            latents=self.pipe.get_image_latents(image, sample=False),
            text_embeddings=self.text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=self.steps,
        )

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        _validate_payload(payload, 0)
        print("ZODIAC inversion started (50 steps in released configuration)", flush=True)
        original = self._carrier(input_path)
        latent = self._invert(original).detach().clone().requires_grad_(True)
        optimizer = self.torch.optim.Adam([latent], lr=0.01)
        schedule = self.torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 80], gamma=0.3
        )
        if self.iterations <= 0:
            raise ValueError("ZoDiac iterations must be positive.")
        if self.save_every <= 0:
            raise ValueError("ZoDiac save_every must be positive.")
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        progress_path = Path(metadata_path).with_suffix(".optimization.pt")
        signature_fields = {
            "input_sha256": _sha256(Path(input_path)),
            "model": self.options.get("diffusion_model", "Manojb/stable-diffusion-2-1-base"),
            "revision": self.options.get(
                "diffusion_revision", "0094d483a120f3f33dafbd187ea4aa60d10de75c"
            ),
            "iterations": self.iterations,
            "steps": self.steps,
            "key_seed": self.key_seed,
            "ssim": self.ssim_threshold,
        }
        signature = hashlib.sha256(
            json.dumps(signature_fields, sort_keys=True).encode()
        ).hexdigest()
        completed = 0
        if progress_path.is_file():
            saved = _state_dict(progress_path, self.torch)
            if saved["signature"] != signature:
                raise ValueError(
                    "ZoDiac optimization checkpoint belongs to a different input/configuration."
                )
            with self.torch.no_grad():
                latent.copy_(saved["latent"].to(self.device))
            optimizer.load_state_dict(saved["optimizer"])
            completed = int(saved["completed"])
            schedule.last_epoch = completed
            schedule._last_lr = [group["lr"] for group in optimizer.param_groups]
            predicted = saved["predicted"].to(self.device)
        print(
            f"ZODIAC optimization {completed}/{self.iterations}; FP32; {self.steps} diffusion steps per iteration",
            flush=True,
        )
        session_start = time.monotonic()
        session_completed = completed
        for iteration in range(completed, self.iterations):
            marked_latent = self.key.inject_watermark(latent)
            predicted = self.pipe(
                "",
                guidance_scale=1.0,
                num_inference_steps=self.steps,
                output_type="tensor",
                use_trainable_latents=True,
                init_latents=marked_latent,
            ).images
            loss = self.loss(predicted, original, marked_latent, self.key)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if (iteration + 1) % self.save_every == 0 or iteration + 1 == self.iterations:
                temporary = progress_path.with_suffix(".tmp")
                self.torch.save(
                    {
                        "signature": signature,
                        "completed": iteration + 1,
                        "latent": latent.detach().cpu(),
                        "predicted": predicted.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                    },
                    temporary,
                )
                temporary.replace(progress_path)
            # Synchronize only for honest elapsed timing; no changes to dtype, loss or optimizer.
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            count = iteration + 1 - session_completed
            elapsed = time.monotonic() - session_start
            eta = elapsed / count * (self.iterations - iteration - 1)
            checkpoint_completed = ((iteration + 1) // self.save_every) * self.save_every
            print(
                f"ZODIAC {iteration + 1}/{self.iterations}; mean {elapsed / count:.1f} s/iteration; "
                f"optimization ETA ~{eta / 60:.1f} min (excludes decode/setup); "
                f"checkpoint >= {checkpoint_completed}",
                flush=True,
            )
        # Author example postprocesses its saved, uint8-round-tripped prediction.
        predicted = (predicted.detach().clamp(0, 1) * 255).round() / 255
        lower, upper = 0.0, 1.0
        with self.torch.no_grad():
            if self.ssim(predicted, original).item() < self.ssim_threshold:
                for _ in range(30):
                    theta = (lower + upper) / 2
                    candidate = (original - predicted) * theta + predicted
                    if self.ssim(candidate, original).item() <= self.ssim_threshold:
                        lower = theta
                    else:
                        upper = theta
                    if upper - lower < 0.01:
                        break
            theta = lower
            result = (original - predicted) * theta + predicted
        self._save(result, output_path)
        key_file = Path(metadata_path).with_suffix(".zodiac_key.pt")
        self.torch.save(
            {"patch": self.key.gt_patch.cpu(), "mask": self.key.watermarking_mask.cpu()}, key_file
        )
        self._metadata(
            metadata_path,
            [],
            task="zero-bit detection",
            key_seed=self.key_seed,
            key_file=key_file.name,
            key_sha256=_sha256(key_file),
            diffusion_model=self.options.get("diffusion_model", "Manojb/stable-diffusion-2-1-base"),
            diffusion_revision=self.options.get(
                "diffusion_revision", "0094d483a120f3f33dafbd187ea4aa60d10de75c"
            ),
            iterations=self.iterations,
            diffusion_steps=self.steps,
            native_ssim=self.ssim_threshold,
            checkpoint_use_reentrant=False,
            optimization_state_file=progress_path.name,
            blend_theta=theta,
            preprocessing="512x512 RGB; no resize",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        metadata = _read_json(metadata_path)
        key_file = Path(metadata_path).parent / metadata["key_file"]
        if _sha256(key_file) != metadata["key_sha256"]:
            raise ValueError("ZoDiac key file differs from the embedding key.")
        key = _state_dict(key_file, self.torch)
        self.key.gt_patch = key["patch"].to(self.device)
        self.key.watermarking_mask = key["mask"].to(self.device)
        with self.torch.inference_mode():
            inverted = self._invert(self._carrier(input_path))
            p_value = float(self.key.tree_ring_p_value(inverted))
        # Authors call 1-p a probability; it is reported as a score here. The
        # orchestrator must calibrate a decision threshold on negative controls.
        return {
            "bits": None,
            "score": 1.0 - p_value,
            "detection": None,
            "details": {
                "tree_ring_p_value": p_value,
                "score_direction": "higher_is_watermarked",
                "author_uncalibrated_threshold": 0.9,
            },
        }
