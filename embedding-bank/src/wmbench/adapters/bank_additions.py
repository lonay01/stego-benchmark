"""Five source-backed additions to the shared, embedding-only image bank.

Heavy imports are confined to worker constructors. Revisions and downloadable
assets are declared in ``registry/bank_additions.json``. This module implements
inference, not training or generation. Passing its offline tests is NOT a claim
that the pretrained models have passed the GPU/image canary.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from .extended import (
    _TensorAdapter,
    _read_json,
    _repository,
    _sha256,
    _state_dict,
    _validate_payload,
    _working_directory,
    _write_json,
)


def validate_fnns_payload(payload: list[int], prefix: list[int]) -> list[int]:
    """FNNS has 16 detection bits, 2 filler bits, then the common 30 bits."""
    bits = _validate_payload(payload, 48)
    expected = _validate_payload(prefix, 16)
    if bits[:16] != expected:
        raise ValueError("FNNS payload must retain the configured 16-bit detection prefix.")
    return bits


def _source_namespace(name: str, path: Path) -> None:
    """Load a source package without its unrelated benchmark/CLI initializer.

    OmniSealBench's root initializer imports audio benchmarking infrastructure.
    Its image model files are otherwise independent. Only that package root is
    bypassed; FNNS, JND, interface, utilities and their code run unchanged.
    A worker must never mix this selected source with another installed version.
    """
    if not path.is_dir():
        raise FileNotFoundError(path)
    existing = sys.modules.get(name)
    if existing is not None:
        if list(getattr(existing, "__path__", [])) != [str(path)]:
            raise RuntimeError(f"Conflicting {name} already loaded; use an isolated worker.")
        return
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package


def _verified_trustmark_asset(root: Path, filename: str, expected_md5: str) -> Path:
    path = root / Path(filename).name
    if not path.is_file():
        raise FileNotFoundError(f"Download the declared TrustMark asset first: {path}")
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    if checksum != expected_md5:
        raise ValueError(f"TrustMark author checksum mismatch: {path.name}")
    return path


class TrustMarkVariantAdapter:
    """Official Adobe B/P encoder with all 100 raw bits; ECC is disabled.

    The upstream public encode/decode API is retained, including its variant-P
    strength factor, resizing, mean removal, feathering and uint8 serialization.
    Asset locations are redirected to the inventoried persistent asset directory.
    """

    native_bits = 100

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = dict(options)
        self.variant = options["variant"]
        if self.variant not in ("B", "P"):
            raise ValueError("This adapter is restricted to the requested B and P variants.")
        self.model_id = options.get("model_id", f"trustmark_{self.variant.lower()}")
        _repository(options, "python")
        from trustmark.trustmark import MODEL_CHECKSUMS, TrustMark

        asset_root = Path(options["asset_root"])
        hashes: dict[str, str] = {}

        class LocalAssetsTrustMark(TrustMark):
            def check_and_download(self, filename):
                # Deliberately prohibit implicit downloads to the source checkout.
                return _verified_trustmark_asset(
                    asset_root, filename, MODEL_CHECKSUMS[Path(filename).name]
                )

            def load_model(self, config_path, weight_path, *args, **kwargs):
                config = _verified_trustmark_asset(
                    asset_root, config_path, MODEL_CHECKSUMS[Path(config_path).name]
                )
                weight = _verified_trustmark_asset(
                    asset_root, weight_path, MODEL_CHECKSUMS[Path(weight_path).name]
                )
                hashes[config.name] = _sha256(config)
                hashes[weight.name] = _sha256(weight)
                return super().load_model(str(config), str(weight), *args, **kwargs)

        self.model = LocalAssetsTrustMark(
            use_ECC=False,
            verbose=False,
            secret_len=100,
            device=options.get("device", "cuda"),
            model_type=self.variant,
            loadRemover=False,
            loadBBoxDetector=False,
        )
        if self.model.encoder is None or self.model.decoder is None:
            raise RuntimeError("TrustMark failed to load its published encoder/decoder.")
        self.weight_hashes = hashes

    def initialize(self) -> None:
        pass  # Constructor performs checkpoint loading.

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        from PIL import Image

        bits = _validate_payload(payload, self.native_bits)
        with Image.open(input_path) as source:
            cover = source.convert("RGB")
            if cover.size != (512, 512):
                raise ValueError("The shared bank requires 512x512 carriers.")
            result = self.model.encode(cover, "".join(map(str, bits)), MODE="binary")
        if result.size != (512, 512) or result.mode != "RGB":
            raise ValueError("TrustMark changed the common image grid or RGB mode.")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, format="PNG")
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "model_id": self.model_id,
                "native_bits": self.native_bits,
                "expected_bits": bits,
                "ecc": "none",
                "variant": self.variant,
                "checkpoint_sha256": self.weight_hashes,
                "serialization": "official TrustMark encode: clipped uint8 truncation; RGB PNG",
                "preprocessing": "official encode/decode; native encoder256; B decoder245 / P224",
                "native_strength_argument": 1.0,
                "variant_strength_multiplier": 1.25 if self.variant == "P" else 1.0,
            },
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        from PIL import Image

        metadata = _read_json(metadata_path)
        if metadata.get("variant") != self.variant:
            raise ValueError("TrustMark decoding variant differs from embedding metadata.")
        with Image.open(input_path) as source:
            text, detected, _ = self.model.decode(source.convert("RGB"), MODE="binary")
        if not detected or len(text) != 100 or any(bit not in "01" for bit in text):
            raise ValueError("TrustMark raw decoder did not return exactly 100 binary bits.")
        return {"bits": list(map(int, text)), "ecc": "none"}


class FINAdapter(_TensorAdapter):
    """Author HEAVY FED+INL model with an explicit 128px residual-transfer adapter."""

    native_bits = 64
    resolution = 128
    threshold = 0.0

    def __init__(self, options: dict[str, Any]) -> None:
        if options.get("resolution_policy") != "residual_128_to_512":
            raise ValueError("FIN requires explicit resolution_policy=residual_128_to_512.")
        super().__init__(options)
        from models.encoder_decoder import FED, INL

        self.fed = FED().to(self.device).eval()
        self.inl = INL().to(self.device).eval()
        for model, key in ((self.fed, "weights"), (self.inl, "inl_weights")):
            state = _state_dict(self._checkpoint(key), self.torch)["net"]
            state = {name: value for name, value in state.items() if "tmp_var" not in name}
            model.load_state_dict(state, strict=True)
            model.requires_grad_(False)

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        if tuple(image.shape[-2:]) != (512, 512):
            raise ValueError("The shared bank requires 512x512 carriers.")
        native = self._native_image(image) * 2 - 1
        message = self.torch.tensor([bits], device=self.device, dtype=self.torch.float32) - 0.5
        with self.torch.inference_mode():
            marked, _ = self.fed([native, message])
            result = image + self._resize((marked - native) / 2, (512, 512))
        self._save(result, output_path)
        self._metadata(
            metadata_path,
            bits,
            variant="HEAVY",
            ecc="none",
            resolution_policy="residual_128_to_512",
            preprocessing="bilinear antialiased 512->128; transfer marked-minus-cover residual back to original512; inverse INL then FED for decoding",
            message_domain=[-0.5, 0.5],
            decode_threshold=0.0,
            deviation="512px residual transfer is a benchmark adaptation, not native FIN128.",
        )

    def _extract_logits(self, image):
        normalized = self._native_image(image) * 2 - 1
        corrected = self.inl(normalized, rev=True)
        zeros = self.torch.zeros((1, 64), device=self.device, dtype=corrected.dtype)
        _, logits = self.fed([corrected, zeros], rev=True)
        return logits


class FNNSAdapter(_TensorAdapter):
    """OmniSealBench's 48-bit FNNS optimization, not a dense steganography model."""

    native_bits = 48
    resolution = 512
    threshold = 0.0

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self.prefix = _validate_payload(options["detection_prefix"], 16)
        _source_namespace("omnisealbench", self.repo / "src" / "omnisealbench")
        from omnisealbench.models.fnns import build_model

        self.model = build_model(
            model_card_or_path=str(self._checkpoint()),
            nbits=48,
            lambda_w=float(options.get("lambda_w", 1.3)),
            nsteps=int(options.get("nsteps", 10)),
            data_aug="none",
            detection_bits=16,
            device=str(self.device),
        )
        # Upstream build_model fixes seed=42 regardless of its seed argument.
        if self.model.seed != 42:
            raise ValueError("Pinned FNNS initialization-seed contract changed.")
        self.model.model.eval()
        self.model.model.requires_grad_(False)  # image/distortion gradients remain enabled

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        bits = validate_fnns_payload(payload, self.prefix)
        image = self._image(input_path)
        if tuple(image.shape[-2:]) != (512, 512):
            raise ValueError("The shared bank requires 512x512 carriers.")
        message = self.torch.tensor(bits, dtype=self.torch.float32, device=self.device)
        with self.torch.enable_grad():
            marked = self.model.generate_watermark([image[0]], message)[0]
        self._save(marked.unsqueeze(0), output_path)
        self._metadata(
            metadata_path,
            bits,
            ecc="none",
            detection_prefix=self.prefix,
            detection_indices=list(range(16)),
            common_indices=list(range(18, 48)),
            filler_indices=[16, 17],
            lambda_w=self.model.alpha,
            nsteps=self.model.nsteps,
            initialization_seed=42,
            preprocessing="OmniSealBench FNNS RGB512 ImageNet normalization; no spatial resize",
        )

    def _extract_logits(self, image):
        # Blind extraction returns all 48 bits, including the 16 detection bits.
        return self.model.model(self.model.normalize(image))


class DistSealPosthocAdapter(_TensorAdapter):
    """DC-AE-f64c128 post-hoc latent embedder, no diffusion generator required."""

    native_bits = 64
    resolution = 512
    threshold = 0.0

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        from diffusers import AutoencoderDC
        from distseal.utils.cfg import get_config_from_checkpoint, setup_model

        checkpoint = self._checkpoint()
        with _working_directory(self.repo):
            config = get_config_from_checkpoint(checkpoint)
            if int(config.args.nbits) != 64:
                raise ValueError("DistSeal checkpoint is not the selected 64-bit variant.")
            if config.args.get("attenuation") is None:
                config.args.attenuation = "none"  # upstream setup_model calls .lower()
            self.model = setup_model(config, checkpoint)
            # Upstream loader uses strict=False; reject incomplete inference weights.
            state = _state_dict(checkpoint, self.torch)["model"]
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.model.img_size = 512
        self.scaling_w = float(options.get("scaling_w", 0.5))
        self.model.blender.scaling_w = self.scaling_w  # official detector_dc-ae card
        snapshot = Path(options["asset_root"]) / options.get("autoencoder_path", "dcae")
        if not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Prepare the pinned DC-AE snapshot first: {snapshot}")
        self.autoencoder = (
            AutoencoderDC.from_pretrained(
                str(snapshot),
                local_files_only=True,
                torch_dtype=self.torch.float32,
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.autoencoder_revision = options["autoencoder_revision"]

    def embed(self, input_path: str, output_path: str, payload: list[int], metadata_path: str):
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        if tuple(image.shape[-2:]) != (512, 512):
            raise ValueError("DistSeal bank input must be RGB512.")
        message = self.torch.tensor([bits], dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            latent = self.autoencoder.encode(image * 2 - 1).latent
            prediction = self.model.embedder(latent, message)
            marked_latent = self.model.blender(latent, prediction)
            result = (self.autoencoder.decode(marked_latent).sample + 1) / 2
        self._save(result, output_path)
        self._metadata(
            metadata_path,
            bits,
            ecc="none",
            scaling_w=self.scaling_w,
            autoencoder_revision=self.autoencoder_revision,
            preprocessing="official posthoc DC-AE path: RGB[-1,1]->latent->embedder->blender->DC-AE decode->RGB[0,1]",
            reference="original COCO PNG, not autoencoder reconstruction",
            variant="posthoc_dc_ae_f64c128",
        )

    def _extract_logits(self, image):
        # Extractor.detect_watermark uses forward(imgs)[:, 1:] > 0.
        # Keep logits for strict length/finite checking by the common decoder.
        return self.model.detector(image)[:, 1:]
