"""Source-backed generation-origin adapters, each used in its own process.

These methods create carriers from prompts. They do not watermark COCO pixels.
No training, silent backbone replacement, or matched-PSNR scaling is performed.
Heavy dependencies are imported only after construction in the model environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import types
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_payload(payload: list[int] | None, bits: int | None) -> None:
    """Reject arbitrary payloads for key/presence detectors before loading a GPU."""
    if payload is None:
        return
    if bits is None:
        raise ValueError("This configuration is a key/presence detector, not a payload encoder.")
    if len(payload) != bits or any(
        type(value) is not int or value not in (0, 1) for value in payload
    ):
        raise ValueError(f"Expected exactly {bits} binary integer payload values.")


def save_state(metadata_path: str | Path, metadata: dict[str, Any], state: Any) -> None:
    """Write JSON + compressed numeric arrays without executable pickle content.

    The JSON is the completion marker and is committed last. Secret keys stay
    in these local artifacts; public result tables need only their SHA-256 IDs.
    """
    import numpy as np

    target = Path(metadata_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}

    def pack(value: Any) -> Any:
        if hasattr(value, "detach"):
            array = value.detach().cpu()
            # NumPy cannot represent torch's experimental complex32 dtype.
            if str(array.dtype) == "torch.complex32":
                array = array.to(dtype=__import__("torch").complex64)
            return {"__tensor__": pack(array.numpy())}
        if hasattr(value, "tocsr"):
            matrix = value.tocsr()
            return {
                "__csr__": [
                    pack(matrix.data),
                    pack(matrix.indices),
                    pack(matrix.indptr),
                    list(matrix.shape),
                ]
            }
        if isinstance(value, np.ndarray):
            name = f"array_{len(arrays):04d}"
            arrays[name] = np.asarray(value)
            return {"__array__": name}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return {"__bytes__": value.hex()}
        if isinstance(value, tuple):
            return {"__tuple__": [pack(item) for item in value]}
        if isinstance(value, list):
            return [pack(item) for item in value]
        if isinstance(value, dict):
            return {str(key): pack(item) for key, item in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported origin state type: {type(value).__name__}")

    tree = pack(state)
    archive = target.with_suffix(".state.npz")
    partial = archive.with_name(archive.name + ".part")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, archive)
    document = dict(metadata)
    document.update(state=tree, state_file=archive.name, state_sha256=file_sha256(archive))
    pending = target.with_name(target.name + ".part")
    pending.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(pending, target)


def load_state(metadata_path: str | Path, device: str = "cpu") -> tuple[dict[str, Any], Any]:
    """Verify state bytes before reconstructing numeric arrays and tensors."""
    import numpy as np

    target = Path(metadata_path)
    document = json.loads(target.read_text(encoding="utf-8"))
    archive = (target.parent / document["state_file"]).resolve()
    if archive.parent != target.parent.resolve():
        raise ValueError("State archive must reside beside its metadata file.")
    if file_sha256(archive) != document["state_sha256"]:
        raise ValueError("Origin key/state archive checksum mismatch.")
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}

    def unpack(value: Any) -> Any:
        if isinstance(value, list):
            return [unpack(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "__tensor__" in value:
            import torch

            return torch.from_numpy(unpack(value["__tensor__"])).to(device)
        if "__array__" in value:
            return arrays[value["__array__"]]
        if "__csr__" in value:
            from scipy.sparse import csr_matrix

            data, indices, indptr, shape = value["__csr__"]
            return csr_matrix((unpack(data), unpack(indices), unpack(indptr)), shape=tuple(shape))
        if "__bytes__" in value:
            return bytes.fromhex(value["__bytes__"])
        if "__tuple__" in value:
            return tuple(unpack(item) for item in value["__tuple__"])
        return {key: unpack(item) for key, item in value.items()}

    return document, unpack(document["state"])


def _seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _namespace(name: str, folder: Path) -> None:
    """Import a source package without its unrelated CLI/training __init__."""
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module


class OriginAdapter:
    """Configuration and sidecar contract shared by independent origin workers."""

    bits: int | None = None

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        repo_root: str = "",
        weights: str = "",
        device: str = "cuda",
        model_id: str = "",
    ) -> None:
        self.options = dict(options or {})
        self.repo = Path(self.options.get("repo_root", repo_root)).resolve()
        self.weights = Path(
            self.options.get("weights", self.options.get("asset_root", weights))
        ).resolve()
        self.device = self.options.get("device", device)
        self.model_id = self.options.get("model_id", model_id) or self.__class__.__name__
        self.steps = int(self.options.get("steps", 50))
        self.inversion_steps = int(self.options.get("inversion_steps", self.steps))
        self.size = int(self.options.get("image_size", 512))
        self.guidance = float(self.options.get("guidance", 7.5))
        self.key_seed = int(self.options.get("key_seed", 999999))
        self.make_clean = bool(self.options.get("make_clean", True))
        self.backbone = str(self.options.get("backbone_path", self.weights / "backbone"))
        self.dtype_name = self.options.get("dtype", "float16")
        if self.options.get("mode", "native") != "native":
            raise ValueError("Origin adapters support native generation, not matched-PSNR scaling.")
        if self.size != 512:
            raise ValueError("This pinned diffusion-origin track is defined at 512 × 512.")

    def _prepare(self) -> None:
        if not self.repo.is_dir():
            raise FileNotFoundError(f"Source checkout missing: {self.repo}")
        if not (Path(self.backbone) / "model_index.json").is_file():
            raise FileNotFoundError(f"Pinned backbone snapshot missing: {self.backbone}")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

    def initialize(self) -> None:
        """Preflight: load the actual models before the worker fingerprints assets."""
        self._load()

    def _metadata(
        self,
        prompt: str,
        seed: int,
        payload: list[int] | None,
        output_path: str,
        clean_path: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "schema": 1,
            "model_id": self.model_id,
            "track": "origin",
            "mode": "native",
            "prompt": prompt,
            "seed": seed,
            "key_seed": self.key_seed,
            "payload": payload,
            "expected_bits": payload,
            "image_path": str(Path(output_path).resolve()),
            "image_sha256": file_sha256(output_path),
            "clean_path": clean_path,
            "image_size": self.size,
            "steps": self.steps,
            "inversion_steps": self.inversion_steps,
            "guidance": self.guidance,
            "options": self.options,
            **extra,
        }

    def _read(self, path: str, device: str | None = None) -> tuple[dict[str, Any], Any]:
        metadata, state = load_state(path, device or self.device)
        if metadata["model_id"] != self.model_id or metadata["options"] != self.options:
            raise ValueError(
                "Model/options differ from the generation sidecar; use its exact configuration."
            )
        return metadata, state

    @staticmethod
    def _save_image(image: Any, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + ".part")
        image.convert("RGB").save(pending, format="PNG")
        os.replace(pending, target)

    @staticmethod
    def _clean_path(output_path: str) -> str:
        path = Path(output_path)
        return str(path.with_name(path.stem + "_clean.png").resolve())


class DiffusionOrigin(OriginAdapter):
    """Common native SD2.1 generation and author DDIM inversion."""

    pipeline_module = "inverse_stable_diffusion"
    utils_module = "optim_utils"

    def _load(self) -> None:
        if hasattr(self, "pipe"):
            return
        import torch
        from diffusers import DPMSolverMultistepScheduler

        self._prepare()
        pipeline_cls = importlib.import_module(
            self.pipeline_module
        ).InversableStableDiffusionPipeline
        self.utils = importlib.import_module(self.utils_module)
        self.dtype = getattr(torch, self.dtype_name)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            self.backbone, subfolder="scheduler"
        )
        self.pipe = pipeline_cls.from_pretrained(
            self.backbone, scheduler=scheduler, torch_dtype=self.dtype
        ).to(self.device)
        self.text_embeddings = self.pipe.get_text_embedding("")
        self.pipe.set_progress_bar_config(disable=False)

    def _sample(self, prompt: str, latents: Any, pipe: Any = None) -> Any:
        import torch

        with torch.no_grad():
            return (pipe or self.pipe)(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                height=self.size,
                width=self.size,
                latents=latents.to(self.device, dtype=self.dtype),
            ).images[0]

    def _invert(self, path: str) -> Any:
        import torch
        from PIL import Image

        with Image.open(path) as image:
            tensor = self.utils.transform_img(image.convert("RGB"), target_size=self.size)
        with torch.no_grad():
            latent = self.pipe.get_image_latents(
                tensor.unsqueeze(0).to(self.device, self.dtype), sample=False
            )
            return self.pipe.forward_diffusion(
                latents=latent,
                text_embeddings=self.text_embeddings,
                guidance_scale=1,
                num_inference_steps=self.inversion_steps,
            )

    def _save_pair(
        self,
        prompt: str,
        seed: int,
        marked_latents: Any,
        clean_latents: Any,
        output_path: str,
        clean_pipe: Any = None,
    ) -> str | None:
        _seed(seed)
        self._save_image(self._sample(prompt, marked_latents), output_path)
        clean_path = None
        if self.make_clean:
            clean_path = self._clean_path(output_path)
            _seed(seed)
            self._save_image(self._sample(prompt, clean_latents, clean_pipe), clean_path)
        return clean_path

    def generate_clean(self, prompt: str, seed: int, output_path: str) -> None:
        """Independent unwatermarked carrier for held-out negative calibration."""
        self._load()
        _seed(seed)
        latents = self.pipe.get_random_latents()
        image = self._sample(prompt, latents, getattr(self, "clean_pipe", None))
        self._save_image(image, output_path)


class TreeRingAdapter(DiffusionOrigin):
    """Official Tree-Ring ring pattern and continuous negative-L1 score."""

    def _args(self) -> Any:
        return types.SimpleNamespace(
            w_seed=self.key_seed,
            w_channel=3,
            w_pattern="ring",
            w_mask_shape="circle",
            w_radius=10,
            w_measurement="l1_complex",
            w_injection="complex",
            w_pattern_const=0,
        )

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        self._load()
        args = self._args()
        patch = self.utils.get_watermarking_pattern(self.pipe, args, self.device)
        # Upstream pattern construction changes all RNGs. Reset before carrier generation.
        _seed(seed)
        clean = self.pipe.get_random_latents()
        mask = self.utils.get_watermarking_mask(clean, args, self.device)
        marked = self.utils.inject_watermark(clean.clone(), mask, patch, args)
        clean_path = self._save_pair(prompt, seed, marked, clean, output_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                None,
                output_path,
                clean_path,
                output_kind="score",
                score_name="negative_l1_complex",
            ),
            {"patch": patch, "mask": mask},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        self._load()
        _, state = self._read(metadata_path)
        latent = self._invert(input_path)
        _, distance = self.utils.eval_watermark(
            latent, latent, state["mask"], state["patch"], self._args()
        )
        return {
            "score": -float(distance),
            "score_name": "negative_l1_complex",
            "higher_is_marked": True,
        }


class GaussianShadingAdapter(DiffusionOrigin):
    """Author ChaCha20 configuration: 256 payload bits, repetition 1 × 8 × 8."""

    bits = 256
    utils_module = "image_utils"

    def _watermark(self) -> Any:
        return importlib.import_module("watermark").Gaussian_Shading_chacha(1, 8, 1e-6, 1000000)

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, self.bits)
        if not self.device.startswith("cuda"):
            raise RuntimeError(
                "Author Gaussian Shading uses .cuda(); launch a CUDA-isolated worker."
            )
        import numpy as np
        import torch

        self._load()
        _seed(seed)
        bits = payload if payload is not None else [random.randrange(2) for _ in range(self.bits)]
        watermark = self._watermark()
        watermark.watermark = torch.tensor(bits, device=self.device).reshape(1, 4, 8, 8)
        expanded = watermark.watermark.repeat(1, 1, 8, 8).flatten().cpu().numpy()
        encrypted = watermark.stream_key_encrypt(expanded)
        marked = watermark.truncSampling(encrypted).to(self.device)
        # ChaCha keys are intentionally random and persisted, rather than regenerated from a public seed.
        _seed(seed)
        clean = self.pipe.get_random_latents()
        clean_path = self._save_pair(prompt, seed, marked, clean, output_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                bits,
                output_path,
                clean_path,
                output_kind="bits",
                clean_pairing="same_prompt_seed_not_same_latent",
            ),
            {
                "key": watermark.key,
                "nonce": watermark.nonce,
                "payload": np.asarray(bits, dtype=np.uint8),
                "initial_latents": marked,
            },
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        self._load()
        _, state = self._read(metadata_path)
        watermark = self._watermark()
        watermark.key, watermark.nonce = state["key"], state["nonce"]
        latent = self._invert(input_path)
        encrypted = (latent > 0).int().flatten().cpu().numpy()
        decoded = watermark.diffusion_inverse(watermark.stream_key_decrypt(encrypted))
        return {"bits": decoded.detach().cpu().flatten().int().tolist(), "native_bits": self.bits}


class METRAdapter(TreeRingAdapter):
    """Base METR with 10 binary rings; excludes separately trained METR++."""

    bits = 10
    pipeline_module = "metr.inverse_stable_diffusion"
    utils_module = "metr.optim_utils"

    def _prepare(self) -> None:
        super()._prepare()
        _namespace("metr", self.repo / "src" / "metr")

    def _args(self) -> Any:
        args = super()._args()
        args.use_random_msgs = True
        args.msg_type = "binary"
        args.msg_scaler = 100
        args.msg = "0" * self.bits
        return args

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, self.bits)
        self._load()
        _seed(seed)
        bits = payload if payload is not None else [random.randrange(2) for _ in range(self.bits)]
        args = self._args()
        message = "".join(map(str, bits))
        patch = self.utils.get_watermarking_pattern(self.pipe, args, self.device, message=message)
        _seed(seed)
        clean = self.pipe.get_random_latents()
        mask = self.utils.get_watermarking_mask(clean, args, self.device)
        marked = self.utils.inject_watermark(clean.clone(), mask, patch, args)
        clean_path = self._save_pair(prompt, seed, marked, clean, output_path)
        save_state(
            metadata_path,
            self._metadata(prompt, seed, bits, output_path, clean_path, output_kind="bits"),
            {},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        self._load()
        self._read(metadata_path)
        bits = self.utils.detect_msg(self._invert(input_path), self._args())
        return {"bits": [int(bit) for bit in bits], "native_bits": self.bits}


class RingIDAdapter(DiffusionOrigin):
    """Official verification pattern #628; outputs distance, not invented bits."""

    utils_module = "utils"

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        import itertools
        import torch

        self._load()
        _seed(self.key_seed)
        base = self.pipe.get_random_latents().to(torch.float64)
        single = torch.tensor(self.utils.ring_mask(size=64, r_out=14, r_in=3), device=self.device)
        channels = [0, 3]
        mask = torch.stack([single, single])
        # Select one combination without constructing all 2**11 GPU patterns.
        combo = next(itertools.islice(itertools.product((-64.0, 64.0), repeat=11), 628, None))
        pattern = self.utils.make_Fourier_ringid_pattern(
            self.device,
            [[value] for value in combo],
            base,
            radius=14,
            radius_cutoff=3,
            ring_watermark_channel=[3],
            heter_watermark_channel=[0],
            heter_watermark_region_mask=single.unsqueeze(0),
        )
        pattern = self.utils.fft(self.utils.ifft(pattern).real)
        pattern[:, [3]] = self.utils.fft(
            torch.fft.fftshift(self.utils.ifft(pattern[:, [3]]), dim=(-1, -2))
        )
        _seed(seed)
        clean = self.pipe.get_random_latents()
        marked = self.utils.generate_Fourier_watermark_latents(
            self.device, 14, 3, mask, channels, original_latents=clean, watermark_pattern=pattern
        )
        clean_path = self._save_pair(prompt, seed, marked, clean, output_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt, seed, None, output_path, clean_path, output_kind="score", key_index=628
            ),
            {"pattern": pattern, "mask": mask, "channels": channels},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        self._load()
        _, state = self._read(metadata_path)
        reversed_fft = self.utils.fft(self._invert(input_path))
        distance = self.utils.get_distance(
            state["pattern"],
            reversed_fft,
            state["mask"],
            p=1,
            mode="complex",
            channel_min=True,
            channel=state["channels"],
        )
        return {
            "score": -float(distance),
            "score_name": "negative_ringid_distance",
            "higher_is_marked": True,
        }


class MaXsiveAdapter(DiffusionOrigin):
    """Author template-injected pipeline and continuous correlation score."""

    pipeline_module = "inverse_watermark_stable_diffusion"
    utils_module = "image_utils"

    def _load(self) -> None:
        if hasattr(self, "pipe"):
            return
        import torch
        from diffusers import DPMSolverMultistepScheduler

        self._prepare()
        self.dtype = getattr(torch, self.dtype_name)
        from torchvision import transforms

        # Equivalent to the author's image_utils.transform_img, without
        # importing WAVE attack packages which this benchmark does not call.
        def transform_img(image: Any, target_size: int = 512) -> Any:
            transform = transforms.Compose(
                [
                    transforms.Resize(target_size),
                    transforms.CenterCrop(target_size),
                    transforms.ToTensor(),
                ]
            )
            return 2.0 * transform(image) - 1.0

        self.utils = types.SimpleNamespace(transform_img=transform_img)
        cls = importlib.import_module(self.pipeline_module).InversableStableDiffusionPipeline
        scheduler_cls = importlib.import_module(
            "modified_DPMSolver"
        ).Modified_DPMSolverMultistepScheduler
        scheduler = scheduler_cls.from_pretrained(self.backbone, subfolder="scheduler")
        self.pipe = cls.from_pretrained(
            self.backbone, scheduler=scheduler, torch_dtype=self.dtype
        ).to(self.device)
        # Share model weights, but use the author's unmodified generation path for negatives.
        clean_cls = importlib.import_module(
            "inverse_stable_diffusion"
        ).InversableStableDiffusionPipeline
        self.clean_pipe = clean_cls(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                self.backbone, subfolder="scheduler"
            ),
            safety_checker=self.pipe.safety_checker,
            feature_extractor=self.pipe.feature_extractor,
        )
        self.text_embeddings = self.pipe.get_text_embedding("")
        args = types.SimpleNamespace(
            channel_copy=1,
            hw_copy=2,
            template_c=3,
            distant_func="corr",
            diffusion_bit=16,
            tpr_file=None,
        )
        _namespace("models", self.repo / "models")
        self.watermark = importlib.import_module("models.MaXsive").MaXsive(args)

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        self._load()
        _seed(seed)
        marked, state = self.watermark.watermark_injection()
        _seed(seed)
        clean = self.clean_pipe.get_random_latents()
        clean_path = self._save_pair(prompt, seed, marked, clean, output_path, self.clean_pipe)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                None,
                output_path,
                clean_path,
                output_kind="score",
                clean_pairing="same_prompt_seed_not_same_latent",
            ),
            state,
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch

        self._load()
        _, state = self._read(metadata_path, device="cpu")
        latent = self._invert(input_path)
        restored = self.watermark.template_restore(latent)
        parts = self.watermark.k2_decode(restored, state["keys"][1])
        vote = self.watermark.voting(parts).reshape(-1).float().to(self.device)
        target = state["keys"][0].reshape(-1).float().to(self.device)
        correlation = torch.corrcoef(torch.stack([target, vote]))[0, 1].abs().item()
        return {
            "score": float(correlation),
            "score_name": "absolute_correlation",
            "higher_is_marked": True,
        }


class StableSignatureAdapter(OriginAdapter):
    """Official SD2 decoder + 48-bit extractor, fixed published message."""

    bits = 48
    secret = "111010110101000001010111010011010100010000100111"

    def _load(self) -> None:
        if hasattr(self, "pipe"):
            return
        import torch
        from diffusers import StableDiffusionPipeline

        self._prepare()
        _namespace("ldm", self.repo / "src" / "ldm")
        decoder_cls = importlib.import_module("ldm.modules.diffusionmodules.model").Decoder
        self.dtype = getattr(torch, self.dtype_name)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.backbone, torch_dtype=self.dtype
        ).to(self.device)
        # Fine-tuning removes encoder and quant_conv. Only the published decoder
        # and post_quant_conv are needed; constructing a second full SD is wasteful.
        component = torch.nn.Module()
        component.decoder = decoder_cls(
            double_z=True,
            z_channels=4,
            resolution=256,
            in_channels=3,
            out_ch=3,
            ch=128,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
        )
        component.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
        state = torch.load(self.weights / "sd2_decoder.pth", map_location="cpu", weights_only=True)
        state = state.get("ldm_decoder", state)
        # Exact load prevents a silently unwatermarked VAE due to wrong key names.
        component.load_state_dict(state, strict=True)
        self.marked_decoder = component.to(self.device, dtype=self.dtype).eval()
        self.clean_decode = self.pipe.vae.decode
        self.extractor = torch.jit.load(
            str(self.weights / "dec_48b_whit.torchscript.pt"), map_location=self.device
        ).eval()

    def generate_clean(self, prompt: str, seed: int, output_path: str) -> None:
        import torch

        self._load()
        self.pipe.vae.decode = self.clean_decode
        _seed(seed)
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                height=self.size,
                width=self.size,
            ).images[0]
        self._save_image(image, output_path)

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        fixed = [int(bit) for bit in self.secret]
        validate_payload(payload, self.bits)
        if payload is not None and payload != fixed:
            raise ValueError(
                "Published Stable Signature checkpoint has a fixed message; changing it requires training."
            )
        import torch

        self._load()

        def marked_decode(latent: Any, return_dict: bool = True, **kwargs: Any) -> Any:
            from diffusers.models.autoencoders.vae import DecoderOutput

            sample = self.marked_decoder.decoder(self.marked_decoder.post_quant_conv(latent))
            return DecoderOutput(sample=sample) if return_dict else (sample,)

        common = dict(
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.size,
            width=self.size,
        )
        with torch.no_grad():
            try:
                self.pipe.vae.decode = marked_decode
                _seed(seed)
                image = self.pipe(prompt, **common).images[0]
                self._save_image(image, output_path)
            finally:
                self.pipe.vae.decode = self.clean_decode
            clean_path = None
            if self.make_clean:
                _seed(seed)
                clean_path = self._clean_path(output_path)
                self._save_image(self.pipe(prompt, **common).images[0], clean_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                fixed,
                output_path,
                clean_path,
                output_kind="bits",
                message_policy="fixed_checkpoint",
            ),
            {},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torchvision import transforms

        self._load()
        self._read(metadata_path)
        with Image.open(input_path) as image:
            tensor = transforms.ToTensor()(image.convert("RGB"))
        tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
        with torch.no_grad():
            logits = self.extractor(tensor.unsqueeze(0).to(self.device))
        return {"bits": (logits > 0).flatten().int().cpu().tolist(), "native_bits": self.bits}


class PRCAdapter(OriginAdapter):
    """Released PRC presence detector, including its decoder-inversion step."""

    def _load(self) -> None:
        if hasattr(self, "pipe"):
            return
        if not self.device.startswith("cuda"):
            raise RuntimeError("Author exact inversion currently hardcodes CUDA.")
        self._prepare()
        _namespace("src", self.repo / "src")
        self.prc = importlib.import_module("src.prc")
        self.gaussians = importlib.import_module("src.pseudogaussians")
        self.inversion = importlib.import_module("inversion")
        self.pipe = self.inversion.stable_diffusion_pipe(
            solver_order=1, model_id=self.backbone, cache_dir=str(self.weights / "cache")
        )
        self.pipe.set_progress_bar_config(disable=False)

    def initialize(self) -> None:
        self._load()
        self._keys()

    def generate_clean(self, prompt: str, seed: int, output_path: str) -> None:
        import numpy as np
        import torch

        self._load()
        _seed(seed)
        latent = torch.from_numpy(np.random.randn(1, 4, 64, 64)).to(self.device)
        image, _, _ = self.inversion.generate(
            prompt=prompt,
            init_latents=latent,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            solver_order=1,
            gen_seed=seed,
            pipe=self.pipe,
        )
        self._save_image(image, output_path)

    def _keys(self) -> tuple[Any, Any, Path]:
        # KeyGen includes galois entropy not controlled by NumPy's legacy RNG.
        # Generate once, persist, and reuse this exact private key across the run.
        key_path = self.weights / f"prc_key_{self.key_seed}.json"
        key_config = {"schema": 1, "n": 4 * 64 * 64, "fpr": 1e-5, "t": 3}
        if key_path.exists():
            stored_config, stored = load_state(key_path)
            if any(stored_config[key] != value for key, value in key_config.items()):
                raise ValueError("Existing PRC key has different parameters.")
            encoding_key = self._restore_key(stored["encoding"], encoding=True)
            decoding_key = self._restore_key(stored["decoding"], encoding=False)
        else:
            _seed(self.key_seed)
            encoding_key, decoding_key = self.prc.KeyGen(
                key_config["n"], false_positive_rate=key_config["fpr"], t=3
            )
            save_state(key_path, key_config, {"encoding": encoding_key, "decoding": decoding_key})
        return encoding_key, decoding_key, key_path

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        import numpy as np
        import torch

        self._load()
        encoding_key, decoding_key, key_path = self._keys()
        _seed(seed)
        codeword = self.prc.Encode(encoding_key)
        marked = self.gaussians.sample(codeword).reshape(1, 4, 64, 64).to(self.device)
        image, _, _ = self.inversion.generate(
            prompt=prompt,
            init_latents=marked,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            solver_order=1,
            gen_seed=seed,
            pipe=self.pipe,
        )
        self._save_image(image, output_path)
        clean_path = None
        if self.make_clean:
            _seed(seed)
            clean = torch.from_numpy(np.random.randn(1, 4, 64, 64)).to(self.device)
            clean_image, _, _ = self.inversion.generate(
                prompt=prompt,
                init_latents=clean,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                solver_order=1,
                gen_seed=seed,
                pipe=self.pipe,
            )
            clean_path = self._clean_path(output_path)
            self._save_image(clean_image, clean_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                None,
                output_path,
                clean_path,
                output_kind="decision",
                key_id=file_sha256(key_path),
                clean_pairing="same_prompt_seed_not_same_latent",
            ),
            {"decoding": decoding_key, "initial_latents": marked},
        )

    def _restore_key(self, key: Any, encoding: bool) -> tuple[Any, ...]:
        key = list(key)
        fields = (0, 1, 2) if encoding else (0, 2, 5)
        for index in fields:
            key[index] = self.prc.GF(key[index])
        return tuple(key)

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch
        from PIL import Image

        self._load()
        _, state = self._read(metadata_path, device="cpu")
        key = self._restore_key(state["decoding"], encoding=False)
        with Image.open(input_path) as image:
            # This path optimizes the VAE inverse. Do not wrap in inference_mode.
            with torch.enable_grad():
                latent = self.inversion.exact_inversion(
                    image.convert("RGB"),
                    prompt="",
                    test_num_inference_steps=self.inversion_steps,
                    guidance_scale=self.guidance,
                    inv_order=0,
                    pipe=self.pipe,
                )
        posterior = self.gaussians.recover_posteriors(
            latent.detach().double().flatten().cpu(), variances=1.5
        )
        detected = bool(self.prc.Detect(key, posterior))
        decodable = self.prc.Decode(key, posterior) is not None
        return {
            "detected": detected or decodable,
            "detector_decision": detected,
            "decoder_accepts": decodable,
            "output_kind": "decision",
        }


class AutoregressiveOrigin(OriginAdapter):
    """AR workers use separate generator/tokenizer assets, not a diffusion VAE."""

    def _prepare(self) -> None:
        if not self.repo.is_dir():
            raise FileNotFoundError(f"Source checkout missing: {self.repo}")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

    def _require_native_cuda(self) -> None:
        import torch

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This pinned autoregressive configuration requires a CUDA GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "This native BF16 configuration requires an Ampere-or-newer CUDA GPU; no silent FP32 fallback."
            )

    @staticmethod
    def _rgb_tensor(tensor: Any, normalized: bool = False) -> Any:
        from PIL import Image
        import numpy as np

        array = tensor.detach().float().cpu()
        if array.ndim == 4:
            array = array[0]
        if array.shape[0] == 3:
            array = array.permute(1, 2, 0)
        if normalized:
            array = (array.clamp(-1, 1) + 1) * 127.5 + 0.5
        return Image.fromarray(np.clip(array.numpy(), 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _checkpoint(path: Path) -> dict[str, Any]:
        from wmbench.checkpoints import load_tensor_checkpoint

        state = load_tensor_checkpoint(path, namespace_metadata=True)
        for key in ("model", "module", "state_dict", "ema"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        return state


class IndexMarkAdapter(AutoregressiveOrigin):
    """Official 512px text-to-image LlamaGen + released IndexMark encoder."""

    def _load(self) -> None:
        if hasattr(self, "gpt"):
            return
        import torch

        self._require_native_cuda()
        self._prepare()
        # The author's CLI disables initialization globally. Restore those
        # methods immediately: strict checkpoint loads must remain meaningful.
        linear_reset, norm_reset = (
            torch.nn.Linear.reset_parameters,
            torch.nn.LayerNorm.reset_parameters,
        )
        try:
            self.source = importlib.import_module("generation")
        finally:
            torch.nn.Linear.reset_parameters, torch.nn.LayerNorm.reset_parameters = (
                linear_reset,
                norm_reset,
            )
        self.vq = self.source.VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
        self.vq.load_state_dict(self._checkpoint(self.weights / "vq_ds16_t2i.pt"), strict=True)
        self.vq = self.vq.eval().requires_grad_(False).to(self.device)
        self.gpt = self.source.GPT_models["GPT-XL"](
            block_size=1024, cls_token_num=120, model_type="t2i"
        )
        self.gpt.load_state_dict(
            self._checkpoint(self.weights / "t2i_XL_stage2_512.pt"), strict=True
        )
        self.gpt = self.gpt.eval().requires_grad_(False).to(self.device, dtype=torch.bfloat16)
        self.text = self.source.T5Embedder(
            self.device,
            dir_or_name="flan-t5-xl",
            local_cache=True,
            cache_dir=str(self.weights / "t5"),
            torch_dtype=torch.bfloat16,
            model_max_length=120,
        )
        mapping_path = self.repo / "codebook_mwpm" / "codebook_index_mapping_knn10_mwpm.json"
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        self.mapping = {int(red): int(green) for red, green in raw_mapping.items()}
        if len(self.mapping) != 8192 or len(set(self.mapping.values())) != 8192:
            raise ValueError("The pinned IndexMark mapping must contain 8192 disjoint pairs.")
        if set(self.mapping) & set(self.mapping.values()):
            raise ValueError("IndexMark red and green lists must be disjoint.")
        self.mapping_sha256 = file_sha256(mapping_path)
        self.hc = types.SimpleNamespace(
            index_mapping=self.mapping,
            red_list_tensor=torch.tensor(list(self.mapping), device=self.device),
        )
        from wmbench.checkpoints import load_tensor_checkpoint

        encoder = load_tensor_checkpoint(
            self.weights / "Index_encoder_512.pt", namespace_metadata=True
        )
        self.vq.encoder.load_state_dict(encoder["encoder_state_dict"], strict=True)

    def _sample_indices(self, prompt: str, seed: int) -> tuple[Any, Any]:
        import torch

        _seed(seed)
        embeddings, masks = self.text.get_text_embeddings([prompt])
        left_embeddings = []
        for embedding, mask in zip(embeddings, masks):
            valid = int(mask.sum().item())
            left_embeddings.append(torch.cat([embedding[valid:], embedding[:valid]]))
        masks = torch.flip(masks, dims=[-1])
        conditioning = torch.stack(left_embeddings) * masks[:, :, None]
        return self.source.generate(
            self.gpt,
            conditioning,
            1024,
            masks,
            cfg_scale=self.guidance,
            temperature=1.0,
            top_k=1000,
            top_p=1.0,
            sample_logits=True,
            index_mapping=self.mapping,
        )

    def generate_clean(self, prompt: str, seed: int, output_path: str) -> None:
        import torch

        self._load()
        with torch.no_grad():
            indices, _ = self._sample_indices(prompt, seed)
            image = self._rgb_tensor(self.vq.decode_code(indices, [1, 8, 32, 32]), normalized=True)
        self._save_image(image, output_path)

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        import numpy as np
        import torch

        self._load()
        with torch.no_grad():
            indices, confidence = self._sample_indices(prompt, seed)
            original = confidence[:, :, 0].float().cpu().numpy().flatten()
            paired = confidence[:, :, 1].float().cpu().numpy().flatten()
            ratio = np.divide(
                original, paired, out=np.full_like(original, np.inf), where=paired != 0
            )
            log_ratio = np.log10(ratio, out=np.full_like(ratio, -np.inf), where=ratio > 0)
            finite = log_ratio[np.isfinite(log_ratio)]
            percentile = float(self.options.get("replacement_percentile", 100))
            threshold = np.percentile(finite, percentile) if finite.size else -np.inf
            marked, count = self.source.vectorized_replacement_process(
                indices, log_ratio, threshold, self.hc
            )
            marked_image = self._rgb_tensor(
                self.vq.decode_code(marked, [1, 8, 32, 32]), normalized=True
            )
            self._save_image(marked_image, output_path)
            clean_path = None
            if self.make_clean:
                clean_path = self._clean_path(output_path)
                clean_image = self._rgb_tensor(
                    self.vq.decode_code(indices, [1, 8, 32, 32]), normalized=True
                )
                self._save_image(clean_image, clean_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                None,
                output_path,
                clean_path,
                output_kind="score",
                score_name="green_token_fraction",
                mapping_sha256=self.mapping_sha256,
                replacements=int(count),
                replacement_percentile=percentile,
                generator="LlamaGen-t2i-XL-512",
                clean_pairing="same_pre_replacement_indices",
            ),
            {
                "green_ids": np.asarray(list(self.mapping.values()), dtype=np.int64),
                "indices": marked,
            },
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        self._load()
        metadata, state = self._read(metadata_path)
        if metadata["mapping_sha256"] != self.mapping_sha256:
            raise ValueError("IndexMark codebook mapping differs from generation.")
        with Image.open(input_path) as image:
            image = image.convert("RGB")
            if image.size != (self.size, self.size):
                raise ValueError("The origin attack pipeline must restore images to 512 × 512.")
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
        with torch.no_grad():
            _, _, information = self.vq.encode(tensor.unsqueeze(0).to(self.device))
            indices = information[2].flatten()
            green_ids = torch.from_numpy(state["green_ids"]).to(indices.device)
            score = torch.isin(indices, green_ids).float().mean().item()
        return {
            "score": score,
            "score_name": "green_token_fraction",
            "higher_is_better": True,
            "output_kind": "score",
            "token_count": int(indices.numel()),
        }


class BitMarkAdapter(AutoregressiveOrigin):
    """Official Infinity-2B BitMark, native 512px bitwise sampling and z-score."""

    def _load(self) -> None:
        if hasattr(self, "generator"):
            return
        if not self.device.startswith("cuda"):
            raise RuntimeError(
                "The released BitMark/Infinity backend requires CUDA + FlashAttention."
            )
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        self._require_native_cuda()
        self._prepare()
        sys.path.insert(0, str(self.repo / "Infinity"))
        self.source = importlib.import_module("tools.run_infinity")
        architecture = importlib.import_module("architecture_wrapper")
        self.detector_source = importlib.import_module("detect_watermark")
        self.args = types.SimpleNamespace(
            architecture="infinity",
            vae_type=32,
            apply_spatial_patchify=0,
            pn="0.25M",
            vae_path=str(self.weights / "infinity_vae_d32reg.pth"),
            model_path=str(self.weights / "infinity_2b_reg.pth"),
            model_type="infinity_2b",
            checkpoint_type="torch",
            cache_dir=str(self.weights / "cache"),
            enable_model_cache=0,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            use_bit_label=1,
            add_lvl_embeding_only_first_block=1,
            text_channels=2048,
            use_flex_attn=0,
            bf16=1,
            watermark_scales=2,
            watermark_delta=2.0,
            watermark_method="2-bit_pattern",
            watermark_count_bit_flip=0,
            watermark_add_noise=0,
            watermark_remove_duplicates=0,
            set="01,10",
        )
        vae_module = importlib.import_module("infinity.models.bsq_vae.vae")
        author_cnn_load = vae_module.load_cnn

        def checked_cnn_load(model: Any, *args: Any, **kwargs: Any) -> Any:
            loaded_model, remaining, loaded_keys = author_cnn_load(model, *args, **kwargs)
            missing = set(dict(loaded_model.named_parameters())) - set(loaded_keys)
            if missing:
                raise ValueError(
                    f"BitMark VAE checkpoint missed trainable tensors: {sorted(missing)[:10]}"
                )
            return loaded_model, remaining, loaded_keys

        vae_module.load_cnn = checked_cnn_load
        try:
            self.vae = architecture.get_vae(self.args)
        finally:
            vae_module.load_cnn = author_cnn_load
        if (self.vae.tgt_h, self.vae.tgt_w) != (self.size, self.size):
            raise ValueError(
                "The pinned Infinity scale schedule does not match the experiment resolution."
            )
        generator_class = self.source.Infinity
        from .bitmark_statistics import repair_bitmark_statistics

        repair_bitmark_statistics(generator_class)
        author_load = generator_class.load_state_dict

        def strict_generator_load(model: Any, state_dict: Any, **kwargs: Any) -> Any:
            kwargs["strict"] = True
            return author_load(model, state_dict, **kwargs)

        generator_class.load_state_dict = strict_generator_load
        try:
            self.generator = self.source.load_transformer(self.vae.vae, self.args)
        finally:
            generator_class.load_state_dict = author_load
        # Author load_tokenizer ignores its path argument and downloads latest.
        # Loading exactly the same classes here fixes that provenance problem.
        text_path = str(self.weights / "t5" / "flan-t5-xl")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            text_path, local_files_only=True, legacy=True
        )
        self.tokenizer.model_max_length = 512
        self.text_encoder = (
            T5EncoderModel.from_pretrained(
                text_path, local_files_only=True, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.watermark = self.detector_source.WatermarkInference(self.args, self.vae)
        clean_args = types.SimpleNamespace(**vars(self.args))
        clean_args.watermark_scales, clean_args.watermark_delta = 0, 0.0
        self.clean_watermark = self.detector_source.WatermarkInference(clean_args, self.vae)
        self.detector = self.detector_source.get_detector(self.args)

    def _generate_image(self, prompt: str, seed: int, watermark: Any) -> Any:
        import torch

        _seed(seed)
        with torch.no_grad():
            _, _, image = self.source.gen_one_img(
                self.generator,
                self.vae.vae,
                self.tokenizer,
                self.text_encoder,
                [prompt],
                cfg_list=self.guidance,
                tau_list=0.5,
                scale_schedule=self.vae.scale_schedule,
                g_seed=seed,
                cfg_insertion_layer=[0],
                vae_type=32,
                sampling_per_bits=1,
                enable_positive_prompt=0,
                watermark=watermark,
                decode_per_scale=False,
            )
        return self._rgb_tensor(image)

    def generate_clean(self, prompt: str, seed: int, output_path: str) -> None:
        self._load()
        self._save_image(self._generate_image(prompt, seed, self.clean_watermark), output_path)

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        validate_payload(payload, None)
        self._load()
        self._save_image(self._generate_image(prompt, seed, self.watermark), output_path)
        clean_path = None
        if self.make_clean:
            clean_path = self._clean_path(output_path)
            self._save_image(self._generate_image(prompt, seed, self.clean_watermark), clean_path)
        save_state(
            metadata_path,
            self._metadata(
                prompt,
                seed,
                None,
                output_path,
                clean_path,
                output_kind="score",
                score_name="z_score",
                generator="Infinity-2B-0.25M",
                clean_pairing="same_prompt_sampling_seed",
                green_patterns="01,10",
                watermark_delta=2.0,
                watermark_scales=list(self.watermark.scales),
            ),
            {},
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch

        self._load()
        self._read(metadata_path)
        with torch.no_grad():
            result = self.detector_source.detect(
                self.args, str(input_path), self.detector, self.vae
            )
        return {
            "score": float(result["z_score"]),
            "score_name": "z_score",
            "higher_is_better": True,
            "green_fraction": float(result["green_fraction"]),
            "output_kind": "score",
        }


class IntegrationRequiredAdapter(OriginAdapter):
    """Explicit catalog entry for an AR integration not implemented in this release."""

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str,
        metadata_path: str,
        payload: list[int] | None = None,
    ) -> None:
        raise NotImplementedError(
            f"{self.model_id}: {self.options.get('readiness_reason', 'AR integration required')}"
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.model_id}: {self.options.get('readiness_reason', 'AR integration required')}"
        )
