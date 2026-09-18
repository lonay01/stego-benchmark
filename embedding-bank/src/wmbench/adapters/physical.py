"""Physical-specialist inference bridges, isolated from the notebook kernel.

Each worker imports at most one upstream repository. RGB PNG is the boundary;
channel order, fixed model sizes and ECC are explicit inside each adapter.
See ``docs/physical_compatibility.md`` for provenance and untested components.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def validate_payload(payload: list[int], length: int) -> list[int]:
    """Require the exact native payload; never silently truncate or pad it."""
    if len(payload) != length or any(bit not in (0, 1) for bit in payload):
        raise ValueError(f"Expected exactly {length} binary native bits, got {len(payload)}.")
    return [int(bit) for bit in payload]


def crc6_append(payload: list[int]) -> list[int]:
    """Systematic CRC with Fang's polynomial x^6+x^3+1, zero initialization."""
    data = validate_payload(payload, 30)
    work = data + [0] * 6
    polynomial = (1, 0, 0, 1, 0, 0, 1)
    for index in range(len(data)):
        if work[index]:
            for offset, bit in enumerate(polynomial):
                work[index + offset] ^= bit
    return data + work[-6:]


def strip_parallel_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Undo DataParallel's one leading ``module.`` without renaming layers."""
    return {key.removeprefix("module."): value for key, value in state.items()}


def matlab_literal(value: str | Path) -> str:
    """Quote a MATLAB char vector (not shell text)."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, target)


def _save_rgb(path: str | Path, pixels: Any) -> None:
    import numpy as np
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, target)


def _read_rgb(path: str | Path) -> Any:
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def _module_from_repo(repo: Path, filename: str, name: str) -> Any:
    path = repo / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream source: {path}. Run prepare first.")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _PhysicalAdapter:
    native_bits = 0

    def __init__(self, options: dict[str, Any]):
        self.options = dict(options)
        self.repo = Path(options.get("repo_root", ".")).resolve()
        self.device = str(options.get("device", "cpu"))
        self.model_id = str(options.get("model_id", type(self).__name__))

    def initialize(self) -> None:
        """Load/verify weights during preparation, before the asset lock is hashed."""
        loader = getattr(self, "_load", None)
        if loader is not None:
            loader()

    def _weight(self, default: str, key: str = "checkpoint") -> Path:
        supplied = self.options.get("weights")
        if isinstance(supplied, dict):
            value = supplied.get(key) or self.options.get(key) or default
        else:
            value = supplied if key == "checkpoint" and supplied else self.options.get(key, default)
        path = Path(value)
        if not path.is_absolute():
            path = self.repo / path
        if not path.is_file():
            raise FileNotFoundError(f"{self.model_id}: required {key} is missing: {path}")
        return path.resolve()

    def _metadata(self, path: str, payload: list[int], **details: Any) -> None:
        _write_json(
            path,
            {
                "schema_version": 1,
                "model_id": self.model_id,
                "native_bits": len(payload),
                "payload": payload,
                **details,
            },
        )


class StegaStampONNX(_PhysicalAdapter):
    """WIBE's public ONNX checkpoint; raw 100-bit, no text/BCH wrapper.

    This is identified as the ONNX benchmark implementation, not asserted
    equivalent to the unavailable original TensorFlow SavedModel.
    """

    native_bits = 100

    def _load(self) -> Any:
        if not hasattr(self, "session"):
            import onnxruntime as ort

            path = self._weight("model_files/stega_stamp/stega_stamp.onnx")
            providers = ["CPUExecutionProvider"]
            if self.device.startswith("cuda"):
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError(
                        "StegaStamp CUDA requested, but ONNX Runtime CUDA provider is unavailable."
                    )
                gpu = int(self.device.split(":")[1]) if ":" in self.device else 0
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": gpu}))
            self.session = ort.InferenceSession(str(path), providers=providers)
            expected_in = {"image", "secret"}
            expected_out = {"stegastamp", "residual", "decoded"}
            if not expected_in.issubset({item.name for item in self.session.get_inputs()}):
                raise ValueError(
                    "Checkpoint does not expose the inspected WIBE StegaStamp input names."
                )
            if not expected_out.issubset({item.name for item in self.session.get_outputs()}):
                raise ValueError(
                    "Checkpoint does not expose the inspected WIBE StegaStamp output names."
                )
        return self.session

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        import cv2
        import numpy as np

        payload = validate_payload(payload, self.native_bits)
        session = self._load()
        original = _read_rgb(input_path)
        resized = cv2.resize(original, (400, 400), interpolation=cv2.INTER_LINEAR)
        marked = session.run(
            ["stegastamp"],
            {
                "image": resized[None].astype(np.float32) / 255,
                "secret": np.asarray([payload], dtype=np.float32),
            },
        )[0][0]
        quantized = np.rint(np.clip(marked, 0, 1) * 255).astype(np.int16)
        difference = quantized - resized.astype(np.int16)
        minimum = int(difference.min())
        # Reproduce WIBE's published uint8 difference-resize path exactly.
        difference = cv2.resize(
            (difference - minimum).astype(np.uint8),
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        output = np.clip(original.astype(np.int16) + difference.astype(np.int16) + minimum, 0, 255)
        _save_rgb(output_path, output)
        self._metadata(
            metadata_path,
            payload,
            implementation="wibe_onnx",
            ecc="none_raw100",
            model_resolution=400,
            size_adapter="wibe_uint8_residual_resize",
            runtime_providers=session.get_providers(),
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import cv2
        import numpy as np

        image = cv2.resize(_read_rgb(input_path), (400, 400), interpolation=cv2.INTER_LINEAR)
        result = self._load().run(
            ["decoded"],
            {
                "image": image[None].astype(np.float32) / 255,
                "secret": np.zeros((1, 100), dtype=np.float32),
            },
        )[0]
        return {
            "bits": (result.reshape(-1) > 0.5).astype(int).tolist(),
            "details": {"ecc": "none_raw100", "implementation": "wibe_onnx"},
        }


class _ScreenCNN(_PhysicalAdapter):
    """Official PIMoG/SSRH Encoder and Decoder, with the original BGR input."""

    native_bits = 30
    checkpoint = ""

    def _load(self) -> Any:
        if not hasattr(self, "network"):
            import torch

            module = _module_from_repo(self.repo, "model.py", f"wmbench_upstream_{self.model_id}")
            network = module.Encoder_Decoder("Identity")
            state = torch.load(self._weight(self.checkpoint), map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            network.load_state_dict(strip_parallel_prefix(state), strict=True)
            self.network = network.to(self.device).eval()
        return self.network

    def _tensor(self, pixels: Any) -> Any:
        import cv2
        import numpy as np
        import torch

        resized = cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_LINEAR)
        # Both original loaders use cv2.imread: their network input is BGR.
        normalized = resized[:, :, ::-1].astype(np.float32) / 127.5 - 1
        return torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].to(self.device)

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        import cv2
        import numpy as np
        import torch

        payload = validate_payload(payload, self.native_bits)
        original = _read_rgb(input_path)
        x = self._tensor(original)
        with torch.inference_mode():
            result = self._load().Encoder(
                x, torch.tensor([payload], device=self.device, dtype=torch.float32)
            )
        encoded_rgb = ((result[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5)[
            :, :, ::-1
        ]
        native_rgb = cv2.resize(original, (128, 128), interpolation=cv2.INTER_LINEAR)
        residual = np.clip(encoded_rgb, 0, 255) - native_rgb.astype(np.float32)
        residual = cv2.resize(
            residual, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        _save_rgb(output_path, np.rint(np.clip(original.astype(np.float32) + residual, 0, 255)))
        self._metadata(
            metadata_path,
            payload,
            model_resolution=128,
            network_color="BGR",
            size_adapter="float_residual_resize",
            ecc="none",
            adaptation="Original author network is fixed 128px; residual lifted to protocol resolution.",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch

        with torch.inference_mode():
            raw = self._load().Decoder(self._tensor(_read_rgb(input_path))).reshape(-1)
        bits = raw.detach().cpu().numpy().round().clip(0, 1).astype(int).tolist()
        return {
            "bits": bits,
            "details": {"native_resolution": 128, "network_color": "BGR", "ecc": "none"},
        }


class PIMoG(_ScreenCNN):
    checkpoint = "models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth"


class SSRH(_ScreenCNN):
    checkpoint = "models/SSRH.pth"


class SepWater(_PhysicalAdapter):
    """Raw 100-bit SepWater inference with all encoder/decoder keys required."""

    native_bits = 100

    def _load(self) -> Any:
        if not hasattr(self, "network"):
            import torch
            from omegaconf import OmegaConf

            checkpoint = self._weight("pre_trained_models/sepwater.ckpt")
            config_path = self._weight("pre_trained_models/inference.yaml", "config")
            sys.path.insert(0, str(self.repo))
            from Latent_Diffusion_Model.util import instantiate_from_config

            config = OmegaConf.load(config_path).model
            length = int(config.params.control_config.params.secret_len)
            if length != self.native_bits:
                raise ValueError(
                    f"SepWater configuration has {length} bits; raw100 adapter requires 100."
                )
            config.params.decoder_config.params.secret_len = length
            # These modules are used only for training. Omitting them leaves the
            # inspected inference graph unchanged and avoids LPIPS/noise startup.
            config.params.noise_config = "__none__"
            config.params.loss_config = {"target": "torch.nn.Identity"}
            network = instantiate_from_config(config)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = strip_parallel_prefix(state.get("state_dict", state))
            missing, unexpected = network.load_state_dict(state, strict=False)
            critical = [key for key in missing if key.startswith(("ae.", "control.", "decoder."))]
            if critical:
                raise RuntimeError(
                    f"SepWater checkpoint is incomplete for inference: {critical[:12]}"
                )
            self.load_details = {
                "missing_training_keys": list(missing),
                "ignored_checkpoint_keys": list(unexpected),
            }
            self.network = network.to(self.device).eval()
        return self.network

    def _tensor(self, pixels: Any) -> Any:
        import numpy as np
        import torch
        from PIL import Image

        resized = np.asarray(Image.fromarray(pixels).resize((224, 224), Image.Resampling.BILINEAR))
        array = resized.astype(np.float32) / 127.5 - 1
        return torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(self.device)

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        import numpy as np
        import torch

        payload = validate_payload(payload, self.native_bits)
        original = _read_rgb(input_path)
        cover = self._tensor(original)
        model = self._load()
        with torch.inference_mode():
            latent = model.encode_first_stage(cover)
            marked_latent, _ = model(
                latent, None, torch.tensor([payload], device=self.device, dtype=torch.float32)
            )
            stego = model.decode_first_stage(marked_latent)
            residual = torch.nn.functional.interpolate(
                stego.clamp(-1, 1) - cover,
                size=original.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        residual = residual[0].cpu().numpy().transpose(1, 2, 0)
        marked = (np.clip(residual + original.astype(np.float32) / 127.5 - 1, -1, 1) + 1) * 127.5
        _save_rgb(output_path, marked.astype(np.uint8))
        self._metadata(
            metadata_path,
            payload,
            model_resolution=224,
            size_adapter="author_residual_resize",
            ecc="none_raw100",
            load_details=self.load_details,
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        import torch

        with torch.inference_mode():
            bits = (self._load().decoder(self._tensor(_read_rgb(input_path))) > 0).reshape(-1)
        return {
            "bits": bits.cpu().int().tolist(),
            "details": {"ecc": "none_raw100", "model_resolution": 224},
        }


class _MatlabAdapter(_PhysicalAdapter):
    """MATLAB subprocess bridge; Octave is deliberately not substituted."""

    def initialize(self) -> None:
        """Check licensed MATLAB and callable toolbox functions before paid jobs."""
        with tempfile.TemporaryDirectory(prefix="wmbench_matlab_preflight_") as temporary:
            self._run(
                "assert(exist('ssim','file')~=0,'Image Processing Toolbox is required');\n"
                "assert(exist('bchenc','file')~=0,'Communications Toolbox is required');\n"
                "assert(exist('bchdec','file')~=0,'Communications Toolbox is required');\n",
                Path(temporary),
            )

    def _run(self, script: str, work: Path) -> None:
        executable = self.options.get("matlab_executable") or shutil.which("matlab")
        if not executable:
            raise RuntimeError(
                "Licensed MATLAB was not found. Configure matlab_executable; Image Processing and Communications Toolboxes are required."
            )
        source = work / "wmbench_job.m"
        source.write_text(
            "set(0,'DefaultFigureVisible','off');\n"
            + f"addpath(genpath({matlab_literal(self.repo)}));\n"
            + script,
            encoding="utf-8",
        )
        result = subprocess.run(
            [str(executable), "-batch", f"run({matlab_literal(source)})"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=float(self.options.get("timeout_seconds", 900)),
            check=False,
        )
        (work / "matlab.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(
                f"MATLAB {self.model_id} failed: {(result.stdout + result.stderr)[-4000:]}"
            )


class FangScreen(_MatlabAdapter):
    """Fang's feature-selected watermark with original 30-bit CRC/BCH payload."""

    native_bits = 30

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        payload = validate_payload(payload, self.native_bits)
        image = _read_rgb(input_path)
        if image.shape[:2] != (512, 512):
            raise ValueError("Fang adapter requires the author's 512×512 carrier geometry.")
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        crc = " ".join(map(str, crc6_append(payload)))
        with tempfile.TemporaryDirectory(prefix="wmbench_fang_") as temporary:
            work = Path(temporary)
            result_path = work / "marked.png"
            script = (
                f"I=imread({matlab_literal(Path(input_path).resolve())});\n"
                f"coded=bchenc(gf([{crc}]),63,36); W=[coded.x 0]; W=reshape(W,[8 8]);\n"
                "marked=Embedding_test(I,W);\n"
                f"imwrite(uint8(marked),{matlab_literal(result_path)});\nclose all;\n"
            )
            self._run(script, work)
            shutil.copy2(result_path, output)
        self._metadata(
            metadata_path,
            payload,
            ecc="CRC6_x6+x3+1_BCH63_36_pad64",
            model_resolution=512,
            matlab_source="Embedding_test.m",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="wmbench_fang_decode_") as temporary:
            work = Path(temporary)
            result_path = work / "decoded.json"
            script = (
                f"I=imresize(imread({matlab_literal(Path(input_path).resolve())}),[512 512]);\n"
                "raw=reshape(Extracting_test(I),1,64); [decoded,errors]=bchdec(gf(raw(1:63)),63,36);\n"
                "result=struct('decoded',decoded.x,'raw',raw,'bch_errors',errors);\n"
                f"fid=fopen({matlab_literal(result_path)},'w'); fprintf(fid,'%s',jsonencode(result)); fclose(fid);\n"
            )
            self._run(script, work)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        decoded = [int(bit) for bit in result["decoded"]]
        bits = decoded[:30]
        valid = result["bch_errors"] >= 0 and decoded == crc6_append(bits)
        return {
            "bits": bits,
            "details": {
                "raw_codeword": result["raw"],
                "ecc_valid": valid,
                "bch_errors": result["bch_errors"],
                "ecc": "CRC6_BCH63_36",
            },
        }


def xiao_headless_script(
    source: str,
    input_path: Path,
    output_path: Path,
    state_path: Path,
    payload: list[int],
    seed: int,
) -> str:
    """Parameterize only UI/I/O in the pinned author script; keep its algorithm."""
    payload = validate_payload(payload, 24)
    text = source.replace("\r\n", "\n")
    begin = text.find("[filename,pathname,index] = uigetfile(")
    end_marker = "I = imread([pathname,filename]);"
    end = text.find(end_marker, begin)
    tail = text.find("%Watermark extraction")
    if begin < 0 or end < 0 or tail < 0 or "W = randi([0,1],L,1);" not in text:
        raise RuntimeError(
            "Xiao main.m differs from the inspected revision; refusing an unsafe source rewrite."
        )
    text = (
        text[:begin]
        + f"I=imread({matlab_literal(input_path)});"
        + text[end + len(end_marker) : tail]
    )
    text = text.replace("clc; clear;", "clc;")
    text = text.replace("rng('shuffle');", f"rng({int(seed)},'twister');")
    text = text.replace("W = randi([0,1],L,1);", "W=[" + ";".join(map(str, payload)) + "];")
    text = text.replace("imwrite(II_ergb,'en.png');", "% encrypted intermediate is not exported")
    text = text.replace(
        "imwrite(I_rergb,'w.png');", f"imwrite(I_rergb,{matlab_literal(output_path)});"
    )
    text += (
        f"\nsave({matlab_literal(state_path)},'emb_loc','num_region','th','Wide_R','Wide_C','M','N');\n"
        "close all;\n"
    )
    return text


class XiaoSW(_MatlabAdapter):
    """Original client-side embedding script; 24 useful bits, not common30."""

    native_bits = 24

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        payload = validate_payload(payload, self.native_bits)
        source = (self.repo / "main.m").read_text(encoding="utf-8-sig")
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        state_path = Path(metadata_path).resolve().with_suffix(".mat")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        seed = int(self.options.get("seed", 0))
        with tempfile.TemporaryDirectory(prefix="wmbench_xiao_") as temporary:
            work = Path(temporary)
            marked = work / "marked.png"
            state = work / "state.mat"
            script = xiao_headless_script(
                source, Path(input_path).resolve(), marked, state, payload, seed
            )
            self._run(script, work)
            shutil.copy2(marked, output)
            shutil.copy2(state, state_path)
        self._metadata(
            metadata_path,
            payload,
            ecc="BCH63_24_pad64",
            key_path=str(state_path),
            seed=seed,
            comparison_track="separate_payload24",
            matlab_source="main.m_headless",
        )

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        state = Path(metadata["key_path"])
        if not state.is_file():
            raise FileNotFoundError(f"Xiao extraction key sidecar missing: {state}")
        with tempfile.TemporaryDirectory(prefix="wmbench_xiao_decode_") as temporary:
            work = Path(temporary)
            output = work / "decoded.json"
            script = (
                f"load({matlab_literal(state)}); I=imread({matlab_literal(Path(input_path).resolve())});\n"
                "I=imresize(I,[M N]); raw=watermark_extract(I,Wide_R,Wide_C,emb_loc,num_region,th);\n"
                "[decoded,errors]=bchdec(gf(reshape(raw(1:63),1,63)),63,24);\n"
                "result=struct('bits',decoded.x,'raw',raw,'bch_errors',errors);\n"
                f"fid=fopen({matlab_literal(output)},'w'); fprintf(fid,'%s',jsonencode(result)); fclose(fid);\n"
            )
            self._run(script, work)
            result = json.loads(output.read_text(encoding="utf-8"))
        return {
            "bits": [int(bit) for bit in result["bits"]],
            "details": {
                "raw_codeword": result["raw"],
                "bch_errors": result["bch_errors"],
                "ecc_valid": result["bch_errors"] >= 0,
                "comparison_track": "separate_payload24",
            },
        }


class UnavailablePhysicalCandidate(_PhysicalAdapter):
    """Keep unresolved catalogue entries visible without fabricating results."""

    def _fail(self) -> None:
        reason = self.options.get(
            "unavailable_reason", "No verified inference/asset contract has been supplied."
        )
        raise RuntimeError(f"{self.model_id} is inventory-only: {reason}")

    def initialize(self) -> None:
        self._fail()

    def embed(
        self, input_path: str, output_path: str, payload: list[int], metadata_path: str
    ) -> None:
        self._fail()

    def decode(self, input_path: str, metadata_path: str) -> dict[str, Any]:
        self._fail()
        raise AssertionError("unreachable")
