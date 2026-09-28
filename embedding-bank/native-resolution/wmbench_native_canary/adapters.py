"""Native carrier bridges. Weights, decoder thresholds and strengths stay fixed.

The observer is optional and never supplies a reference message to a decoder.
It observes the image returned by the selected implementation, not necessarily
an internal unquantized network tensor (PIL-based APIs quantize internally).
"""

from __future__ import annotations

import numpy as np

from wmbench.adapters.bank_additions import FINAdapter
from wmbench.adapters.extended import _validate_payload
from wmbench.adapters.legacy import WibeAdapter
from wmbench.adapters.physical import StegaStampONNX, validate_payload
from wmbench.io import write_json
from wmbench.matching import load_rgb, save_rgb


class FINNative(FINAdapter):
    """HEAVY FED encoder / INL+FED decoder on a literal 128px carrier."""

    def __init__(self, options):
        if options.get("resolution_policy") != "author_128_direct":
            raise ValueError("FINNative requires author_128_direct")
        # Reuse only the audited checkpoint loader, not the parent's embed path.
        super().__init__({**options, "resolution_policy": "residual_128_to_512"})
        self.options["resolution_policy"] = "author_128_direct"

    def embed(self, input_path, output_path, payload, metadata_path):
        bits = _validate_payload(payload, self.native_bits)
        image = self._image(input_path)
        if tuple(image.shape[-2:]) != (128, 128):
            raise ValueError("FINNative requires a 128x128 carrier")
        message = self.torch.tensor([bits], device=self.device, dtype=self.torch.float32) - 0.5
        with self.torch.inference_mode():
            marked, _ = self.fed([image * 2 - 1, message])
            result = (marked + 1) / 2
        self._save(result, output_path)
        self._metadata(metadata_path, bits, variant="HEAVY", ecc="none",
                       resolution_policy="author_128_direct",
                       preprocessing="RGB128 -> normalize -> FED; decoder inverse INL then FED",
                       message_domain=[-0.5, 0.5], decode_threshold=0.0)


class NativeWibe(WibeAdapter):
    """Same upstream operations as WibeAdapter, with an observable PNG boundary."""

    def decode_tensor(self, image, length):
        torch = self.torch
        image = image.detach().squeeze(0).cpu() if image.ndim == 4 else image.detach().cpu()
        with torch.no_grad():
            if self.seal is not None:
                result = self.seal.detect(image.unsqueeze(0).to(self.device), is_video=False)
                bits = (result["preds"][0, 1:] > 0).int().cpu().numpy()
            else:
                bits = self.wrapper.extract(image, self._payload([0] * length))
                if isinstance(bits, torch.Tensor):
                    bits = bits.detach().cpu().numpy()
        bits = np.asarray(bits).reshape(-1)
        if not np.isfinite(bits).all():
            raise ValueError("Non-finite decoder output")
        return {"bits": np.clip(np.rint(bits), 0, 1).astype(int).tolist(), "ecc": "none"}

    def embed(self, input_path, output_path, payload, metadata_path):
        torch = self.torch
        image, watermark = self._image(input_path), self._payload(payload)
        with torch.no_grad():
            if self.seal is not None:
                result = self.seal.embed(image.unsqueeze(0).to(self.device),
                    msgs=watermark.watermark.to(self.device), is_video=False,
                    lowres_attenuation=True)["imgs_w"].squeeze(0)
            elif self.model_id == "invismark":
                from wibench.utils import normalize_image, denormalize_image
                result = self.wrapper.invismark.model._encode(normalize_image(image).cpu(),
                    watermark.watermark.float().to(self.device))[0]
                result = denormalize_image(result)
            else:
                result = self.wrapper.embed(image, watermark)
        result = result.detach().squeeze().cpu()
        if tuple(result.shape) != tuple(image.shape) or not torch.isfinite(result).all():
            raise ValueError(f"Invalid encoder output: {tuple(result.shape)}")
        observer = getattr(self, "before_png", None)
        if observer is not None:
            observer(lambda: self.decode_tensor(result, len(payload)), "upstream_wrapper_return")
        output = torch.round(result.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        save_rgb(output_path, output)
        write_json(metadata_path, {"expected_bits": payload, "ecc": "none", "model_id": self.model_id,
            "dtype": "float32", "serialization": "clip_round_u8", "wrapper": self.options["wrapper"]})


class StegaStampNative(StegaStampONNX):
    """Same audited raw100 ONNX graph, without WIBE's residual-size transfer."""

    def embed(self, input_path, output_path, payload, metadata_path):
        payload = validate_payload(payload, self.native_bits)
        image = load_rgb(input_path)
        if image.shape != (400, 400, 3):
            raise ValueError("StegaStampNative requires a 400x400 carrier")
        session = self._load()
        marked = session.run(["stegastamp"], {"image": image[None].astype(np.float32) / 255,
                             "secret": np.asarray([payload], dtype=np.float32)})[0]
        if marked.shape != (1, 400, 400, 3) or not np.isfinite(marked).all():
            raise ValueError("Unexpected StegaStamp output")
        observer = getattr(self, "before_png", None)
        if observer is not None:
            def decode_float():
                bits = session.run(["decoded"], {"image": marked.astype(np.float32),
                                   "secret": np.zeros((1, 100), dtype=np.float32)})[0]
                if not np.isfinite(bits).all():
                    raise ValueError("Non-finite StegaStamp decode")
                return {"bits": (bits.reshape(-1) > 0.5).astype(int).tolist()}
            observer(decode_float, "onnx_stegastamp_float_output")
        save_rgb(output_path, np.rint(np.clip(marked[0], 0, 1) * 255).astype(np.uint8))
        self._metadata(metadata_path, payload, implementation="wibe_onnx", ecc="none_raw100",
                       model_resolution=400, size_adapter="none_direct_output",
                       runtime_providers=session.get_providers())


def validate_geometry(adapter, model):
    """Inspect exposed runtime fields. Absence is recorded, never invented."""
    size = model["native_geometry"]["width"]
    observations = {}
    resolution = getattr(adapter, "resolution", None)
    if resolution is not None:
        observations["adapter_resolution"] = int(resolution)
        if int(resolution) != size:
            raise ValueError(f"Selected geometry {size} != adapter resolution {resolution}")
    wrapper = getattr(adapter, "wrapper", None)
    params = getattr(wrapper, "params", None)
    for key in ("H", "W"):
        value = params.get(key) if isinstance(params, dict) else getattr(params, key, None)
        if value is not None:
            observations[f"wrapper.params.{key}"] = int(value)
            if model["id"] in ("hidden_benchmark", "cin", "rosteals", "vine_b", "vine_r") and int(value) != size:
                raise ValueError(f"Checkpoint {key}={value} does not match selected carrier {size}")
    # MBRS exposes the selected config on its Watermark class.
    config = getattr(getattr(wrapper, "wa", None), "settings", None)
    if model["id"] == "mbrs" and isinstance(config, dict):
        for key in ("H", "W"):
            if key in config:
                observations[f"wrapper.wa.settings.{key}"] = int(config[key])
                if int(config[key]) != size:
                    raise ValueError("MBRS checkpoint geometry mismatch")
    return {"selected_carrier": [size, size], "runtime_fields": observations,
            "note": "Saved encoder output shape is checked for every pair; unavailable config fields are not inferred."}
