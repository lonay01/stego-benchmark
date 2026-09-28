"""Canary-only capacity settings and the previously diagnosed import repairs."""

from __future__ import annotations

import copy
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from wmbench.adapters.bank_additions import FNNSAdapter
from wmbench.adapters.extended import SSLWatermarkingAdapter
from wmbench.adapters.legacy import EditGuardAdapter
from wmbench.io import read_json, write_json
from wmbench.matching import load_rgb, save_rgb


def dependency_repairs(model):
    result = copy.deepcopy(model)
    if model["id"] in ("editguard", "sepwater"):
        result.pop("validated_lock", None)
        result["requirements"] = [r for r in result["requirements"] if not r.lower().startswith("setuptools")]
        result["requirements"] += ["setuptools==70.3.0", "wheel==0.45.1"]
    if model["id"] == "editguard":
        result["adapter"] = "wmbench_full_payload.compat:EditGuardImportFix"
        result["build_requirements"] = ["basicsr==1.4.2"]
    return result


@contextmanager
def constrained_builds(lock):
    from wmbench import environment

    original = environment.command

    def constrained(args, cwd=None, env=None):
        args = list(args)
        if "--no-build-isolation" in args:
            if not lock.is_file():
                raise FileNotFoundError(lock)
            args += ["--constraint", str(lock)]
        return original(args, cwd=cwd, env=env)

    environment.command = constrained
    try:
        yield
    finally:
        environment.command = original


class EditGuardImportFix(EditGuardAdapter):
    def __init__(self, options):
        import torchvision.transforms._functional_tensor as functional

        name = "torchvision.transforms.functional_tensor"
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.rgb_to_grayscale = functional.rgb_to_grayscale
            sys.modules[name] = module
        super().__init__(options)


class FNNSFullPayload(FNNSAdapter):
    def embed(self, input_path, output_path, payload, metadata_path):
        super().embed(input_path, output_path, payload, metadata_path)
        metadata = read_json(metadata_path)
        metadata.pop("common_indices", None)
        metadata.pop("filler_indices", None)
        metadata.update(message_indices=list(range(16, 48)), information_bits=32)
        write_json(metadata_path, metadata)


class SSLFullPayload(SSLWatermarkingAdapter):
    def __init__(self, options):
        super().__init__(options)
        if tuple(self.carrier.shape) != (2048, 2048) or self.native_bits != 2048:
            raise ValueError("Selected SSL normalization does not expose the expected 2048-dimensional limit")


class DWTFullPayload:
    """Same package/scales/block as the legacy baseline; full spatial capacity."""

    native_bits = 4096

    def __init__(self, options):
        from imwatermark import WatermarkEncoder, WatermarkDecoder
        import inspect
        import hashlib
        from imwatermark.dwtDctSvd import EmbedDwtDctSvd

        if int(options["native_bits"]) != 4096:
            raise ValueError("This canary fixes DWT capacity at 512px, block4, one active channel")
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder("bits", self.native_bits)
        self.implementation_sha256 = hashlib.sha256(inspect.getsource(EmbedDwtDctSvd).encode()).hexdigest()

    def embed(self, input_path, output_path, payload, metadata_path):
        import numpy as np

        image = load_rgb(input_path)
        if image.shape != (512, 512, 3) or len(payload) != self.native_bits:
            raise ValueError("Expected RGB512 and 4096 raw message bits")
        if any(b not in (0, 1) for b in payload):
            raise ValueError("Non-binary payload")
        self.encoder.set_watermark("bits", payload)
        marked = self.encoder.encode(image[:, :, ::-1].copy(), "dwtDctSvd", scales=[0, 36, 0], block=4)
        save_rgb(output_path, marked[:, :, ::-1].astype(np.uint8))
        write_json(metadata_path, {
            "expected_bits": payload, "ecc": "none", "native_bits": self.native_bits,
            "geometry": [512, 512], "block": 4, "scales": [0, 36, 0],
            "implementation_sha256": self.implementation_sha256,
        })

    def decode(self, input_path, metadata_path):
        if read_json(metadata_path)["native_bits"] != self.native_bits:
            raise ValueError("Wrong DWT message length")
        result = self.decoder.decode(load_rgb(input_path)[:, :, ::-1].copy(), "dwtDctSvd", scales=[0, 36, 0], block=4)
        return {"bits": result.astype(int).tolist(), "ecc": "none"}
