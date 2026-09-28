"""Selected checkpoint contracts, not a claim of universal model defaults."""

from __future__ import annotations

import copy

from wmbench.embedding_bank import bank_registry
from wmbench_full_payload.compat import dependency_repairs
from wmbench_full_payload.contracts import capacity as payload_capacity
from wmbench_full_payload.contracts import message as payload_message
from wmbench_full_payload.contracts import score

POLICY = {
    "schema": "native-resolution-canary-v1",
    "mode": "native", "replicates": 5, "seed": 20260921,
    "message": "shake256-full-payload-v1-msb-first",
    "input": "historical_rgb512_center_crop_no_resampling",
    "strength": "selected_author_configuration_no_external_alpha",
    "payload": "all_information_positions_of_selected_checkpoint_or_configuration",
    "attacks": [], "matched_psnr": False,
    "serialization": "adapter_native_png_no_external_carrier_upscale",
}

EXCLUDED = {
    "omniguard": "No verified matching copyright checkpoint; not a failed decoding measurement.",
    "fang_screen": "Requires licensed MATLAB and toolboxes; excluded from this Python/Colab canary.",
    "camera_shooting_instance": "No verified executable adapter/checkpoint contract.",
}

# Small carriers are literal pixel crops, not a resize to these sizes.
# Larger/flexible carriers retain 512; an author's internal resize is retained.
FIXED = {
    "fin": 128, "hidden_benchmark": 128, "arwgan": 128,
    "pimog": 128, "ssrh": 128, "mbrs": 128, "cin": 128,
    "sepwater": 224,
    "rosteals": 256, "vine_b": 256, "vine_r": 256,
    "maskwm": 256, "sepmark": 256,
    "stegastamp": 400,
}


def models_for_trial(project):
    models = copy.deepcopy(bank_registry(project))
    # Guard against an unnoticed change to the historical catalogue.
    if len(models) != 32 or set(EXCLUDED) - set(models):
        raise ValueError("Catalogue changed: review the 29 selected / 3 excluded contracts")
    for name, model in models.items():
        size = FIXED.get(name, 512)
        model["native_geometry"] = {
            "width": size, "height": size, "source_width": 512, "source_height": 512,
            "input_rule": "center_crop_no_resampling",
            "kind": "fixed_checkpoint_carrier" if name in FIXED else "selected_512_carrier",
            "note": ("Use selected checkpoint's small carrier, without lifting its residual to 512."
                     if name in FIXED else
                     "Keep the 512 carrier; retain official internal resizing where present. Not a universal training-size claim."),
        }
        if name in EXCLUDED:
            model["canary_exclusion"] = EXCLUDED[name]
        if name == "fin":
            model["adapter"] = "wmbench_native_canary.adapters:FINNative"
            model["options"]["resolution_policy"] = "author_128_direct"
        elif model["adapter"] == "wmbench.adapters.legacy:WibeAdapter":
            model["adapter"] = "wmbench_native_canary.adapters:NativeWibe"
        elif name == "stegastamp":
            model["adapter"] = "wmbench_native_canary.adapters:StegaStampNative"
        elif name == "fnns_omniseal":
            model["adapter"] = "wmbench_full_payload.compat:FNNSFullPayload"
            model["payload_layout"] = {"detection_indices": list(range(16)), "message_indices": list(range(16, 48))}
        # Unlike the previous structural-limit test, retain configured DWT32/SSL30.
        if name in ("dwt_dct_svd", "ssl_watermarking"):
            expected = 32 if name == "dwt_dct_svd" else 30
            if int(model["native_bits"]) != expected:
                raise ValueError(f"Review changed default payload for {name}")
        models[name] = dependency_repairs(model)
    return models


def capacity(model):
    spec = payload_capacity(model)
    geometry = model["native_geometry"]
    spec["bits_per_pixel"] = spec["information_bits"] / (geometry["width"] * geometry["height"])
    if model["id"] in ("dwt_dct_svd", "ssl_watermarking"):
        spec.update(capacity_kind="selected_configurable_payload",
                    basis="Frozen baseline configuration, not structural maximum or reliable capacity")
    return spec


def message(model, image_id, replicate):
    assignment = payload_message(model, image_id, replicate)
    assignment["capacity"] = capacity(model)
    return assignment


def crop_box(size):
    if type(size) is not int or not 1 <= size <= 512:
        raise ValueError("Crop size must be an integer in 1..512")
    offset = (512 - size) // 2
    return offset, offset, offset + size, offset + size


def center_crop(image, size):
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("Expected the frozen RGB512 source, not an arbitrary COCO JPEG")
    return image.crop(crop_box(size))
