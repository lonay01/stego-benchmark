"""Reviewed configurations, not score-selected variants or universal defaults."""

from copy import deepcopy

from wmbench.embedding_bank import bank_registry
from wmbench_full_payload.compat import dependency_repairs
from wmbench_full_payload.contracts import MODES, message as payload_message, score
from wmbench_native_canary.contracts import FIXED, capacity, crop_box

POLICY = {
    "schema": "production-bank-v1", "replicates": 5, "seed": 20260921,
    "input": "historical_rgb512_center_crop_no_resampling",
    "message": "shake256-full-payload-v1-msb-first",
    "targets": [39, 42, 45], "tolerance": 0.05, "max_alpha": 4.0,
    "iterations": 64, "matching": "saved_native_u8_scalar_bisection_v1",
    "clean": "blind_decode_reloaded_png", "attacks": [],
}
AUTHOR = {"rosteals": "rosteals", "sepwater": "sepwater", "distseal_posthoc_dcae": "distseal"}
EXPECTED = {
    "trustmark_q", "trustmark_b", "trustmark_p", "wam", "videoseal", "pixelseal",
    "chunkyseal", "vine_b", "vine_r", "rosteals", "invismark", "rivagan_benchmark",
    "dwt_dct_svd", "mbrs", "cin", "meta_hidden_revisited", "arwgan", "ssl_watermarking",
    "maskwm", "robust_wide", "sepmark", "stegastamp", "pimog", "ssrh", "editguard",
    "sepwater", "fin", "fnns_omniseal", "distseal_posthoc_dcae",
}


def models_for_run(project):
    models = deepcopy(bank_registry(project))
    if set(models) != EXPECTED:
        raise ValueError("Review changed catalogue: production requires the 29 selected entries")
    for name, model in models.items():
        size = 256 if name == "meta_hidden_revisited" else FIXED.get(name, 512)
        model["native_geometry"] = {"width": size, "height": size,
            "source_width": 512, "source_height": 512, "input_rule": POLICY["input"],
            "crop_box_xyxy": list(crop_box(size))}
        if name == "fin":
            model["adapter"] = "wmbench_native_canary.adapters:FINNative"
            model["options"]["resolution_policy"] = "author_128_direct"
        elif model["adapter"] == "wmbench.adapters.legacy:WibeAdapter":
            model["adapter"] = "wmbench_native_canary.adapters:NativeWibe"
        elif name == "stegastamp":
            model["adapter"] = "wmbench_native_canary.adapters:StegaStampNative"
        elif name == "fnns_omniseal":
            model["adapter"] = "wmbench_full_payload.compat:FNNSFullPayload"
        model = dependency_repairs(model)
        if name in AUTHOR:
            model.update(engine="author", author_name=AUTHOR[name],
                         adapter="wmbench_bank.adapters:AuthorAdapter")
            model["selected_profile"] = {
                "rosteals": "author_fix_full_config_256_raw100_export",
                "sepwater": "released_224_full_config_raw100_export",
                "distseal_posthoc_dcae": "author_posthoc_dcae_scale0.5_raw64_floor",
            }[name]
        else:
            model["engine"] = "registry"
        if name == "editguard":
            model["selected_profile"] = "copyright_degrade_checkpoint_not_best_of_clean_and_degrade"
            model["validated_lock"] = "locks/editguard.lock"
            model["python"] = "3.10.12"
        if name in ("dwt_dct_svd", "ssl_watermarking"):
            assert model["native_bits"] == (32 if name == "dwt_dct_svd" else 30)
        models[name] = model
    return models


def message(model, image_id, replicate):
    assignment = payload_message(model, image_id, replicate)
    assignment["capacity"] = capacity(model)
    return assignment
