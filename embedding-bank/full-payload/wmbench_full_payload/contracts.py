"""Explicit capacity contracts and reproducible, uncoded information messages."""

from __future__ import annotations

import copy
import hashlib

from wmbench.embedding_bank import bank_registry

POLICY = {
    "schema": "full-payload-canary-v1",
    "seed": 20260921,
    "replicates": 5,
    "message": "shake256-full-payload-v1-msb-first",
    "targets": [39, 42, 45],
    "tolerance": 0.05,
    "max_alpha": 4.0,
    "iterations": 64,
    "matching": "saved_native_u8_scalar_bisection_v1",
    "reference": "original_rgb512",
    "diagnostics": "fin-hidden-distseal-replicate1-each-image-v1",
}
MODES = ("native", "matched_39", "matched_42", "matched_45")


def models_for_trial(project):
    """Override only this canary. Never edit the production registry in place."""
    from .compat import dependency_repairs

    models = copy.deepcopy(bank_registry(project))
    for name, model in models.items():
        if name == "dwt_dct_svd":
            # Haar LL at 512 -> 256; one message bit per 4x4 block in U.
            model["native_bits"] = (512 // 2 // 4) ** 2
            model["options"]["native_bits"] = model["native_bits"]
            model["adapter"] = "wmbench_full_payload.compat:DWTFullPayload"
        elif name == "ssl_watermarking":
            # Author generate_carriers(k,d) requires k<=d. Verify d at runtime.
            model["native_bits"] = 2048
            model["options"]["native_bits"] = 2048
            model["adapter"] = "wmbench_full_payload.compat:SSLFullPayload"
        elif name == "fnns_omniseal":
            model["adapter"] = "wmbench_full_payload.compat:FNNSFullPayload"
            model["payload_layout"] = {
                "detection_indices": list(range(16)),
                "message_indices": list(range(16, 48)),
            }
            model["capability"]["explanation"] = (
                "Pinned OmniSealBench FNNS: 16 fixed detection bits and all 32 variable message bits; "
                "no filler/common30 split in this canary. GPU full-payload check required."
            )
        models[name] = dependency_repairs(model)
    return models


def capacity(model):
    raw = int(model["native_bits"])
    reserved = {int(k): int(v) for k, v in model.get("reserved_bits", {}).items()}
    if raw < 0 or any(k < 0 or k >= raw or v not in (0, 1) for k, v in reserved.items()):
        raise ValueError("Invalid declared payload layout")
    positions = [i for i in range(raw) if i not in reserved]
    kind = "fixed_checkpoint_interface"
    evidence = "Pinned registry/checkpoint; shape and complete decode length checked during canary"
    if model["id"] == "dwt_dct_svd":
        kind = "structural_limit_at_fixed_geometry"
        evidence = "512x512, Haar LL256x256, block4, one active chroma plane: 4096 positions"
    elif model["id"] == "ssl_watermarking":
        kind = "orthogonal_carrier_dimension_limit"
        evidence = "Author k<=d; selected normalization output d=2048, asserted on loaded model"
    elif raw == 0:
        kind, evidence = "unresolved", "No verified message/embedding interface"
    return {
        "adapter_bits": raw,
        "information_bits": len(positions),
        "information_indices": positions,
        "reserved_bits": {str(k): v for k, v in sorted(reserved.items())},
        "coded_bits": 64 if model["id"] == "fang_screen" else raw,
        "interface": "post_native_crc_bch" if model["id"] == "fang_screen" else "raw_message",
        "bits_per_pixel": len(positions) / (512 * 512),
        "capacity_kind": kind,
        "basis": evidence,
        "error_free_capacity_claim": False,
    }


def message(model, image_id, replicate):
    if type(replicate) is not int or not 1 <= replicate <= POLICY["replicates"]:
        raise ValueError("Replicate must be 1..5")
    spec = capacity(model)
    length = spec["information_bits"]
    if not length:
        raise ValueError("No validated positive-length payload contract")
    key = f"{POLICY['message']}|{POLICY['seed']}|{int(image_id)}|{replicate}".encode()
    stream = hashlib.shake_256(key).digest((length + 7) // 8)
    information = [(byte >> shift) & 1 for byte in stream for shift in range(7, -1, -1)][:length]
    payload = [0] * spec["adapter_bits"]
    for position, bit in zip(spec["information_indices"], information):
        payload[position] = bit
    for position, bit in spec["reserved_bits"].items():
        payload[int(position)] = bit
    return {"payload": payload, "information": information, "capacity": spec}


def score(result, assignment):
    """A reference is consulted only after blind extraction returns."""
    bits = result.get("bits")
    expected, spec = assignment["payload"], assignment["capacity"]
    if bits is None or len(bits) != len(expected) or any(b not in (0, 1) for b in bits):
        raise ValueError("Decoder did not return the complete binary adapter-length message")
    bits = list(map(int, bits))
    errors = [i for i, (left, right) in enumerate(zip(bits, expected)) if left != right]
    count = sum(i in errors for i in spec["information_indices"])
    reserved_errors = [i for i in errors if str(i) in spec["reserved_bits"]]
    rejected = (result.get("details") or {}).get("ecc_valid") is False
    return {
        "status": "rejected" if rejected else "measured",
        "decoded_bits": bits,
        "expected_bits": expected,
        "information_bits": spec["information_bits"],
        "information_errors": None if rejected else count,
        "ber": None if rejected else count / spec["information_bits"],
        "bit_accuracy": None if rejected else 1 - count / spec["information_bits"],
        "exact": False if rejected else count == 0,
        "error_positions": errors,
        "reserved_errors": reserved_errors,
        "adapter_ber_diagnostic": len(errors) / len(bits),
        "interface": spec["interface"],
        "details": result.get("details"),
        "note": "ECC-rejected tentative bits are diagnostic only" if rejected else "",
    }
