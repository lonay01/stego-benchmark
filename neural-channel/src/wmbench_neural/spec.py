"""Frozen scientific settings; scheduling never changes an attack or seed."""

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


BACKBONES = {
    "sdxl": ["stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b"],
    "ip2p": ["timbrooks/instruct-pix2pix", "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"],
    "inpaint": ["diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "115134f363124c53c7d878647567d04daf26e41e"],
}
ATTACKS = [dict(id="clean", family="clean", level="clean", repeat=0, engine="clean")]
for passes in (1, 2):
    ATTACKS.append(dict(id=f"vae_{passes}", family="vae", level=str(passes), repeat=0,
                        engine="vae", passes=passes))
for strength in (.15, .30, .50):
    for repeat in (1, 2, 3):
        ATTACKS.append(dict(id=f"sdxl_{strength:.2f}_r{repeat}", family="diffusion",
            level=f"{strength:.2f}", repeat=repeat, engine="sdxl", strength=strength,
            steps=20, guidance_scale=5.0))
for level, prompt, scale in (
    ("lighting", "make the image look like it was taken at golden hour", 1.8),
    ("style", "turn the image into a watercolor painting", 1.2),
):
    for repeat in (1, 2, 3):
        ATTACKS.append(dict(id=f"ip2p_{level}_r{repeat}", family="instruction", level=level,
            repeat=repeat, engine="ip2p", prompt=prompt, image_guidance_scale=scale,
            steps=20, guidance_scale=7.5))
for repeat in (1, 2, 3):
    ATTACKS.append(dict(id=f"inpaint_r{repeat}", family="inpaint", level="central_25pct",
        repeat=repeat, engine="inpaint", strength=.99, steps=20, guidance_scale=7.5))
BY_ID = {a["id"]: a for a in ATTACKS}
POLICY = {
    "version": "neural-bank-v1", "attacks": ATTACKS, "backbones": BACKBONES,
    "bootstrap": {"seed": 71209, "resamples": 2000},
    "seed_schema": "neural-seeds-v1/image_id/message_replicate/attack_id",
    "geometry": "saved native PNG -> common Pillow 10.4 BICUBIC 512 -> neural attack -> RGB8 PNG -> common BICUBIC native PNG -> unchanged decoder",
    "resize_only": "auxiliary diagnostic, excluded from the 21-condition matrix and macro",
    "quantization": "clip [0,1], numpy.rint(x*255), uint8; PNG compression 3",
    "mask": [128, 128, 384, 384], "negative_prompt": "",
    "clean": "exact saved native carrier; paired numerical-profile exception only for StegaStamp",
    "retention": "temporary attacked PNG ZIPs; delete only after verified durable decoding; retain hashes and receipts",
    "batching": "fixed width including duplicate padding; separate per-item CPU generators; no OOM batch-size fallback",
}


def seed(case, attack):
    key = [POLICY["seed_schema"], int(case["image_id"]), int(case["replicate"]), attack["id"]]
    return int(digest(key)[:8], 16)


def pack_id(attack, image_id):
    return f"{attack['id']}/{int(image_id):012d}"


def batches(items, width):
    """Padding fixes the numerical batch shape, including a five-image canary."""
    if width < 1:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(items), width):
        real = items[start:start+width]
        yield real + [real[-1]] * (width-len(real)), len(real)
