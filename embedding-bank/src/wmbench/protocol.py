"""Scientific defaults, payload contracts and model-independent paired random seeds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BASE_SEED = 20260811
LEGACY_DATASET_HASH = "900dcad2face902185209805ef2d934e1e503b02f6ca354bda086be983e2d073"
WIBE_COMMIT = "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
BACKBONES = {
    "sdxl": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "revision": "462165984030d82259a11f4367a4eed129e94a7b",
    },
    "ip2p": {
        "repo_id": "timbrooks/instruct-pix2pix",
        "revision": "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d",
    },
    "inpaint": {
        "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "revision": "115134f363124c53c7d878647567d04daf26e41e",
    },
    "dino": {
        "repo_id": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    },
    "clip": {
        "repo_id": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    },
}


def paired_seed(image_id: str, attack_id: str, repeat: int, base: int = BASE_SEED) -> int:
    material = f"{base}|{image_id}|{attack_id}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def payload_bank() -> list[list[int]]:
    """Exactly the previous notebook's five balanced messages, not a new random bank."""
    bank = []
    for index in range(5):
        bits = np.array([0] * 15 + [1] * 15, dtype=np.uint8)
        np.random.default_rng(BASE_SEED + index * 9176).shuffle(bits)
        bank.append(bits.tolist())
    return bank


def full_payload(
    common: list[int],
    length: int,
    image_id: str,
    reserved: dict | None = None,
    model_id: str = "",
    payload_id: int = 0,
) -> tuple[list[int], list[int]]:
    if length <= 0:
        return [], []
    key = f"{BASE_SEED}|{model_id}|{payload_id}|{length}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    bits = np.random.default_rng(seed).integers(0, 2, size=length, dtype=np.uint8).tolist()
    count = min(len(common), length)
    indices = list(range(length - count, length))
    bits[-count:] = common[-count:]
    for position, value in (reserved or {}).items():
        if int(position) in indices:
            raise ValueError("Reserved native bit overlaps the common-message interval")
        bits[int(position)] = int(value)
    return bits, indices


def attack_manifest(repeats: int = 3, identity_only: bool = False) -> list[dict]:
    attacks = [{"id": "identity", "family": "identity", "engine": "identity", "repeat": 0}]
    if identity_only:
        return attacks
    attacks.extend(
        {"id": f"vae_rt_{passes}", "family": "vae", "engine": "vae", "passes": passes, "repeat": 0}
        for passes in (1, 2)
    )
    stochastic = [
        {
            "id": f"sdxl_i2i_s{int(s * 100):03d}",
            "family": "img2img",
            "engine": "sdxl",
            "strength": s,
            "guidance_scale": 5.0,
        }
        for s in (0.15, 0.30, 0.50)
    ] + [
        {
            "id": "ip2p_golden_hour",
            "family": "editing",
            "engine": "ip2p",
            "prompt": "make the image look like it was taken at golden hour",
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.8,
        },
        {
            "id": "ip2p_watercolor",
            "family": "editing",
            "engine": "ip2p",
            "prompt": "turn the image into a watercolor painting",
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.2,
        },
        {
            "id": "sdxl_inpaint_center25",
            "family": "inpainting",
            "engine": "inpaint",
            "strength": 0.99,
            "guidance_scale": 7.5,
            "mask_box": [128, 128, 384, 384],
        },
    ]
    for attack in stochastic:
        for repeat in range(repeats):
            attacks.append(
                {
                    **attack,
                    "repeat": repeat,
                    "steps": 20,
                    "scheduler": "EulerAncestralDiscreteScheduler"
                    if attack["engine"] == "ip2p"
                    else "checkpoint_config",
                }
            )
    return attacks


def load_registry(root: Path) -> dict[str, dict]:
    import json

    models = {}
    for path in sorted((root / "registry").glob("*_models.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data if isinstance(data, list) else data.get("models", [])
        for row in rows:
            if row["id"] in models:
                raise ValueError(f"Duplicate model ID: {row['id']}")
            models[row["id"]] = row
    return models
