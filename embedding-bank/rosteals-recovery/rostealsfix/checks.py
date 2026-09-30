"""Fail early on the actual package layout and author import/attack chain."""

import importlib
import importlib.util
from pathlib import Path

from authornight.storage import EvidenceError, read, sha256, write


def validate_module_origin(root, scratch):
    from authornight.spec import SOURCES
    repo, revision = SOURCES["imagenetc"]
    expected = (Path(scratch) / "sources" / (repo.split("/")[-1] + "-" + revision)
                / "ImageNet-C/imagenet_c/imagenet_c/__init__.py").resolve()
    spec = importlib.util.find_spec("imagenet_c")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise EvidenceError(f"Wrong imagenet_c module: {getattr(spec, 'origin', None)}; expected {expected}")
    relative = "ImageNet-C/imagenet_c/imagenet_c/__init__.py"
    audit = read(Path(root) / "_meta/sources/imagenetc.json")
    if sha256(expected) != audit["files"][relative]:
        raise EvidenceError("ImageNet-C package changed after source validation")
    return str(expected)


def targets(node):
    if isinstance(node, dict):
        if isinstance(node.get("target"), str):
            yield node["target"]
        for value in node.values():
            yield from targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from targets(value)


def import_and_attack_check(job):
    """Real imports and all 70 synthetic attack conditions, no reference bits."""
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf
    from authornight.latent import set_seed
    root, source = Path(job["output"]), Path(job["source"])
    origin = validate_module_origin(root, job["scratch"])
    from imagenet_c import corrupt, corruption_dict
    if not callable(corrupt) or len(corruption_dict) != 19:
        raise EvidenceError("Unexpected ImageNet-C API")
    imported = []
    config = OmegaConf.to_container(OmegaConf.load(source / "models/VQ4_mir_inference.yaml"), resolve=True)
    for target in sorted(set(targets(config))):
        module, attribute = target.rsplit(".", 1)
        getattr(importlib.import_module(module), attribute)
        imported.append(target)
        print("IMPORT OK:", target, flush=True)
    for module in ("tools.ecc", "tools.sifid", "tools.eval_metrics", "tools.helpers",
                   "cldm.loss", "lpips", "slack"):
        importlib.import_module(module)
        imported.append(module)
    from tools.augment_imagenetc import RandomImagenetC
    noise = RandomImagenetC(1, 5, "test")
    if len(noise.corrupt_ids) != 14:
        raise EvidenceError("Unexpected author test-pool size")
    from tools.ecc import ECC
    ecc = ECC()
    packet = ecc.encode_text(["secrets"])
    if packet.shape != (1, 100) or ecc.decode_text(packet.copy())[0] != "secrets":
        raise EvidenceError("Author ECC clean roundtrip failed")
    image = Image.fromarray(np.random.default_rng(20260930).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    checks = []
    for noise_id in noise.corrupt_ids:
        for severity in range(1, 6):
            set_seed(20260930 + int(noise_id) * 10 + severity)
            result = np.asarray(noise(image.copy(), int(noise_id), severity))
            if result.shape != (256, 256, 3) or result.dtype != np.uint8:
                raise EvidenceError("Invalid author attack output")
            checks.append([noise.method_names[int(noise_id)], severity])
        print(f"ATTACK API OK: {noise.method_names[int(noise_id)]}, severities 1..5", flush=True)
    evidence = {"passed": True, "imagenet_c_file": origin, "imports": imported,
                "synthetic_attack_conditions": checks, "author_ecc_clean_roundtrip": True,
                "note": "Synthetic API smoke only; not watermark robustness evidence"}
    write(root / "_meta/fix/import_check.json", evidence)
    print("IMPORT AND ATTACK CHECK PASSED: 70/70", flush=True)
    return evidence
