"""Fixed sources, cohorts and seeds. Runtime deadlines are not scientific settings."""

from pathlib import Path
from .storage import digest, disjoint, freeze, read

SOURCES = {
    "rosteals": ("TuBui/RoSteALS", "8c09b022a362ffdf977a86f355e5af339793d9c9"),
    "sepwater": ("CVhnu/sepwater", "4b99a4d3a99f504a09e5e743cffe745b8856e36a"),
    "distseal": ("facebookresearch/distseal", "d0a71cd272bb29e3b49cdc965bbaf4d24f2684ac"),
    "taming": ("CompVis/taming-transformers", "3ba01b241669f5ade541ce990f7650a3b8f65318"),
    "imagenetc": ("hendrycks/robustness", "8190fe329f5f072a06e3a2aea02bb5dda69aed9f"),
    "omnisealbench": ("facebookresearch/omnisealbench", "bd141394a0dfeac0b4cfedfdb4feaebb2a2c6db6"),
}
HF_DCAE = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"
HF_DCAE_REV = "d58275ea9674845e5e36cbf280987f67bee0e85c"
HF_DCAE_NATIVE = "mit-han-lab/dc-ae-f64c128-in-1.0"
MESSAGES = 5


def seed(*parts):
    return int(digest(["author-night-v1", *parts])[:8], 16) % (2**31 - 1)


def secret_text(image_id, repetition):
    # Seven ASCII characters encoded into 56 data positions in author ECC().
    # These hex strings are a text-CLI diagnostic, not 56 uniformly random bits.
    return digest(["text", str(image_id), repetition])[:7]


def generated_batches(per_class, batch_size):
    # Balanced prefixes: one image of EVERY class before the next repetition.
    for repetition in range(per_class):
        for start in range(0, 1000, batch_size):
            labels = list(range(start, min(start + batch_size, 1000)))
            yield {"key": f"r{repetition:02d}_c{start:04d}", "repetition": repetition,
                   "labels": labels, "seed": seed("generation", repetition, start)}


def make_plan(settings, code_hash):
    bank, root, scratch = [Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch")]
    disjoint(bank, root, scratch)
    n = settings["n_images"]
    if type(n) is not int or not 5 <= n <= 100:
        raise ValueError("N_COCO must be 5..100")
    if type(settings["generated_per_class"]) is not int or not 1 <= settings["generated_per_class"] <= 50:
        raise ValueError("GEN_PER_CLASS must be 1..50")
    if type(settings["generation_batch_size"]) is not int or not 1 <= settings["generation_batch_size"] <= 16:
        raise ValueError("GEN_BATCH_SIZE must be 1..16")
    if not 0 < float(settings["max_hours"]) <= 24:
        raise ValueError("MAX_HOURS must be in (0,24]")
    if root.exists() and not (root / "plan.json").exists():
        unknown = [p.name for p in root.iterdir() if p.name not in ("_release", "_logs", "_reports")]
        if unknown:
            raise ValueError(f"Use a new empty OUTPUT_ROOT, found {unknown[:5]}")
    manifest = read(bank / "_inputs/manifest.json")
    images = manifest["images"][:n]
    if len(images) != n or len({str(x["image_id"]) for x in images}) != n:
        raise ValueError("The original canary bank must contain N_COCO distinct source image IDs")
    plan = {"schema": "three-method-author-night-v1", "sources": SOURCES,
            "code_sha256": code_hash, "images": images, "messages_per_image": MESSAGES,
            "input": "COCO val2017 original JPEG for these fixed IDs; no benchmark centre crop",
            "profiles": {"rosteals": ["raw100", "author_bch56"],
                         "sepwater": ["raw100", "author_bch56"], "distseal": ["raw64"]},
            "sizes": {"rosteals": 256, "sepwater": 224, "distseal": 512},
            "generation": {"per_class": settings["generated_per_class"], "classes": 1000,
                "batch_size": settings["generation_batch_size"], "guidance": 1.0,
                "steps": 30, "latent_scale": 0.2889, "scaling_w": 0.5,
                "message": "official detector_dc-ae card fixed 64-bit signature",
                "order": "balanced prefixes, independently seeded batches, not original multi-GPU RNG order"},
            "cli_checks": 5, "training": False, "matched_psnr": False,
            "fid_reference": None, "inception_score": {"splits": 10, "seed": 20260930, "minimum_samples": 1000},
            "full_paper_reproduction": False,
            "paper_limits": ["COCO replaces RoSteALS/SepWater paper test datasets",
                            "SepWater256 checkpoint/config pair not located; 224 paper metric not established",
                            "ImageNet validation unavailable: FID blocked, never replaced with COCO",
                            "DistSeal selected post-hoc DCAE variant only; not all RAR/distillation experiments"]}
    # JSON normalizes tuples before freezing/comparison.
    import json
    plan = json.loads(json.dumps(plan))
    plan["identity"] = digest(plan)
    freeze(root / "plan.json", plan)
    return plan
