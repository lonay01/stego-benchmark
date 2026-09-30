"""Author DistSeal post-hoc DC-AE paths: COCO reconstruction and UViT generation.

No benchmark adapter, residual scaling, cropped payload or quality matching is
used. Generation embeds directly into sampled latents, never re-encoded RGB.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from .environment import download
from .latent import set_seed
from .metrics import bits_score, pixels, png_bytes, quality
from .spec import HF_DCAE, HF_DCAE_REV, HF_DCAE_NATIVE, generated_batches, seed
from .storage import EvidenceError, cached, commit, freeze, image_zip, read, sha256


def hf_snapshot(repo, root, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    receipt = root / "_meta/distseal" / (repo.split("/")[-1] + ".json")
    if receipt.exists():
        prior = read(receipt)
        if prior["repo"] != repo or revision and prior["revision"] != revision:
            raise EvidenceError("Autoencoder revision changed")
        revision = prior["revision"]
    elif revision is None:
        revision = HfApi().model_info(repo).sha
    freeze(receipt, {"repo": repo, "revision": revision})
    folder = Path(snapshot_download(repo, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.pt"]))
    files = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    freeze(receipt.with_name(receipt.stem + ".files.json"), files)
    return folder


def sample_pixels(signed):
    """Exact uint8 expression in scripts/sample_imagenet_dcae.py."""
    import torch
    return torch.clamp(127.5 * signed + 128.0, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


class AuthorDistSeal:
    def __init__(self, source, root, need_ae=True):
        import torch
        from omegaconf import OmegaConf
        self.source, self.root = Path(source), Path(root)
        os.chdir(source)
        sys.path.insert(0, str(source))
        from distseal.loader import load_message, load_watermarker
        from distseal.utils.cfg import get_config_from_checkpoint, setup_model
        card = OmegaConf.load(source / "cards/detector_dc-ae.yaml")
        weight = download(card.checkpoint, root / "_assets/distseal/detector_dcae_diffusion.pt")
        signature = download(card.message, root / "_assets/distseal/detector_dcae_diffusion_msg.npy")
        card.checkpoint, card.message = str(weight), str(signature)
        local_card = root / "_meta/distseal/detector.local.yaml"
        from .storage import atomic
        atomic(local_card, OmegaConf.to_yaml(card).encode())
        compatibility = None
        try:
            self.wm, self.message, scaling = load_watermarker(str(local_card), device="cpu")
        except AttributeError as error:
            config = get_config_from_checkpoint(weight)
            if config.args.get("attenuation") is not None or "lower" not in str(error):
                raise
            # Narrow compatibility fix only: both mean no attenuation. Preserve
            # the original failure in the audit; never alter weights/architecture.
            compatibility = {"original_error": repr(error), "field": "args.attenuation",
                             "before": None, "after": "none"}
            config.args.attenuation = "none"
            self.wm = setup_model(config, weight)
            self.message, scaling = load_message(str(signature)), card.scaling_w
        state = torch.load(weight, map_location="cpu", weights_only=True)["model"]
        keys = [k for k in self.wm.state_dict() if k.startswith(("embedder.", "detector."))]
        failed = [k for k in keys if k not in state or not torch.equal(self.wm.state_dict()[k].cpu(), state[k].cpu())]
        if not keys or failed:
            raise EvidenceError(f"Incomplete DistSeal inference weights: {failed[:10]}")
        if self.message.numel() != 64 or float(scaling) != .5:
            raise EvidenceError("Wrong author card signature or scale")
        freeze(root / "_meta/distseal/weight_audit.json", {
            "weight_sha256": sha256(weight), "signature_sha256": sha256(signature),
            "checked_tensors": len(keys), "all_equal": True, "compatibility": compatibility,
            "checkpoint_args": torch.load(weight, map_location="cpu", weights_only=True)["args"]})
        del state
        self.wm.blender.scaling_w = scaling
        self.wm.img_size = 512  # notebook's posthoc_dcae.ipynb
        self.wm = self.wm.cuda().eval().requires_grad_(False)
        self.message = self.message.cuda()
        self.ae = None
        if need_ae:
            from distseal.augmentation.neuralcompression import DCAutoencoder
            self.ae = DCAutoencoder(str(hf_snapshot(HF_DCAE, root, HF_DCAE_REV))).cuda().eval()
        import lpips
        self.lpips = lpips.LPIPS(net="alex").cuda().eval()

    def tensor(self, array):
        import torch
        return torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1)[None].float().cuda() / 255

    def decode(self, image):
        import torch
        from torchvision.transforms import Resize
        with torch.no_grad():
            # Same resize to extractor resolution as author's OmniSealBench wrapper.
            logits = self.wm.detector(Resize((512, 512))(image))[:, 1:]
        if logits.ndim != 2 or logits.shape[1] != 64 or not torch.isfinite(logits).all():
            raise EvidenceError(f"Unexpected DistSeal detector output {logits.shape}")
        return logits.detach().float().cpu().numpy()

    def quality(self, left, right):
        result = quality(pixels(left), pixels(right))
        result["lpips_raw"] = float(self.lpips(left * 2 - 1, right * 2 - 1).item())
        return result


def run_coco(job):
    import torch
    root = Path(job["output"])
    plan = read(root / "plan.json")
    model = AuthorDistSeal(Path(job["source"]), root)
    for item in read(root / "_inputs/originals.json"):
        input_path = root / item["path"]
        if sha256(input_path) != item["sha256"]:
            raise EvidenceError("Changed COCO input")
        # Exactly the author's notebook resize, rather than our centre crop.
        source = model.tensor(np.array(Image.open(input_path).convert("RGB").resize((512, 512))))
        for repetition in range(5):
            key = f"{item['image_id']:012d}_m{repetition}_raw64"
            relative = f"distseal/coco/records/{key}.json"
            if cached(root, relative, plan["identity"]):
                continue
            if time.time() >= job["deadline"]:
                return
            started = time.monotonic()
            set_seed(seed("distseal", "coco", item["image_id"], repetition))
            with torch.no_grad():
                message = model.wm.get_random_msg().cuda()
                payload = message.reshape(-1).int().tolist()
                if len(payload) != 64:
                    raise EvidenceError("Expected 64 author bits")
                latent, size = model.ae.encode_pre_quant(source)
                baseline = model.ae.decode(latent, size).clamp(0, 1)
                pred = model.wm.embedder(latent, message)
                marked_latent = model.wm.blender(latent, pred)
                raw = model.ae.decode(marked_latent, size)
                marked = raw.clamp(0, 1)
                scores = {"raw_unclipped": bits_score(model.decode(raw)[0], payload),
                          "raw_internal": bits_score(model.decode(marked)[0], payload),
                          "clean_control": bits_score(model.decode(source)[0], payload),
                          "ae_control": bits_score(model.decode(baseline)[0], payload)}
                images = {"ae.png": png_bytes(pixels(baseline))}
                for rule in ("floor", "round_even", "half_up"):
                    integer = pixels(marked, rule)
                    encoded = png_bytes(integer)
                    loaded = np.array(Image.open(io.BytesIO(encoded)))
                    scores["author_png" if rule == "floor" else rule + "_png"] = bits_score(model.decode(model.tensor(loaded))[0], payload)
                    images["author.png" if rule == "floor" else rule + ".png"] = encoded
                checks = {"png_pixels_equal": np.array_equal(pixels(marked), np.array(Image.open(io.BytesIO(images['author.png']))))}
                if repetition == 0:
                    again = model.ae.decode(model.wm.blender(latent, model.wm.embedder(latent, message)), size).clamp(0, 1)
                    checks["repeat_pixels_equal"] = np.array_equal(pixels(again), pixels(marked))
                    checks["repeat_bits_equal"] = np.array_equal(model.decode(again) > 0, model.decode(marked) > 0)
                qualities = {"source_to_author_png": quality(pixels(source), pixels(marked)),
                             "source_to_ae": model.quality(source, baseline),
                             "source_to_marked": model.quality(source, marked),
                             "ae_to_marked": model.quality(baseline, marked)}
            archive = root / "distseal/coco/artifacts" / (key + ".zip")
            image_zip(archive, images)
            commit(root, relative, plan["identity"], {"method": "distseal", "dataset": "COCO-val2017-original",
                "profile": "raw64", "image_id": item["image_id"], "repetition": repetition,
                "payload": payload, "seconds": time.monotonic()-started, "status": "measured",
                "measurements": scores, "quality": qualities, "checks": checks}, [archive])
            print(f"SAVED DistSeal COCO {key}; BER {100*scores['author_png']['ber']:.3f}%", flush=True)


def run_generation(job):
    import torch
    from omegaconf import OmegaConf
    root, source = Path(job["output"]), Path(job["source"])
    plan = read(root / "plan.json")
    model = AuthorDistSeal(source, root, need_ae=False)
    from distseal.loader import load_generator
    from deps.efficientvit.ae_model_zoo import DCAE_HF
    decoder = DCAE_HF.from_pretrained(str(hf_snapshot(HF_DCAE_NATIVE, root))).cuda().eval().requires_grad_(False)
    card = OmegaConf.load(source / "cards/uvit-h_base.yaml")
    weight = download(card.checkpoint, root / "_assets/distseal/uvit-h-in-512px-train2000k.pt")
    card.checkpoint = str(weight)
    from .storage import atomic
    local_card = root / "_meta/distseal/generator.local.yaml"
    atomic(local_card, OmegaConf.to_yaml(card).encode())
    generator = load_generator(str(local_card), device="cuda")
    freeze(root / "_meta/distseal/generator_audit.json", {"sha256": sha256(weight),
        "configuration": OmegaConf.to_container(card), "message": model.message.reshape(-1).int().tolist(),
        "serializer": "clamp(127.5 * signed_rgb + 128, 0, 255).to(uint8)",
        "posthoc": True, "baseline": "same sampled latent, no watermark"})
    per_class = min(plan["generation"]["per_class"], job.get("generation_rounds", 50))
    payload = model.message.reshape(-1).int().tolist()
    for batch in generated_batches(per_class, plan["generation"]["batch_size"]):
        relative = f"distseal/generated/records/{batch['key']}.json"
        if cached(root, relative, plan["identity"]):
            continue
        if time.time() >= job["deadline"]:
            return
        set_seed(batch["seed"])
        rng = torch.Generator(device="cuda").manual_seed(batch["seed"])
        labels = torch.tensor(batch["labels"], dtype=torch.int, device="cuda")
        start = time.monotonic()
        rows, images = [], {}
        with torch.no_grad():
            latent = generator.generate(labels, torch.full_like(labels, 1000), 1.0, rng) / .2889
            # Decode one image at a time to bound AE activation memory. The
            # independently sampled latents are unchanged; record this execution policy.
            for i, label in enumerate(batch["labels"]):
                z = latent[i:i+1]
                baseline_signed = decoder.decode(z)
                marked_signed = decoder.decode(model.wm.blender(z, model.wm.embedder(z, model.message)))
                baseline = ((baseline_signed + 1) / 2).clamp(0, 1)
                marked = ((marked_signed + 1) / 2).clamp(0, 1)
                base_png, wm_png = sample_pixels(baseline_signed)[0], sample_pixels(marked_signed)[0]
                name = f"class{label:04d}_sample{batch['repetition']:02d}"
                images[name + "_clean.png"] = png_bytes(base_png)
                images[name + "_wm.png"] = png_bytes(wm_png)
                scores = {"raw_internal": bits_score(model.decode(marked)[0], payload),
                          "author_png": bits_score(model.decode(model.tensor(wm_png))[0], payload),
                          "clean_control": bits_score(model.decode(model.tensor(base_png))[0], payload),
                          "floor_control": bits_score(model.decode(model.tensor(pixels(marked)))[0], payload)}
                rows.append({"name": name, "class_id": label, "payload": payload,
                    "measurements": scores, "quality": {"paired_png": quality(base_png, wm_png),
                    "paired_raw": model.quality(baseline, marked)}})
        archive = root / "distseal/generated/artifacts" / (batch["key"] + ".zip")
        image_zip(archive, images)
        commit(root, relative, plan["identity"], {"method": "distseal", "dataset": "ImageNet-class-conditional-UViT",
            "profile": "author_fixed64", "status": "measured", "batch": batch,
            "seconds": time.monotonic()-start, "samples": rows}, [archive])
        print(f"SAVED DistSeal generation {batch['key']} ({len(rows)} paired images)", flush=True)
