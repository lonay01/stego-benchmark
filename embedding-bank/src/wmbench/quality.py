"""Perceptual metrics are recorded, not used to optimize message survival."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io import digest, read_json, sha256, write_json
from .matching import load_rgb, psnr
from .protocol import BACKBONES


class QualityMetrics:
    def __init__(self, options: dict):
        import torch
        import lpips
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = options.get("device", "cuda")
        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        dino, clip = BACKBONES["dino"], BACKBONES["clip"]
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino["repo_id"], revision=dino["revision"]
        )
        self.dino = (
            AutoModel.from_pretrained(dino["repo_id"], revision=dino["revision"])
            .to(self.device)
            .eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip["repo_id"], revision=clip["revision"]
        )
        self.clip = (
            CLIPModel.from_pretrained(clip["repo_id"], revision=clip["revision"])
            .to(self.device)
            .eval()
        )
        self.cache = Path(options["asset_root"]) / "quality_features"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _features(self, path: str, prompt: str) -> dict:
        key = digest({"image": sha256(path), "prompt": prompt, "backbones": BACKBONES})
        cached = self.cache / (key + ".json")
        if cached.exists():
            return read_json(cached)
        torch = self.torch
        image = Image.open(path).convert("RGB")
        with torch.inference_mode():
            inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
            dino = self.dino(**inputs).last_hidden_state[:, 0]
            dino = torch.nn.functional.normalize(dino, dim=-1)[0]
            inputs = self.clip_processor(
                images=image,
                text=[prompt or "an image"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            output = self.clip(**inputs)
            similarity = torch.nn.functional.cosine_similarity(
                output.image_embeds, output.text_embeds
            ).item()
        features = {"dino": dino.cpu().tolist(), "clip_target_similarity": similarity}
        write_json(cached, features)
        return features

    def compare(
        self, first: str, second: str, prompt: str = "", mask_box: list | None = None
    ) -> dict:
        from skimage.metrics import structural_similarity

        torch = self.torch
        a, b = load_rgb(first), load_rgb(second)
        if a.shape != b.shape:
            raise ValueError("Quality metrics require a common image grid")

        def tensor(array):
            return (
                torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
                / 127.5
                - 1
            )

        with torch.inference_mode():
            perceptual = self.lpips(tensor(a), tensor(b)).item()
        fa, fb = self._features(first, prompt), self._features(second, prompt)
        result = {
            "psnr": psnr(a, b),
            "ssim": structural_similarity(a, b, data_range=255, channel_axis=2),
            "lpips": perceptual,
            "dino_cosine": float(np.dot(fa["dino"], fb["dino"])),
            "clip_target_before": fa["clip_target_similarity"],
            "clip_target_after": fb["clip_target_similarity"],
            "clip_target_delta": fb["clip_target_similarity"] - fa["clip_target_similarity"],
        }
        if mask_box:
            mask = np.zeros(a.shape[:2], dtype=bool)
            left, top, right, bottom = mask_box
            mask[top:bottom, left:right] = True
            error = np.mean(np.abs(a.astype(float) - b.astype(float)), axis=2) / 255
            result.update(
                {
                    "mean_abs_change_inside": float(error[mask].mean()),
                    "mean_abs_change_outside": float(error[~mask].mean()),
                }
            )
        return result


def quality_model() -> dict:
    return {
        "id": "quality_metrics",
        "track": "operator",
        "python": "3.10",
        "validated_lock": "locks/quality_metrics.lock",
        "adapter": "wmbench.quality:QualityMetrics",
        "assets": [],
        "requirements": [
            "torch==2.5.1",
            "torchvision==0.20.1",
            "lpips==0.1.4",
            "scikit-image==0.24.0",
            "transformers==4.46.3",
            "huggingface-hub==0.34.4",
            "scipy==1.14.1",
        ],
    }
