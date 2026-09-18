"""Fixed neural channel operators. No watermark decoder or payload is available here."""

from __future__ import annotations


import numpy as np
from PIL import Image

from .io import finite_json
from .generation_errors import GenerationBlocked
from .matching import save_rgb
from .protocol import BACKBONES


class NeuralChannel:
    def __init__(self, options: dict):
        import torch
        import diffusers

        self.torch = torch
        self.engine = options["engine"]
        self.device = options.get("device", "cuda")
        self.dtype = (
            torch.float32
            if self.engine == "vae"
            else getattr(torch, options.get("dtype", "float16"))
        )
        name = "sdxl" if self.engine == "vae" else self.engine
        reference = BACKBONES[name]
        common = {"revision": reference["revision"], "torch_dtype": self.dtype}
        if self.engine == "vae":
            self.model = diffusers.AutoencoderKL.from_pretrained(
                reference["repo_id"], subfolder="vae", **common
            )
        elif self.engine == "sdxl":
            self.model = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
                reference["repo_id"], add_watermarker=False, **common
            )
        elif self.engine == "inpaint":
            self.model = diffusers.StableDiffusionXLInpaintPipeline.from_pretrained(
                reference["repo_id"], add_watermarker=False, **common
            )
        elif self.engine == "ip2p":
            self.model = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
                reference["repo_id"], **common
            )
            self.model.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
                self.model.scheduler.config
            )
        else:
            raise ValueError(self.engine)
        self.model.to(self.device)
        if self.engine != "vae":
            self.model.set_progress_bar_config(disable=True)
        else:
            self.model.eval()

    def apply(
        self, input_path: str, output_path: str, attack: dict, seed: int, caption: str
    ) -> dict:
        torch = self.torch
        image = Image.open(input_path).convert("RGB")
        if image.size != (512, 512):
            raise ValueError("Neural channel input is fixed at 512x512")
        if self.engine == "vae":
            with torch.inference_mode():
                for _ in range(attack["passes"]):
                    value = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0)
                    value = value.to(self.device, torch.float32) / 127.5 - 1
                    # Posterior mode, no scaling-factor multiply: decode(encode(x).mode()).
                    latent = self.model.encode(value).latent_dist.mode()
                    value = self.model.decode(latent).sample[0].clamp(-1, 1)
                    array = torch.round((value + 1) * 127.5).byte().cpu().permute(1, 2, 0).numpy()
                    image = Image.fromarray(array)
            save_rgb(output_path, np.array(image))
            return {"passes": attack["passes"], "posterior": "mode", "dtype": "float32"}
        if not caption.strip() and self.engine in ("sdxl", "inpaint"):
            raise ValueError(
                "COCO caption is missing; refusing a silently different empty-prompt experiment"
            )
        kwargs = {
            "image": image,
            "prompt": attack.get("prompt", caption),
            "negative_prompt": "",
            "num_inference_steps": attack["steps"],
            "guidance_scale": attack["guidance_scale"],
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }
        if "strength" in attack:
            kwargs["strength"] = attack["strength"]
        if self.engine == "ip2p":
            kwargs["image_guidance_scale"] = attack["image_guidance_scale"]
        if self.engine == "inpaint":
            mask = np.zeros((512, 512), dtype=np.uint8)
            left, top, right, bottom = attack["mask_box"]
            mask[top:bottom, left:right] = 255
            kwargs.update({"mask_image": Image.fromarray(mask), "height": 512, "width": 512})
        calls = []

        def callback(_pipeline, _step, timestep, callback_kwargs):
            calls.append(float(timestep))
            return callback_kwargs

        kwargs["callback_on_step_end"] = callback
        with torch.inference_mode():
            result = self.model(**kwargs)
        flags = getattr(result, "nsfw_content_detected", None)
        if flags and any(flags):
            raise GenerationBlocked(
                "generation_blocked: pipeline safety placeholder; not a watermark failure"
            )
        output = np.array(result.images[0].convert("RGB"))
        if output.shape != (512, 512, 3) or np.ptp(output) == 0:
            raise RuntimeError("generation_blocked: empty or malformed output")
        save_rgb(output_path, output)
        return {
            "actual_denoising_steps": len(calls),
            "timesteps": calls,
            "scheduler": type(self.model.scheduler).__name__,
            "scheduler_config": finite_json(dict(self.model.scheduler.config)),
            "dtype": str(self.dtype),
            "editor_watermark": "disabled_or_absent",
        }


def channel_models() -> dict[str, dict]:
    requirements = [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "diffusers==0.32.2",
        "transformers==4.46.3",
        "accelerate==1.2.1",
        "safetensors==0.4.5",
        "huggingface-hub==0.34.4",
        "sentencepiece==0.2.0",
    ]
    return {
        "channel_" + name: {
            "id": "channel_" + name,
            "track": "operator",
            "python": "3.10",
            "validated_lock": f"locks/channel_{name}.lock",
            "adapter": "wmbench.channels:NeuralChannel",
            "assets": [],
            "requirements": requirements,
            "options": {"engine": name, "dtype": "float32" if name == "vae" else "float16"},
        }
        for name in ("vae", "sdxl", "ip2p", "inpaint")
    }
