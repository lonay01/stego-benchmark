"""Batched, decoder-blind neural operators. No adaptive attack selection."""

import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image

from .spec import BACKBONES


def rgb8(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite generated pixels")
    return np.rint(np.clip(value, 0, 1)*255).astype(np.uint8)


def grid(image, size=(512, 512)):
    return image.convert("RGB").resize(size, Image.Resampling.BICUBIC) if image.size != size else image.convert("RGB")


class NeuralChannel:
    def __init__(self, engine, cache):
        import torch
        import diffusers
        self.torch, self.engine = torch, engine
        self.device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("Neural generation needs a CUDA GPU; CPU fallback is disabled")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        repo, revision = BACKBONES["sdxl" if engine == "vae" else engine]
        dtype = torch.float32 if engine == "vae" else torch.float16
        common = dict(revision=revision, torch_dtype=dtype, cache_dir=str(cache), use_safetensors=True)
        if engine == "vae":
            self.model = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae", **common).eval()
        elif engine == "sdxl":
            self.model = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(repo, add_watermarker=False, **common)
        elif engine == "inpaint":
            self.model = diffusers.StableDiffusionXLInpaintPipeline.from_pretrained(repo, add_watermarker=False, **common)
        elif engine == "ip2p":
            self.model = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(repo, **common)
            self.model.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
        else:
            raise ValueError(engine)
        self.model.to("cuda")
        # Do not introduce slicing/tiling/offload midway through an experiment.
        if engine != "vae":
            self.model.set_progress_bar_config(disable=True)
        from wmbench.io import finite_json
        components = {"vae": self.model} if engine == "vae" else self.model.components
        self.identity = dict(engine=engine, repo=repo, revision=revision,
            libraries={k: importlib.metadata.version(k) for k in
                       ("torch", "torchvision", "diffusers", "transformers", "huggingface-hub", "numpy", "Pillow")},
            components={k: str(v.dtype) for k, v in components.items() if hasattr(v, "dtype")},
            vae_force_upcast=getattr(getattr(self.model, "vae", self.model).config, "force_upcast", None),
            scheduler=type(self.model.scheduler).__name__ if engine != "vae" else None,
            scheduler_config=finite_json(dict(self.model.scheduler.config)) if engine != "vae" else None,
            safety_checker=type(getattr(self.model, "safety_checker", None)).__name__,
            add_watermarker=False, offload=False, vae_tiling=False, grid=[512, 512],
            torch_gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), tf32=False, generator_device="cpu",
            python=platform.python_version(), driver=subprocess.check_output(
                ["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip())
        self.identity["checkpoint_files"] = audit_checkpoint(cache, repo, revision, engine)

    def apply(self, images, attack, seeds, captions):
        torch, engine = self.torch, self.engine
        if any(im.size != (512, 512) or im.mode != "RGB" for im in images):
            raise ValueError("Neural input must be RGB512; use the declared geometry transform")
        if not len(images) == len(seeds) == len(captions):
            raise ValueError("Batch inputs, seeds and captions must have equal lengths")
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if engine == "vae":
            arrays = np.stack([np.asarray(im) for im in images])
            with torch.inference_mode():
                for _ in range(attack["passes"]):
                    tensor = torch.from_numpy(arrays.copy()).permute(0, 3, 1, 2).to(self.device, torch.float32)/127.5-1
                    latent = self.model.encode(tensor).latent_dist.mode()
                    decoded = self.model.decode(latent).sample
                    arrays = rgb8(((decoded.clamp(-1, 1)+1)/2).permute(0, 2, 3, 1).cpu().numpy())
            outputs, flags, timesteps = list(arrays), [False]*len(images), []
        else:
            prompts = [attack.get("prompt", c) for c in captions]
            if any(not p.strip() for p in prompts):
                raise ValueError("Missing frozen COCO caption")
            kwargs = dict(image=images, prompt=prompts, negative_prompt=[""]*len(images),
                num_inference_steps=20, guidance_scale=attack["guidance_scale"], output_type="np",
                generator=[torch.Generator(device="cpu").manual_seed(s) for s in seeds])
            if "strength" in attack:
                kwargs["strength"] = attack["strength"]
            if engine == "ip2p":
                kwargs["image_guidance_scale"] = attack["image_guidance_scale"]
            else:
                kwargs.update(original_size=(512, 512), target_size=(512, 512), crops_coords_top_left=(0, 0),
                    negative_original_size=(512, 512), negative_target_size=(512, 512), negative_crops_coords_top_left=(0, 0))
            if engine == "inpaint":
                mask = np.zeros((512, 512), dtype=np.uint8)
                mask[128:384, 128:384] = 255
                kwargs.update(mask_image=[Image.fromarray(mask)]*len(images), height=512, width=512)
            timesteps = []
            def callback(_pipeline, _step, timestep, values):
                timesteps.append(float(timestep))
                return values
            with torch.inference_mode():
                result = self.model(**kwargs, callback_on_step_end=callback)
            flags = getattr(result, "nsfw_content_detected", None)
            flags = [False]*len(images) if flags is None else list(flags)
            if len(result.images) != len(images) or len(flags) != len(images):
                raise ValueError("Unexpected pipeline output batch size")
            outputs = [None if blocked else rgb8(a) for a, blocked in zip(result.images, flags)]
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        for output in outputs:
            if output is not None and (output.shape != (512, 512, 3) or np.ptp(output) == 0):
                raise ValueError("Malformed/constant pipeline output: technical failure, not valid BER")
        details = dict(actual_denoising_steps=len(timesteps), timesteps=timesteps,
            batch_size=len(images), batch_seconds=elapsed, amortized_attack_seconds=elapsed/len(images),
            posterior="mode" if engine == "vae" else None, size_conditioning=[512, 512] if engine in {"sdxl", "inpaint"} else None)
        return outputs, flags, details


def audit_checkpoint(cache, repo, revision, engine):
    """Hash once per local file incarnation, not tens of GB on every method.

    The memo lives on disposable local SSD; a new Colab VM rehashes downloads.
    The resulting content hashes, not sizes/mtimes, enter persistent identity.
    """
    from wmbench.io import read_json, write_json, sha256
    folder = Path(cache) / ("models--"+repo.replace("/", "--")) / "snapshots" / revision
    scan = folder / "vae" if engine == "vae" else folder
    if not scan.is_dir():
        raise FileNotFoundError(f"Pinned Hugging Face snapshot missing: {scan}")
    memo_path = Path(cache) / "neural-checksum-memo.json"
    memo = read_json(memo_path) if memo_path.exists() else {}
    result = {}
    for path in sorted(scan.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        key = str(path.resolve())
        prior = memo.get(key)
        checksum = prior["sha256"] if prior and prior["stamp"] == stamp else sha256(path)
        memo[key] = dict(stamp=stamp, sha256=checksum)
        result[path.relative_to(folder).as_posix()] = checksum
    if not result or not any(n.endswith(".safetensors") for n in result):
        raise ValueError("No pinned checkpoint safetensors files were audited")
    write_json(memo_path, memo)
    return result


class PerceptualMetric:
    def __init__(self):
        import torch
        import lpips
        self.torch = torch
        self.model = lpips.LPIPS(net="alex", version="0.1").to("cuda").eval()
        import hashlib
        self.identity = dict(metric="LPIPS AlexNet v0.1", versions={p: importlib.metadata.version(p)
            for p in ("lpips", "torch", "torchvision", "scikit-image")},
            weights={k: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for k, t in self.model.state_dict().items()})

    def compare(self, left, right):
        torch = self.torch
        tensors = [torch.from_numpy(np.stack(a).copy()).permute(0, 3, 1, 2).to("cuda", torch.float32)/127.5-1 for a in (left, right)]
        with torch.inference_mode():
            values = self.model(*tensors).flatten().cpu().tolist()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite LPIPS")
        return values
