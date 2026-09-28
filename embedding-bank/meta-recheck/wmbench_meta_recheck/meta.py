"""Meta's reviewed inference classes, two fixed configurations, no score tuning."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import urllib.request

import numpy as np
from PIL import Image

from wmbench.io import atomic_bytes, digest, read_json, sha256, write_json
from wmbench.matching import load_rgb, psnr
from wmbench.worker import runtime_info, seed_everything
from wmbench_full_payload.contracts import message, score
from wmbench_full_payload.storage import EvidenceError, cached, freeze, inside, retry_record, sealed
from wmbench_native_canary.contracts import center_crop
from wmbench_author_reference.upstream import tensor_digest

MODEL = {"id": "meta_hidden_revisited", "native_bits": 48}
PROFILES = {
    "checkpoint_256": {"size": 256, "jnd": False, "scaling_w": 0.3},
    "demo_512": {"size": 512, "jnd": True, "scaling_w": 1.5},
}
POLICY = {"schema": "meta-hidden-recheck-v1", "images": 5, "messages": 5,
          "profiles": PROFILES, "scaling_i": 1.0, "scale_channels": False,
          "payload": "all_48_bits_same_shake256_stream_as_native_canary",
          "input": "frozen_rgb512_center_crop_no_resampling",
          "serializer": "Meta_demo_clip_round_then_ToPILImage",
          "attacks": [], "matched_psnr": False, "training": False}


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_asset(path, relative, contract):
    data = path.read_bytes()
    expected = contract["files"][relative]
    if len(data) != expected["size"] or blob(data) != expected["git_blob"]:
        raise EvidenceError(f"Meta source/checkpoint differs: {relative}")
    if relative.endswith(".pth") and sha256(path) != contract["checkpoint_sha256"]:
        raise EvidenceError("Meta checkpoint SHA256 differs")


def download_assets(project, root):
    """Only a 5.5 MiB checkpoint and reviewed source files, not Stable Diffusion."""
    contract = read_json(project / "meta_contract.json")
    folder = root / "_assets/meta"
    for relative in contract["files"]:
        path = inside(folder, relative)
        if not path.exists():
            url = f"https://raw.githubusercontent.com/{contract['repo']}/{contract['revision']}/{relative}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        data = response.read(contract["files"][relative]["size"] + 1)
                    if blob(data) != contract["files"][relative]["git_blob"]:
                        raise EvidenceError(f"Unexpected download bytes: {relative}")
                    atomic_bytes(path, data)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        verify_asset(path, relative, contract)
    return folder


def case_folder(root, case):
    folder = root / case["profile"] / case["stem"]
    if sha256(folder / "source.png") != case["source_sha256"]:
        raise EvidenceError("Meta source crop changed")
    return folder


def row(root, case, identity, verify=True):
    folder = case_folder(root, case) if verify else root / case["profile"] / case["stem"]
    path = folder / "measurement.json"
    key = digest({"trial": identity, "case": case})
    if verify:
        return cached(root, path, key)
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("signature") != key:
        raise EvidenceError("Meta result signature changed")
    return value


def complete(root, case, identity):
    value = row(root, case, identity)
    return bool(value and value["status"] == "measured")


def context(project, bank, root):
    manifest = read_json(bank / "_inputs/manifest.json")
    images = manifest["images"][:5]
    if len(images) != 5 or len({i["image_id"] for i in images}) != 5:
        raise EvidenceError("Need the same first five distinct images")
    identity = {"policy": POLICY, "inputs": images, "contract": read_json(project / "meta_contract.json"),
                "code": digest({p.relative_to(project).as_posix(): sha256(p)
                                for p in sorted((project / "src").rglob("*.py"))})}
    freeze(root / "trial.json", identity)
    cases = []
    for image in images:
        source = inside(bank, image["path"])
        if sha256(source) != image["sha256"]:
            raise EvidenceError("Frozen RGB512 source changed")
        for profile, spec in PROFILES.items():
            with Image.open(source) as im:
                crop = center_crop(im, spec["size"])
            for rep in range(1, 6):
                stem = f"{int(image['image_id']):012d}_m{rep:02d}"
                folder = root / profile / stem
                output = folder / "source.png"
                if output.exists():
                    if not np.array_equal(load_rgb(output), np.asarray(crop)):
                        raise EvidenceError("Saved Meta crop differs")
                else:
                    import io
                    stream = io.BytesIO()
                    crop.save(stream, format="PNG")
                    atomic_bytes(output, stream.getvalue())
                assignment = message(MODEL, image["image_id"], rep)
                assignment["capacity"]["bits_per_pixel"] = 48 / spec["size"]**2
                assignment["capacity"]["basis"] = "Meta hidden_replicate.pth: 48 information bits; strict encoder/decoder state loading"
                case = {"profile": profile, "stem": stem, "image_id": image["image_id"],
                        "replicate": rep, "assignment": assignment, "source_sha256": sha256(output)}
                freeze(folder / "case.json", case)
                cases.append(case)
    plan = {"identity": digest(identity), "cases": cases, "policy": POLICY}
    freeze(root / "plan.json", plan)
    return plan


def prepare(project, root, slot, device):
    assets = download_assets(project, root)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Notebook bootstrap must install uv first")
    venv = slot / "environments/meta"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    locks = root / "_meta/environment"
    locks.mkdir(parents=True, exist_ok=True)
    requirements = ["torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "Pillow==10.4.0"]
    request = {"python": "3.10.12", "requirements": requirements}
    freeze(locks / "request.json", request)
    if not python.exists():
        subprocess.run([uv, "venv", "--python", request["python"], str(venv)], check=True)
    source, lock = locks / "requirements.in", locks / "requirements.lock"
    atomic_bytes(source, ("\n".join(requirements) + "\n").encode())
    if not lock.exists():
        temporary = slot / "requirements.lock.tmp"
        subprocess.run([uv, "pip", "compile", "--python", str(python), str(source), "-o", str(temporary)], check=True)
        atomic_bytes(lock, temporary.read_bytes())
    subprocess.run([uv, "pip", "sync", "--python", str(python), str(lock)], check=True)
    installed = json.loads(subprocess.check_output([uv, "pip", "list", "--python", str(python), "--format=json"], text=True))
    freeze(locks / "installed.json", sorted(installed, key=lambda x: x["name"]))
    write_json(root / "_meta/prepared.json", {"status": "ready", "python": str(python),
               "assets": str(assets), "lock_sha256": sha256(lock), "device": device})


def load_classes(folder):
    """Execute four unchanged Meta classes; omit unused ViT/DVMark training imports."""
    import torch
    spec = importlib.util.spec_from_file_location("meta_reviewed_attenuations", folder / "attenuations.py")
    attenuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attenuation)
    wanted = {"ConvBNRelu", "HiddenEncoder", "HiddenDecoder", "EncoderWithJND"}
    tree = ast.parse((folder / "models.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted or len(nodes) != 4:
        raise EvidenceError("Unexpected Meta model definitions")
    namespace = {"__name__": "meta_reviewed_classes", "torch": torch, "nn": torch.nn, "attenuations": attenuation}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(folder / "models.py"), "exec"), namespace)
    return namespace, attenuation


class MetaHidden:
    def __init__(self, assets, device):
        import torch
        from torchvision import transforms
        self.torch, self.device = torch, device
        classes, attenuation = load_classes(assets / "hidden")
        # This checkpoint contains argparse.Namespace metadata. No unsafe fallback.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(assets / "hidden/ckpts/hidden_replicate.pth", map_location="cpu", weights_only=True)
        self.training_config = vars(checkpoint["params"])
        expected = {"num_bits": 48, "redundancy": 1, "img_size": 256, "encoder": "hidden",
                    "encoder_depth": 4, "encoder_channels": 64, "use_tanh": True, "decoder_depth": 8,
                    "decoder_channels": 64, "scaling_i": 1.0, "scaling_w": 0.3,
                    "scale_channels": False, "attenuation": None}
        if any(self.training_config.get(k) != v for k, v in expected.items()):
            raise EvidenceError("Unexpected Meta checkpoint inference configuration")
        self.encoder = classes["HiddenEncoder"](num_blocks=4, num_bits=48, channels=64)
        self.decoder = classes["HiddenDecoder"](num_blocks=8, num_bits=48, channels=64)
        state = {k.removeprefix("module."): v for k, v in checkpoint["encoder_decoder"].items()}
        for prefix, model in (("encoder.", self.encoder), ("decoder.", self.decoder)):
            model.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
            model.to(device).eval().requires_grad_(False)
        self.normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        self.unnormalize = transforms.Normalize([-.485/.229, -.456/.224, -.406/.225], [1/.229, 1/.224, 1/.225])
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.encoders = {}
        for profile, config in PROFILES.items():
            jnd = attenuation.JND(preprocess=self.unnormalize) if config["jnd"] else None
            self.encoders[profile] = classes["EncoderWithJND"](self.encoder, jnd, False, 1.0, config["scaling_w"]).to(device).eval()
        self.weights = {"encoder": tensor_digest(self.encoder), "decoder": tensor_digest(self.decoder)}

    def decode(self, normalized):
        with self.torch.inference_mode():
            logits = self.decoder(normalized).detach().float().cpu().numpy().reshape(-1)
        if logits.shape != (48,) or not np.isfinite(logits).all():
            raise EvidenceError("Meta decoder returned invalid logits")
        return {"bits": (logits > 0).astype(int).tolist(), "logits": logits.tolist()}

    def measure(self, source, payload, profile, attempt):
        torch = self.torch
        if source.size != (PROFILES[profile]["size"],) * 2 or source.mode != "RGB" or len(payload) != 48:
            raise EvidenceError("Invalid Meta carrier or full payload")
        x = self.normalize(self.to_tensor(source)).unsqueeze(0).to(self.device)
        bipolar = 2 * torch.tensor([payload], dtype=torch.float32, device=self.device) - 1
        with torch.inference_mode():
            marked = self.encoders[profile](x, bipolar)
            rgb = self.unnormalize(marked)
            clipped = rgb.clamp(0, 1)
            rounded = torch.round(255 * clipped) / 255
            png = self.to_pil(rounded.squeeze(0).cpu())
        raw = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
        if not np.isfinite(raw).all():
            raise EvidenceError("Meta produced non-finite pixels")
        np.save(attempt / "raw_rgb.npy", raw, allow_pickle=False)
        png.save(attempt / "watermarked.png", format="PNG")
        with Image.open(attempt / "watermarked.png") as reloaded:
            if not np.array_equal(np.asarray(reloaded), np.asarray(png)):
                raise EvidenceError("PNG round-trip changed pixel values")
            encoded = self.normalize(self.to_tensor(reloaded)).unsqueeze(0).to(self.device)
        stages = {"raw": marked, "clipped": self.normalize(clipped), "round8_memory": self.normalize(rounded),
                  "png": encoded, "clean_control": x}
        decoded = {name: self.decode(value) for name, value in stages.items()}
        return {"decoded": decoded, "psnr_original_to_png": psnr(np.asarray(source), np.asarray(png)),
                "png_file_matches_pil": True,
                "round8_vs_png_same_bits": decoded["round8_memory"]["bits"] == decoded["png"]["bits"],
                "clipped_fraction": float(np.mean((raw < 0) | (raw > 1)))}


def work(project, root, plan):
    prepared = read_json(root / "_meta/prepared.json")
    contract = read_json(project / "meta_contract.json")
    assets = Path(prepared["assets"])
    for relative in contract["files"]:
        verify_asset(assets / relative, relative, contract)
    execution = runtime_info()
    freeze(root / "_meta/runtime.json", execution)
    if prepared["device"] == "cuda" and not execution["cuda_available"]:
        raise RuntimeError("CUDA requested but not available; refusing a silent CPU fallback")
    seed_everything(0)
    model = MetaHidden(assets, prepared["device"])
    freeze(root / "_meta/weights.json", model.weights)
    freeze(root / "_meta/checkpoint_training_config.json", model.training_config)
    errors = 0
    for index, case in enumerate(plan["cases"], 1):
        if complete(root, case, plan["identity"]):
            continue
        folder = case_folder(root, case)
        path = folder / "measurement.json"
        retry_record(path)
        attempt = folder / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        start = time.monotonic()
        print(f"Meta {index}/50 {case['profile']} {case['stem']}", flush=True)
        try:
            with Image.open(folder / "source.png") as source:
                result = model.measure(source, case["assignment"]["payload"], case["profile"], attempt)
            decoded = result.pop("decoded")
            data = {**result, "status": "measured", "stages": {
                stage: {**score(value, case["assignment"]), "logits": value["logits"]} for stage, value in decoded.items()}}
        except EvidenceError:
            raise
        except Exception as error:
            errors += 1
            data = {"status": "error", "reason": str(error), "traceback": traceback.format_exc()}
        sealed(root, path, digest({"trial": plan["identity"], "case": case}),
               {**data, "execution": execution, "seconds": time.monotonic() - start}, list(attempt.iterdir()))
        print(f"  {data['status']}; PNG BER={data.get('stages', {}).get('png', {}).get('ber', '-')}", flush=True)
    if errors:
        raise RuntimeError(f"{errors} Meta technical errors; measured results are saved")
