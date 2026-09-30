"""RoSteALS and SepWater author networks, configurations, ECC and PNG export.

The batch driver replaces only the dataset loop. The five-statement residual
export is executed from the pinned author script, not reimplemented. Separate
unmodified CLI processes cross-check the first five BCH cases.
"""

from __future__ import annotations

import ast
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .environment import download
from .metrics import bits_score, pixels, png_bytes, quality
from .spec import seed, secret_text
from .storage import EvidenceError, cached, commit, freeze, image_zip, read, sha256, write


def set_seed(value):
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def export_function(path):
    tree = ast.parse(Path(path).read_text("utf-8-sig"))
    nodes = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("res", "stego_uint8")), key=lambda n: n.lineno)
    if len(nodes) != 5:
        raise EvidenceError("Author residual export no longer matches the reviewed five statements")
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")

    def export(stego, cover, original):
        import torch
        env = {"torch": torch, "np": np, "stego": stego, "cover": cover,
               "cover_org": original, "w": original.width, "h": original.height}
        exec(code, env)
        return env["stego_uint8"]

    return export


def state_audit(network, state):
    import torch
    checked, failures = [], []
    for key, tensor in network.state_dict().items():
        if key.startswith(("ae.", "control.", "decoder.")):
            checked.append(key)
            if key not in state or tensor.shape != state[key].shape or not torch.equal(tensor.cpu(), state[key].cpu()):
                failures.append(key)
    if not checked or failures:
        raise EvidenceError(f"Unfilled author inference weights: {failures[:10]}")
    return {"checked_tensors": len(checked), "all_equal": True}


def drive_weight(file_id, target):
    receipt = target.with_name(target.name + ".receipt.json")
    if receipt.exists():
        if sha256(target) != read(receipt)["sha256"]:
            raise EvidenceError(f"Changed checkpoint: {target}")
        return target
    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".partial")
    if not gdown.download(id=file_id, output=str(part), quiet=False, resume=True):
        raise RuntimeError(f"Author Drive download failed: {file_id}")
    if part.stat().st_size < 1024**2:
        raise RuntimeError("Checkpoint download is suspiciously small")
    os.replace(part, target)
    freeze(receipt, {"google_drive_id": file_id, "sha256": sha256(target), "bytes": target.stat().st_size})
    return target


class AuthorLatent:
    def __init__(self, name, source, root):
        import torch
        from torchvision import transforms
        from omegaconf import OmegaConf
        self.name, self.source, self.root = name, Path(source), Path(root)
        self.size = 256 if name == "rosteals" else 224
        os.chdir(source)
        sys.path.insert(0, str(source))
        if name == "rosteals":
            from ldm.util import instantiate_from_config
            self.config = source / "models/VQ4_mir_inference.yaml"
            self.checkpoint = download(
                "https://kahlan.cvssp.org/data/Flickr25K/tubui/cvpr23_wmf/epoch=000017-step=000449999.ckpt",
                root / "_assets/rosteals/epoch=000017-step=000449999.ckpt")
            script = source / "inference.py"
        else:
            from Latent_Diffusion_Model.util import instantiate_from_config
            self.config = source / "pre_trained_models/inference.yaml"
            self.checkpoint = drive_weight("17RXXzQ-rMGJ7Xd6e-Az7xeylMj2r0iQS", root / "_assets/sepwater/last.ckpt")
            script = source / "embed.py"
        config = OmegaConf.load(self.config).model
        if int(config.params.control_config.params.secret_len) != 100:
            raise EvidenceError("Wrong payload length")
        config.params.decoder_config.params.secret_len = 100  # exact author main()
        # Keep the full author config, including training-only loss and noise.
        self.model = instantiate_from_config(config)
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        audit = state_audit(self.model, state)
        audit.update(missing=list(missing), unexpected=list(unexpected), source_config_sha256=sha256(self.config),
                     checkpoint_sha256=sha256(self.checkpoint), full_config=OmegaConf.to_container(config, resolve=True),
                     checkpoint_metadata={k: checkpoint[k] for k in ("epoch", "global_step") if k in checkpoint})
        freeze(root / "_meta" / name / "weight_audit.json", audit)
        del state, checkpoint
        self.model = self.model.cuda().eval().requires_grad_(False)
        self.transform = transforms.Compose([transforms.Resize((self.size, self.size)), transforms.ToTensor(),
                                             transforms.Normalize([.5]*3, [.5]*3)])
        self.export = export_function(script)
        from tools.ecc import ECC
        self.ecc = ECC()
        clean_text = "secrets"
        packet = self.ecc.encode_text([clean_text])
        ecc_trials = []
        for nerrors in (0, 1, 3, 5, 6):
            test = packet.copy()
            test[0, :nerrors] = 1 - test[0, :nerrors]
            set_seed(seed(name, "ecc-selftest", nerrors))
            try:
                decoded = self.ecc.decode_text(test)[0]
                ecc_trials.append({"flipped_data_bits": nerrors, "text": decoded, "exact": decoded == clean_text})
            except Exception as error:
                ecc_trials.append({"flipped_data_bits": nerrors, "error": repr(error)})
        freeze(root / "_meta" / name / "author_ecc_selftest.json", {
            "trials": ecc_trials, "note": "Unmodified released ECC helper; a failed error-correction test is reported, not patched"})
        import lpips
        self.lpips = lpips.LPIPS(net="alex").cuda().eval()
        self.sifid = None
        if name == "rosteals":
            from tools.sifid import SIFID
            self.sifid = SIFID()

    def decode(self, signed):
        """No expected message, embedding metadata or original image is accepted."""
        import torch
        with torch.no_grad():
            logits = self.model.decoder(signed).detach().float().cpu().numpy().reshape(-1)
        if logits.shape != (100,) or not np.isfinite(logits).all():
            raise EvidenceError("Invalid author decoder result")
        return logits

    def tensor(self, image):
        return self.transform(image).unsqueeze(0).cuda()

    def measure(self, image, payload):
        import torch
        cover = self.tensor(image)
        message = torch.tensor([payload], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            latent = self.model.encode_first_stage(cover)
            ae = self.model.decode_first_stage(latent)
            marked_latent, _ = self.model(latent, None, message)
            stego = self.model.decode_first_stage(marked_latent)
            exported = self.export(stego, cover, image)
            internal = pixels((stego + 1) / 2, "floor")
            internal_input = pixels((cover + 1) / 2, "floor")
            internal_ae = pixels((ae + 1) / 2, "floor")
            final_png = png_bytes(exported)
            reloaded = Image.open(io.BytesIO(final_png)).convert("RGB")
            measures = {"raw_internal": bits_score(self.decode(stego), payload),
                        "clipped_internal": bits_score(self.decode(stego.clamp(-1, 1)), payload),
                        "author_png": bits_score(self.decode(self.tensor(reloaded)), payload),
                        "clean_control": bits_score(self.decode(cover), payload),
                        "ae_control": bits_score(self.decode(ae), payload)}
            for rule in ("floor", "round_even", "half_up"):
                converted = pixels((stego + 1) / 2, rule)
                measures[f"internal_{rule}_png"] = bits_score(self.decode(self.tensor(Image.fromarray(converted))), payload)
            # A float memory/serialization roundtrip is exact and is not an image attack.
            buffer = io.BytesIO()
            np.save(buffer, stego.detach().cpu().numpy(), allow_pickle=False)
            buffer.seek(0)
            restored = torch.from_numpy(np.load(buffer, allow_pickle=False)).cuda()
            checks = {"float_roundtrip_bits_equal": np.array_equal(self.decode(stego), self.decode(restored)),
                      "png_pixels_equal": np.array_equal(exported, np.asarray(reloaded))}
            q = {"source_to_author_png": quality(np.asarray(image), exported),
                 "internal_source_to_marked": quality(internal_input, internal),
                 "internal_source_to_ae": quality(internal_input, internal_ae),
                 "internal_ae_to_marked": quality(internal_ae, internal)}
            # Same latent/marked tensors as the evaluation program, not a re-encode.
            q["internal_source_to_marked"]["lpips_raw"] = float(self.lpips(cover, stego).item())
            q["internal_source_to_ae"]["lpips_raw"] = float(self.lpips(cover, ae).item())
            q["internal_ae_to_marked"]["lpips_raw"] = float(self.lpips(ae, stego).item())
            original_tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].float().cuda()/127.5-1
            exported_tensor = torch.from_numpy(exported.copy()).permute(2, 0, 1)[None].float().cuda()/127.5-1
            q["source_to_author_png"]["lpips_raw"] = float(self.lpips(original_tensor, exported_tensor).item())
            if self.sifid is not None:
                for key, a, b in (("internal_source_to_marked", cover, stego),
                                  ("internal_source_to_ae", cover, ae), ("internal_ae_to_marked", ae, stego)):
                    q[key]["sifid_raw"] = float(self.sifid(a[0], b[0]))
                q["source_to_author_png"]["sifid_raw"] = float(self.sifid(original_tensor[0], exported_tensor[0]))
        return {"measurements": measures, "quality": q, "checks": checks,
                "internal_size": self.size, "export_size": list(image.size)}, {
                    "author.png": final_png, "internal.png": png_bytes(internal), "ae.png": png_bytes(internal_ae)}


def cli_reference(reference, input_file, text, destination, env=None):
    """Execute unmodified author CLI in a NEW process, not the batch network."""
    source, size, name = reference.source, reference.size, reference.name
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "author.png"
    args = [sys.executable, "-m", "authornight.author_cli", str(source / ("inference.py" if name == "rosteals" else "embed.py")),
            "--config", str(reference.config), "--weight", str(reference.checkpoint),
            "--image_size", str(size), "--secret", text]
    if name == "sepwater":
        covers = destination / "covers"
        covers.mkdir(exist_ok=True)
        # PNG input contains exactly the JPEG-decoded RGB pixels used by the batch path.
        Image.open(input_file).convert("RGB").save(covers / "author.png")
        args += ["--cover", str(covers), "--output", str(destination)]
    else:
        args += ["--cover", str(input_file), "--output", str(output)]
    logfile = destination / "stdout.log"
    with logfile.open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    if name == "sepwater":
        extract = [sys.executable, "-m", "authornight.author_cli", str(source / "extract.py"), "--config", str(reference.config),
                   "--weight", str(reference.checkpoint), "--image_size", str(size),
                   "--secret", text, "--cover", str(destination)]
        with logfile.open("a", encoding="utf-8") as log:
            subprocess.run(extract, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    return output, logfile


def run_coco(job):
    import torch
    root, source = Path(job["output"]), Path(job["source"])
    name, deadline = job["method"], job["deadline"]
    plan = read(root / "plan.json")
    identity = plan["identity"]
    reference = AuthorLatent(name, source, root)
    originals = read(root / "_inputs/originals.json")
    for image_index, item in enumerate(originals):
        input_file = root / item["path"]
        if sha256(input_file) != item["sha256"]:
            raise EvidenceError("COCO original changed")
        image = Image.open(input_file).convert("RGB")
        for repetition in range(5):
            for profile in ("raw100", "author_bch56"):
                key = f"{item['image_id']:012d}_m{repetition}_{profile}"
                relative = f"{name}/coco/records/{key}.json"
                if cached(root, relative, identity):
                    continue
                if time.time() >= deadline:
                    return
                case_seed = seed(name, item["image_id"], repetition, profile)
                set_seed(case_seed)
                text = secret_text(item["image_id"], repetition)
                payload = (np.random.default_rng(seed("raw100", item["image_id"], repetition)).integers(0, 2, 100)
                           if profile == "raw100" else reference.ecc.encode_text([text])[0]).astype(int).tolist()
                started = time.monotonic()
                data, images = reference.measure(image, payload)
                if repetition == 0:
                    set_seed(case_seed)
                    again, extra = reference.measure(image, payload)
                    data["checks"]["repeat_pixels_equal"] = extra["author.png"] == images["author.png"]
                    data["checks"]["repeat_bits_equal"] = again["measurements"]["author_png"]["bits"] == data["measurements"]["author_png"]["bits"]
                if profile == "author_bch56":
                    for stage in ("raw_internal", "author_png"):
                        prediction = np.array([data["measurements"][stage]["bits"]])
                        set_seed(case_seed)  # author's helper can return random output on ECC failure
                        decoded = reference.ecc.decode_text(prediction)[0]
                        data["measurements"][stage]["author_ecc_text"] = decoded
                        data["measurements"][stage]["author_ecc_text_exact"] = decoded == text
                        data["measurements"][stage]["text_prefix_ber"] = float(np.mean(prediction[0, :56] != np.array(payload[:56])))
                    if image_index < 5 and repetition == 0:
                        destination = Path(job["scratch"]) / "cli" / name / key
                        try:
                            output, logfile = cli_reference(reference, input_file, text, destination)
                            data["checks"]["unmodified_cli_pixels_equal"] = np.array_equal(
                                np.asarray(Image.open(output).convert("RGB")), np.asarray(Image.open(io.BytesIO(images["author.png"]))))
                            images["cli.log"] = logfile.read_bytes()
                            images["cli.png"] = output.read_bytes()
                            import re
                            cli_text = logfile.read_text("utf-8", errors="replace")
                            if name == "sepwater":
                                match = re.search(r"\[author\.png\] Bit Acc: ([\d.]+)%", cli_text)
                                expected = 100 * (1-data["measurements"]["author_png"]["text_prefix_ber"])
                                data["checks"]["unmodified_extract_cli_accuracy_equal"] = bool(match and abs(float(match[1])-expected) <= .0051)
                            else:
                                match = re.search(r"Bit acc: ([\d.]+)", cli_text)
                                expected = 1-data["measurements"]["raw_internal"]["ber"]
                                data["checks"]["unmodified_cli_accuracy_equal"] = bool(match and abs(float(match[1])-expected) < 1e-7)
                        except Exception as error:
                            data["cli_error"] = f"{type(error).__name__}: {error}"
                            data["checks"]["unmodified_cli_pixels_equal"] = False
                            logfile = destination / "stdout.log"
                            if logfile.exists():
                                images["cli.log"] = logfile.read_bytes()
                archive = root / name / "coco/artifacts" / f"{key}.zip"
                image_zip(archive, images)
                commit(root, relative, identity, {"method": name, "dataset": "COCO-val2017-original",
                    "profile": profile, "image_id": item["image_id"], "repetition": repetition,
                    "seed": case_seed, "payload": payload, "text": text if profile == "author_bch56" else None,
                    "seconds": time.monotonic() - started, "status": "measured", **data}, [archive])
                print(f"SAVED {name}: image {image_index+1}/{len(originals)}, message {repetition+1}/5, {profile}; "
                      f"PNG BER {100*data['measurements']['author_png']['ber']:.3f}%", flush=True)
                torch.cuda.empty_cache()
    if name == "sepwater" and time.time() < deadline:
        # Inspect the other published asset rather than assuming it is a 256px watermarker.
        audit_path = root / "_meta/sepwater/f4_asset_audit.json"
        if not audit_path.exists():
            try:
                weight = drive_weight("1-aiZbYyETi8zQZ_ZGB8PaxHoFrV4elns", root / "_assets/sepwater/f4.ckpt")
                state = torch.load(weight, map_location="cpu")
                state = state.get("state_dict", state)
                groups = {prefix: sum(k.startswith(prefix) for k in state) for prefix in ("ae.", "control.", "decoder.", "encoder.")}
                write(audit_path, {"sha256": sha256(weight), "tensor_groups": groups,
                    "control_shapes": {k: list(v.shape) for k, v in state.items() if k.startswith("control.") and hasattr(v, "shape")},
                    "selected_for_inference": False,
                    "reason": "No published matching 256px watermark configuration; do not invent or resize learned weights"})
            except Exception as error:
                write(audit_path, {"status": "asset_inspection_failed", "reason": str(error)})


def run_corruptions(job):
    """RoSteALS released test pool; expanded five severities, clearly labelled."""
    import torch
    import zipfile
    root, source = Path(job["output"]), Path(job["source"])
    reference = AuthorLatent(job["method"], source, root)
    name = job["method"]
    if name == "rosteals":
        from tools.augment_imagenetc import RandomImagenetC
        noise = RandomImagenetC(1, 5, "test")
    else:
        from tools.noise_select import random_select
        noise = random_select(1, 5, "val")
    plan = read(root / "plan.json")
    transform = __import__("torchvision").transforms.Compose([
        __import__("torchvision").transforms.ToTensor(),
        __import__("torchvision").transforms.Normalize([.5]*3, [.5]*3)])
    for path in sorted((root / name / "coco/records").glob("*_raw100.json")):
        key = path.stem
        relative = f"{name}/corruptions/{key}.json"
        if cached(root, relative, plan["identity"]):
            continue
        if time.time() >= job["deadline"]:
            return
        row = cached(root, path.relative_to(root).as_posix(), plan["identity"])
        archive_path = root / next(iter(row["artifacts"]))
        with zipfile.ZipFile(archive_path) as archive:
            image = Image.open(io.BytesIO(archive.read("internal.png"))).convert("RGB")
        scores = []
        for noise_id in noise.corrupt_ids:
            for severity in range(1, 6):
                if time.time() >= job["deadline"]:
                    return
                set_seed(seed(name, "corruption", key, int(noise_id), severity))
                try:
                    perturbed = noise(image.copy(), int(noise_id), severity)
                    with torch.no_grad():
                        score = bits_score(reference.decode(transform(perturbed)[None].cuda()), row["payload"])
                    score.pop("logits")
                    score.pop("bits")
                    scores.append({"attack": noise.method_names[int(noise_id)], "severity": severity, "status": "measured", **score})
                except Exception as error:
                    scores.append({"attack": noise.method_names[int(noise_id)], "severity": severity,
                                   "status": "error", "error": f"{type(error).__name__}: {error}"})
        commit(root, relative, plan["identity"], {"method": name,
            "status": "measured" if all(s['status'] == 'measured' for s in scores) else "has_errors", "profile": "all_five_severities",
            "source_record": path.relative_to(root).as_posix(), "scores": scores,
            "note": "Expanded fixed grid, not the released evaluation script's one randomly selected severity"})
        print(f"SAVED {name} corruption grid: {key}, {len(scores)} tests", flush=True)
