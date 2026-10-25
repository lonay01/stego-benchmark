"""Preserved historical batches, fixed attacks, durable Native-only examples."""

from collections import defaultdict
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .common import (ATTACK_IDS, CONTROLS, commit, control_pixels, inside,
                     immutable_bytes, pixel_stats, png_bytes, read, rgb, sha,
                     verify_artifacts)


def quality(left, right):
    from skimage.metrics import structural_similarity
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    return dict(psnr=10 * np.log10(255**2 / mse) if mse else "Infinity",
                ssim=float(structural_similarity(left, right, data_range=255, channel_axis=2,
                           gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))


def paths(root, method, case, condition):
    base = Path(root) / method
    return (base / "images" / condition / (case["id"] + ".png"),
            base / "generated" / condition / (case["id"] + ".json"))


def save_png(root, case, condition, pixels, extra=None):
    target, receipt = paths(root, case["method"], case, condition)
    relative = target.relative_to(root).as_posix()
    checksum = immutable_bytes(target, png_bytes(pixels))
    value = dict(case_key=case["key"], condition=condition, status="generated",
                 path=relative, pixels=pixel_stats(pixels), artifacts={relative: checksum},
                 png_sha256=checksum, **(extra or {}))
    commit(receipt, value)
    return value


def controls(job):
    root, bank = Path(job["results"]), Path(job["bank"])
    for case in read(root / job["model"]["id"] / "native-cases.json"):
        origin = inside(bank, case["output"])
        if sha(origin) != case["carrier_sha256"]:
            raise RuntimeError("Original carrier checksum changed")
        carrier = rgb(origin)
        source_path = inside(bank, case["source_crop"])
        source = rgb(source_path)
        for condition in CONTROLS:
            _, receipt = paths(root, case["method"], case, condition)
            if receipt.exists():
                verify_artifacts(root, read(receipt))
                continue
            result = control_pixels(condition, carrier, source)
            if condition in CONTROLS[:4] and not np.array_equal(result, carrier):
                raise RuntimeError(f"Identity control changed pixels: {condition}")
            save_png(root, case, condition, result,
                     dict(quality=quality(carrier, result), input_pixels=pixel_stats(carrier),
                          source_crop_sha256=sha(source_path)))
        print(f"CONTROLS {case['method']} {case['id']}", flush=True)


def vae_float(channel, images):
    """Independent explicit reference of the declared VAE normalization boundary."""
    torch = channel.torch
    source = np.stack([np.asarray(im) for im in images])
    tensor = torch.from_numpy(source.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32)
    tensor = tensor / 127.5 - 1
    with torch.inference_mode():
        latent = channel.model.encode(tensor).latent_dist.mode()
        output = channel.model.decode(latent).sample
    return ((output.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()


def generated_complete(root, case, attack):
    _, receipt = paths(root, case["method"], case, attack)
    if not receipt.exists():
        return False
    value = read(receipt)
    if value["case_key"] != case["key"]:
        raise RuntimeError("Cached generation belongs to a different case")
    verify_artifacts(root, value)
    if value.get("float_path") and sha(inside(root, value["float_path"])) != value["float_sha256"]:
        raise RuntimeError("Cached float artifact is corrupt")
    return True


def generate(job):
    from wmbench_neural.channel import NeuralChannel, rgb8
    from wmbench_neural.spec import BY_ID, batches, seed
    from neural_sdxl_identity_fix import canonical_identity, changed_paths

    root, method = Path(job["results"]), job["model"]["id"]
    all_cases = read(root / method / "attack-cases.json")  # Deliberately no expected messages.
    groups = defaultdict(list)
    for case in all_cases:
        groups[case["image_id"]].append(case)
    attacks = [BY_ID[a] for a in ATTACK_IDS if BY_ID[a]["engine"] == job["engine"]]
    todo = [(a, group) for a in attacks for group in groups.values()
            if not all(generated_complete(root, c, a["id"]) for c in group if c["mode"] == "native")]
    if not todo:
        print(f"REUSE generation {method} {job['engine']}", flush=True)
        return
    channel = NeuralChannel(job["engine"], Path(job["scratch"]) / "hf")
    old = read(Path(job["neural"]) / "_execution" / (job["engine"] + ".json"))
    actual, expected = canonical_identity(channel.identity), canonical_identity(old)
    if actual != expected:
        raise RuntimeError("Historical generator execution differs: " +
                           ", ".join(changed_paths(expected, actual)[:16]))
    commit(root / "_execution" / (job["engine"] + ".json"), actual)
    captions = read(root / "_inputs/captions.json")["captions"]
    batch_width = job["settings"]["gpu_batch_size"]
    for attack, group in todo:
        if time.time() >= job["deadline"]:
            raise TimeoutError("Session budget reached; completed diagnostic rows preserved")
        for indexes, count in batches(list(range(len(group))), batch_width):
            if not any(group[i]["mode"] == "native" for i in indexes[:count]):
                continue  # No stateful RNG: each actual item has its own frozen seed.
            batch_cases = [group[i] for i in indexes]
            images = []
            for case in batch_cases:
                path = inside(Path(job["bank"]), case["output"])
                if sha(path) != case["carrier_sha256"]:
                    raise RuntimeError("Input carrier changed during replay")
                images.append(Image.fromarray(rgb(path)))
            outputs, flags, details = channel.apply(images, attack,
                [seed(c, attack) for c in batch_cases],
                [captions[str(c["image_id"])]["text"] for c in batch_cases])
            floats = vae_float(channel, images) if attack["id"] == "vae_1" else None
            for j, case in enumerate(batch_cases[:count]):
                if case["mode"] != "native":
                    continue
                if generated_complete(root, case, attack["id"]):
                    continue
                base = dict(seed=seed(case, attack), parameters=attack,
                            batch_case_ids=[c["id"] for c in batch_cases], batch_position=j,
                            batch_width=batch_width, executed_steps=details["actual_denoising_steps"],
                            input_pixels=pixel_stats(np.asarray(images[j])))
                if flags[j]:
                    _, receipt = paths(root, method, case, attack["id"])
                    commit(receipt, dict(case_key=case["key"], condition=attack["id"],
                           status="blocked_generation", reason="pipeline safety checker", artifacts={}, **base))
                    continue
                if floats is not None:
                    if not np.array_equal(rgb8(floats[j]), outputs[j]):
                        raise RuntimeError("Independent VAE float reference does not round to pipeline RGB8")
                    output = io.BytesIO()
                    # np.save has deterministic bytes; no pickle and no ZIP timestamps.
                    np.save(output, floats[j], allow_pickle=False)
                    path = root / method / "float" / (case["id"] + ".npy")
                    checksum = immutable_bytes(path, output.getvalue())
                    base["float_path"] = path.relative_to(root).as_posix()
                    base["float_sha256"] = checksum
                    base["float_stats"] = pixel_stats(floats[j])
                base["quality"] = quality(np.asarray(images[j]), outputs[j])
                value = save_png(root, case, attack["id"], outputs[j], base)
                # The floating file is verified explicitly on resume/decode.
                if floats is not None and sha(root / value["float_path"]) != value["float_sha256"]:
                    raise RuntimeError("Float artifact readback failed")
            print(f"GENERATED {method} {attack['id']} image={group[0]['image_id']} "
                  f"batch={indexes[:count]} {details['batch_seconds']:.1f}s", flush=True)
