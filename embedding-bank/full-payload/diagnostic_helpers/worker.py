"""One method per process. All writes go to diagnostic folders, never the bank."""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.embedding_bank import common_message, message_for_model, inference_identity, valid_record
from wmbench.io import digest, read_json, sha256, write_json
from wmbench.matching import load_rgb, psnr
from wmbench.worker import adapter_instance, runtime_info, seed_everything

from .core import bit_metrics, cached, freeze, inside, seed, separate_roots


class Stages:
    def __init__(self, root, signature, payload, indices):
        self.root, self.signature = root, signature
        self.payload, self.indices = payload, indices
        self.rows = []

    def measure(self, name, operation, *, control=False):
        """Persist every observation, including bit errors; never retry for a better BER."""
        path = self.root / f"{name}.json"
        key = digest({"case": self.signature, "stage": name})
        previous = cached(path, key)
        if previous and previous.get("status") != "error":
            self.rows.append(previous)
            return previous
        if previous:
            previous_path = self.root / "technical_error_history" / f"{time.time_ns()}_{name}.json"
            previous_path.parent.mkdir(exist_ok=True)
            path.replace(previous_path)
        start = time.monotonic()
        try:
            result = operation()
            files = result.pop("_files", [])
            row = {"status": "measured", **result}
            if "bits" in row:
                row.update(bit_metrics(row.pop("bits"), self.payload, self.indices))
        except Exception as error:
            files = []
            row = {"status": "error", "reason": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        row.update(stage=name, control=control, signature=key, seconds=time.monotonic() - start,
                   artifacts=[{"path": p.relative_to(self.root).as_posix(), "sha256": sha256(p)} for p in files])
        write_json(path, row)
        self.rows.append(row)
        print(f"DIAG {name}: {row['status']}; common30 errors={row.get('common30_errors', '-')}", flush=True)
        return row


def tensor_bits(adapter, image):
    with adapter.torch.inference_mode():
        logits = adapter._extract_logits(image).reshape(-1)
    if not adapter.torch.isfinite(logits).all():
        raise ValueError("Non-finite decoder logits")
    return {"bits": logits.gt(adapter.threshold).int().cpu().tolist(),
            "logits": logits.detach().cpu().tolist()}


def png_once(path: Path, writer):
    """Do not overwrite a serialized stage on resume, even with another valid PNG."""
    if path.exists():
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG: {path}")
            image.verify()
        return
    temporary = path.with_name(path.stem + ".pending.png")
    writer(temporary)
    with Image.open(temporary) as image:
        image.verify()
    temporary.replace(path)


def decoded_png(path: Path, decode):
    return {**decode(), "_files": [path]}


def fin_stages(adapter, source: Path, stages: Stages):
    """Author PIL128 path vs current tensor128 -> residual512 path; same payload."""
    import torch
    from torchvision import transforms as T
    from torchvision.utils import save_image

    image = adapter._image(str(source))
    with Image.open(source) as pil:
        native_author = T.ToTensor()(T.Resize(128)(pil.convert("RGB")))[None].to(adapter.device)
    native_bank = adapter._native_image(image)
    message = torch.tensor([stages.payload], device=adapter.device, dtype=torch.float32) - .5

    def decode_native(value, use_inl=True):
        with torch.inference_mode():
            normalized = value * 2 - 1
            corrected = adapter.inl(normalized, rev=True) if use_inl else normalized
            _, logits = adapter.fed([corrected, torch.zeros_like(message)], rev=True)
        if not torch.isfinite(logits).all():
            raise ValueError("FIN returned non-finite logits")
        return {"bits": logits.reshape(-1).gt(0).int().cpu().tolist(),
                "logits": logits.reshape(-1).cpu().tolist()}

    for name, native in (("author128", native_author), ("bank128", native_bank)):
        with torch.inference_mode():
            marked, _ = adapter.fed([native * 2 - 1, message])
            marked = (marked + 1) / 2
        stages.measure(f"{name}_float", lambda: decode_native(marked))
        stages.measure(f"{name}_clipped_float", lambda: decode_native(marked.clamp(0, 1)))
        png = stages.root / f"{name}.png"
        # Match the corresponding serializer, rather than changing both factors.
        if name == "author128":
            png_once(png, lambda p: save_image(marked, str(p)))
        else:
            png_once(png, lambda p: adapter._save(marked, str(p)))
        stages.measure(f"{name}_png", lambda: decoded_png(png, lambda: decode_native(adapter._image(str(png)))))
        if name == "author128":
            stages.measure("author128_no_inl_control", lambda: decode_native(marked, False), control=True)
        else:
            with torch.inference_mode():
                lifted = image + adapter._resize(marked - native, (512, 512))
            stages.measure("bank512_float", lambda: tensor_bits(adapter, lifted))
            stages.measure("bank512_clipped_float", lambda: tensor_bits(adapter, lifted.clamp(0, 1)))
            png512 = stages.root / "bank512.png"
            png_once(png512, lambda p: adapter._save(lifted, str(p)))
            stages.measure("bank512_png", lambda: decoded_png(png512, lambda: tensor_bits(adapter, adapter._image(str(png512)))))


def hidden_stages(adapter, source: Path, stages: Stages):
    """Test the pinned benchmark network at its own H,W and after residual lifting.

    This is the available third-party PyTorch checkpoint, not original Torch7.
    No assumption of 128px: dimensions are read from its saved configuration.
    """
    import torch
    from torchvision import transforms as T
    from torchvision.utils import save_image
    from wibench.utils import resize_torch_img, normalize_image, denormalize_image

    wrapper = adapter.wrapper
    image = adapter._image(str(source))
    native = resize_torch_img(image, (wrapper.params.H, wrapper.params.W))
    message = adapter._payload(stages.payload)

    def decode(value):
        with torch.inference_mode():
            logits = wrapper.encoder_decoder.decoder(normalize_image(value).to(wrapper.device))
        if not torch.isfinite(logits).all():
            raise ValueError("HiDDeN returned non-finite logits")
        return {"bits": logits.reshape(-1).gt(.5).int().cpu().tolist(),
                "logits": logits.reshape(-1).cpu().tolist(),
                "native_hw": [wrapper.params.H, wrapper.params.W]}

    def save(value, target):
        from wmbench.matching import save_rgb
        array = value.detach().cpu().squeeze().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
        save_rgb(target, array)

    # Upstream validation uses CenterCrop, whereas WIBE resizes the whole image.
    with Image.open(source) as pil:
        cropped = T.ToTensor()(T.CenterCrop((wrapper.params.H, wrapper.params.W))(pil.convert("RGB")))
    with torch.inference_mode():
        authored = denormalize_image(wrapper.encoder_decoder.encoder(
            normalize_image(cropped).to(wrapper.device), message.watermark.to(wrapper.device)).cpu())
    stages.measure("author_centercrop_float", lambda: decode(authored))
    stages.measure("author_centercrop_clipped_float", lambda: decode(authored.clamp(0, 1)))
    author_png = stages.root / "author_centercrop.png"
    png_once(author_png, lambda p: save_image(authored, str(p)))
    stages.measure("author_centercrop_png", lambda: decoded_png(author_png, lambda: decode(adapter._image(str(author_png)))))

    with torch.inference_mode():
        encoded = wrapper.encoder_decoder.encoder(normalize_image(native).to(wrapper.device),
                                                  message.watermark.to(wrapper.device))
        marked = denormalize_image(encoded.cpu())
    stages.measure("checkpoint_native_float", lambda: decode(marked))
    stages.measure("checkpoint_native_clipped_float", lambda: decode(marked.clamp(0, 1)))
    png = stages.root / "checkpoint_native.png"
    png_once(png, lambda p: save(marked, p))
    stages.measure("checkpoint_native_png", lambda: decoded_png(png, lambda: decode(adapter._image(str(png)))))
    with torch.inference_mode():
        lifted = wrapper.embed(image, message)
    stages.measure("bank512_float", lambda: {"bits": np.asarray(wrapper.extract(
        lifted, adapter._payload([0] * len(stages.payload)))).reshape(-1).tolist()})
    stages.measure("bank512_clipped_float", lambda: {"bits": np.asarray(wrapper.extract(
        lifted.clamp(0, 1), adapter._payload([0] * len(stages.payload)))).reshape(-1).tolist()})
    png512 = stages.root / "bank512.png"
    png_once(png512, lambda p: save(lifted, p))
    stages.measure("bank512_png", lambda: {"bits": np.asarray(wrapper.extract(
        adapter._image(str(png512)), adapter._payload([0] * len(stages.payload)))).reshape(-1).tolist(),
        "_files": [png512]})


def distseal_stages(adapter, source: Path, stages: Stages):
    """Separate autoencoder reconstruction error from latent-watermark error."""
    torch = adapter.torch
    image = adapter._image(str(source))
    message = torch.tensor([stages.payload], device=adapter.device, dtype=torch.float32)
    with torch.inference_mode():
        latent = adapter.autoencoder.encode(image * 2 - 1).latent
        reconstruction = (adapter.autoencoder.decode(latent).sample + 1) / 2
        prediction = adapter.model.embedder(latent, message)
        marked = (adapter.autoencoder.decode(adapter.model.blender(latent, prediction)).sample + 1) / 2
    rec_path, wm_path = stages.root / "ae_only.png", stages.root / "ae_watermarked.png"
    png_once(rec_path, lambda p: adapter._save(reconstruction, str(p)))
    png_once(wm_path, lambda p: adapter._save(marked, str(p)))
    x, rec, wm = load_rgb(source), load_rgb(rec_path), load_rgb(wm_path)
    stages.measure("ae_quality_control", lambda: {
        "psnr_original_to_ae": psnr(x, rec), "psnr_original_to_watermarked": psnr(x, wm),
        "psnr_ae_to_watermarked": psnr(rec, wm),
        "reference_for_production_matched": "original COCO PNG (unchanged)",
        "note": "PSNRs are not additive; reconstruction is a control, not a replacement reference",
        "_files": [rec_path, wm_path],
    }, control=True)
    stages.measure("ae_only_decode_control", lambda: tensor_bits(adapter, adapter._image(str(rec_path))), control=True)
    stages.measure("ae_watermarked_float", lambda: tensor_bits(adapter, marked))
    stages.measure("ae_watermarked_png", lambda: tensor_bits(adapter, adapter._image(str(wm_path))))


def verify_prepared_against_history(bank, model, prepared):
    path = bank / model["id"] / "_meta/model_identity.json"
    current = inference_identity(prepared["identity"])
    if path.is_file() and read_json(path) != current:
        raise ValueError("Prepared model/checkpoint identity differs from the saved bank. Not a like-for-like recheck.")
    return current


def run_model(job: dict):
    bank, output, slot = (Path(job[k]).resolve() for k in ("bank", "output", "slot"))
    separate_roots(bank, output, slot)
    model = job["model"]
    prepared = read_json(output / "prepared.json")
    config = read_json(prepared["config"])
    identity = verify_prepared_against_history(bank, model, prepared)
    freeze(output / "inference_identity.json", identity)
    seed_everything(0)
    adapter = adapter_instance(config)
    if hasattr(adapter, "initialize"):
        adapter.initialize()
    write_json(output / "runtime.json", runtime_info())
    failures = 0
    for image in job["images"]:
        image_id = str(image["image_id"])
        root = output / "cases" / f"{int(image_id):012d}"
        root.mkdir(parents=True, exist_ok=True)
        source = inside(bank, image["path"])
        if sha256(source) != image["sha256"]:
            raise ValueError("Original source checksum changed")
        record_path = bank / model["id"] / "_meta/records" / f"{int(image_id):012d}_native.json"
        record = valid_record(bank, record_path) if record_path.exists() else None
        expected, indices = message_for_model(model, image_id)
        if record:
            if record["payload"] != expected or record["common_indices"] != indices:
                raise ValueError("Saved payload/indices differ from the pinned bank protocol")
            if record["source"] != image["path"] or record["status"] != "ok":
                raise ValueError("Saved Native record does not match source/status")
        if [expected[i] for i in indices] != common_message(image_id):
            raise ValueError("Common30 payload mapping failed")
        signature = digest({"run": job["identity"], "model": identity, "image": image,
                            "payload": expected, "indices": indices, "old_record": record})
        if cached(root / "complete.json", signature):
            print(f"REUSE {model['id']} {image_id}", flush=True)
            continue
        stages = Stages(root, signature, expected, indices)
        write_json(root / "payload_audit.json", {"payload": expected, "common_indices": indices,
                   "common30": common_message(image_id), "historical_record_present": bool(record)})
        try:
            old_png = None
            if record:
                old_png = inside(bank, record["output"])
                old_meta = inside(bank, record["metadata"])
                # Some upstream decoders may write beside metadata: give them a COPY.
                metadata = root / "historical_keys" / "metadata.json"
                shutil.copytree(old_meta.parent, metadata.parent, dirs_exist_ok=True)
                seed_everything(seed(image_id, "native|decode"))
                stages.measure("historical_png_decode", lambda: adapter.decode(str(old_png), str(metadata)))
                historical_identity = record_path.with_name(record_path.stem + "_identity.json")
                if historical_identity.is_file():
                    prior = valid_record(bank, historical_identity)
                    stages.measure("historical_bit_recalculation", lambda: {
                        "bits": prior["decoded_bits"], "stored_common30_ber": prior["common30_ber"],
                        "stored_native_ber": prior["native_ber"],
                    })
            repeat_png, repeat_meta = root / "repeat.png", root / "repeat_keys/metadata.json"
            repeat_meta.parent.mkdir(parents=True, exist_ok=True)

            def embed_once():
                seed_everything(seed(image_id, "embed"))
                adapter.embed(str(source), str(repeat_png), expected, str(repeat_meta))
                result = {"psnr": psnr(load_rgb(source), load_rgb(repeat_png)),
                          "_files": [repeat_png, *[p for p in repeat_meta.parent.rglob('*') if p.is_file()]]}
                if old_png:
                    result["pixels_equal_historical"] = bool(np.array_equal(load_rgb(old_png), load_rgb(repeat_png)))
                    result["file_equal_historical"] = sha256(old_png) == sha256(repeat_png)
                return result

            stages.measure("repeat_embedding", embed_once)
            if repeat_png.exists() and repeat_meta.exists():
                seed_everything(seed(image_id, "native|decode"))
                stages.measure("repeat_png_decode", lambda: adapter.decode(str(repeat_png), str(repeat_meta)))
                # The six mild-error methods get clean controls too; no FPR estimate from 5 samples.
                stages.measure("original_decode_control", lambda: adapter.decode(str(source), str(repeat_meta)), control=True)
            if model["id"] == "fin":
                fin_stages(adapter, source, stages)
            elif model["id"] == "hidden_benchmark":
                hidden_stages(adapter, source, stages)
            elif model["id"] == "distseal_posthoc_dcae":
                distseal_stages(adapter, source, stages)
            errors = [r for r in stages.rows if r["status"] == "error"]
            if errors:
                failures += 1
                write_json(root / "case_error.json", {"errors": errors})
            else:
                artifacts = [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)}
                             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "complete.json"]
                write_json(root / "complete.json", {"signature": signature, "status": "completed",
                           "artifacts": artifacts, "stages": len(stages.rows)})
        except Exception as error:
            failures += 1
            write_json(root / "case_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            print(f"ERROR {model['id']} {image_id}: {error}", flush=True)
        print(f"DIAG finished image {model['id']} {image_id}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} diagnostic cases have technical errors; inspect case_error.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    run_model(read_json(parser.parse_args().job))
