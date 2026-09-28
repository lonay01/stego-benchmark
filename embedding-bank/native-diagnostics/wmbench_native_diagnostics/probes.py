"""Observe the actual baseline forward pass without changing its output or RNG."""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path

import numpy as np

from wmbench.io import sha256


def rgb_tensor(torch, array, device):
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).permute(2, 0, 1)[None].to(device)


def pixels(tensor):
    value = tensor.detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3 or not np.isfinite(value.numpy()).all():
        raise ValueError("Expected finite RGB CHW/BCHW image tensor")
    return value.permute(1, 2, 0).numpy().copy()


@contextlib.contextmanager
def capture_forward(adapter, name):
    """Hooks copy tensors only. No decoding or random operation runs in the hook."""
    captures, patches, hooks = {}, [], []

    def patch(obj, attr, transform):
        # Restore the instance dictionary exactly, not a bound-method shadow.
        had_instance = attr in vars(obj)
        original_instance = vars(obj).get(attr)
        original = getattr(obj, attr)
        setattr(obj, attr, transform(original))
        patches.append((obj, attr, had_instance, original_instance))

    def image_hook(key, normalized=False):
        def observe(_module, _args, output):
            captures[key] = pixels((output + 1) / 2 if normalized else output)
        return observe

    try:
        if name == "hidden_benchmark":
            hooks.append(adapter.wrapper.encoder_decoder.encoder.register_forward_hook(image_hook("raw", True)))
        elif name in ("rosteals", "sepwater"):
            network = adapter.wrapper.model if name == "rosteals" else adapter._load()
            def wrap_encode(original):
                def wrapped(*args, **kwargs):
                    latent = original(*args, **kwargs)
                    captures["latent"] = latent.detach().clone()
                    return latent
                return wrapped
            def wrap_decode(original):
                def wrapped(*args, **kwargs):
                    output = original(*args, **kwargs)
                    captures["signed"] = pixels(output)
                    captures["raw"] = pixels((output + 1) / 2)
                    return output
                return wrapped
            patch(network, "encode_first_stage", wrap_encode)
            patch(network, "decode_first_stage", wrap_decode)
        elif name in ("sepmark", "distseal_posthoc_dcae"):
            def wrap_save(original):
                def wrapped(tensor, path):
                    captures["raw"] = pixels(tensor)
                    return original(tensor, path)
                return wrapped
            patch(adapter, "_save", wrap_save)
            if name == "distseal_posthoc_dcae":
                def wrap_encode(original):
                    def wrapped(*args, **kwargs):
                        output = original(*args, **kwargs)
                        captures["latent"] = output.latent.detach().clone()
                        return output
                    return wrapped
                patch(adapter.autoencoder, "encode", wrap_encode)
        else:
            raise ValueError(f"Unsupported diagnostic method: {name}")
        if name in ("hidden_benchmark", "rosteals"):
            def wrap_embed(original):
                def wrapped(*args, **kwargs):
                    output = original(*args, **kwargs)
                    captures["wrapper"] = pixels(output)
                    return output
                return wrapped
            patch(adapter.wrapper, "embed", wrap_embed)
        yield captures
    finally:
        for hook in hooks:
            hook.remove()
        for obj, attr, had_instance, original_instance in reversed(patches):
            if had_instance:
                setattr(obj, attr, original_instance)
            else:
                delattr(obj, attr)


def reconstruct_same_latent(adapter, name, latent):
    """No second posterior sample: use the latent from the marked forward pass."""
    import torch
    with torch.inference_mode():
        if name == "distseal_posthoc_dcae":
            output = adapter.autoencoder.decode(latent).sample
        else:
            model = adapter.wrapper.model if name == "rosteals" else adapter._load()
            output = model.decode_first_stage(latent)
    return pixels((output + 1) / 2)


def decode_float(adapter, name, image):
    """Fixed decoder/normalization; there is deliberately no reference argument."""
    import torch
    tensor = rgb_tensor(torch, image, adapter.device)
    with torch.inference_mode():
        if name == "hidden_benchmark":
            logits = adapter.wrapper.encoder_decoder.decoder(tensor * 2 - 1)
            threshold = .5
        elif name == "rosteals":
            logits = adapter.wrapper.model.decoder(tensor * 2 - 1)
            threshold = 0.
        elif name == "sepwater":
            logits = adapter._load().decoder(tensor * 2 - 1)
            threshold = 0.
        else:
            logits, threshold = adapter._extract_logits(tensor), adapter.threshold
    values = logits.detach().float().cpu().reshape(-1).numpy()
    if not np.isfinite(values).all():
        raise ValueError("Decoder returned non-finite logits")
    return {"bits": (values > threshold).astype(int).tolist(),
            "details": {"threshold": threshold, "logits": values.tolist(),
                        "min_abs_margin": float(np.min(np.abs(values - threshold))),
                        "normalization": "RGB[0,1]" if name == "distseal_posthoc_dcae" else "RGB[-1,1]"}}


def quantize(image, rule):
    """Predeclared serializers, never selected using the message or decoder."""
    scaled = np.clip(np.asarray(image, dtype=np.float32), 0, 1) * np.float32(255)
    if rule == "round_even":
        scaled = np.rint(scaled)
    elif rule == "floor":
        scaled = np.floor(scaled)
    elif rule == "half_up":
        scaled = np.floor(scaled + np.float32(.5))
    else:
        raise ValueError(rule)
    return scaled.astype(np.uint8)


def export_sepwater_author(raw, source):
    """Author embed.py NumPy float64 residual export, labelled diagnostic only."""
    cover = source.astype(np.float32) / np.float32(127.5) - np.float32(1)
    # Caller supplies the original signed tensor, not an inverted RGB conversion.
    residual = np.clip(raw, -1, 1) - cover
    return (np.clip(residual + source.astype(np.float64) / 127.5 - 1., -1, 1) * 127.5 + 127.5).astype(np.uint8)


def array_stats(raw, source):
    return {"min": float(raw.min()), "max": float(raw.max()),
            "fraction_below_zero": float(np.mean(raw < 0)), "fraction_above_one": float(np.mean(raw > 1)),
            "max_abs_rounding_change": float(np.max(np.abs(quantize(raw, "round_even") / 255 - np.clip(raw, 0, 1)))),
            "source_min": int(source.min()), "source_max": int(source.max())}


def class_source_files(cls):
    """Find source without requiring a transient module to stay in sys.modules.

    WIBE removes nested imports after constructing a wrapper. inspect then calls
    those Python classes "built-in", although their own methods still retain
    real co_filename paths. Do not reimport a model or modify the module cache.
    Missing source metadata is recorded, never confused with checkpoint validity.
    """
    label = f"{cls.__module__}.{cls.__qualname__}"
    issues = []
    try:
        path = inspect.getsourcefile(cls)
    except (TypeError, OSError) as error:
        path = None
        issues.append(f"{type(error).__name__}: {error}")
    if path and Path(path).is_file():
        return {path}, {"class": label, "strategy": "inspect", "issues": issues}
    files = set()
    # Own methods only: inherited torch.nn.Module methods are not model source.
    for member in vars(cls).values():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if inspect.isfunction(member):
            path = member.__code__.co_filename
            if path and not path.startswith("<") and Path(path).is_file():
                files.add(path)
    return files, {"class": label, "strategy": "method_code" if files else "unavailable",
                   "issues": issues}


def audit_model(adapter, name):
    """Record inference contracts and reject unfilled RoSteALS inference keys."""
    import torch
    from wmbench_full_payload.storage import EvidenceError
    network = (adapter.wrapper.encoder_decoder if name == "hidden_benchmark" else
               adapter.wrapper.model if name == "rosteals" else
               adapter._load() if name == "sepwater" else
               adapter.model if name == "distseal_posthoc_dcae" else adapter.encoder)
    modules = [network]
    if name == "sepmark":
        modules += [adapter.decoder, adapter.fragile_decoder]
    if name == "distseal_posthoc_dcae":
        modules += [adapter.autoencoder]
    training = [f"{index}:{key}" for index, module in enumerate(modules)
                for key, child in module.named_modules() if child.training]
    if training:
        raise EvidenceError(f"Inference modules left in training mode: {training[:10]}")
    files, resolutions = set(), []
    classes = [type(adapter), type(network)]
    if name in ("hidden_benchmark", "rosteals"):
        classes.append(type(adapter.wrapper))
    for cls in classes:
        paths, resolution = class_source_files(cls)
        files.update(paths)
        resolutions.append(resolution)
    audit = {"training_modules": training, "model_class": type(network).__name__,
             "normalization": "RGB[0,1]" if name == "distseal_posthoc_dcae" else "RGB[-1,1]",
             "source_hashes": {str(p): sha256(Path(p)) for p in sorted(files)},
             "source_resolution": resolutions,
             "parameter_dtypes": sorted({str(p.dtype) for m in modules for p in m.parameters()})}
    if name == "hidden_benchmark":
        audit["checkpoint_config"] = vars(adapter.wrapper.params)
    elif name == "rosteals":
        path = Path(adapter.wrapper.params.weights_path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        active = network.state_dict()
        keys = [key for key in active if key.startswith(("ae.", "control.", "decoder."))]
        if not keys:
            raise EvidenceError("No expected RoSteALS inference namespaces")
        missing = [key for key in keys if key not in state]
        mismatched = [key for key in keys if key in state and (active[key].shape != state[key].shape or
                       not torch.equal(active[key].detach().cpu(), state[key].detach().cpu()))]
        audit["checkpoint_coverage"] = {"checked_keys": len(keys), "missing": missing, "different": mismatched}
        if missing or mismatched:
            raise EvidenceError(f"RoSteALS checkpoint coverage failure: {missing[:5]} {mismatched[:5]}")
    elif name == "sepwater":
        audit["checkpoint_loading"] = adapter.load_details
    elif name == "distseal_posthoc_dcae":
        audit["checkpoint_loading"] = adapter.checkpoint_loading
    return audit
