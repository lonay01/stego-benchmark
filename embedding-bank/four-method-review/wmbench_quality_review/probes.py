"""Observed tensors and explicit native-core controls; no decoder sees answers."""

from contextlib import contextmanager
import numpy as np

from wmbench_author_reference.upstream import tensor_digest
from wmbench_full_payload.storage import EvidenceError
from wmbench_native_diagnostics.probes import (
    capture_forward, decode_float, pixels, reconstruct_same_latent, rgb_tensor,
)


@contextmanager
def capture(adapter, name):
    if name != "editguard":
        with capture_forward(adapter, name) as data:
            yield data
        return
    data = {}

    def observe(_module, _args, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise EvidenceError("Unexpected EditGuard forward contract")
        data["raw"] = pixels(output[1][:, [2, 1, 0]])  # native BGR -> report RGB

    handle = adapter.net.register_forward_hook(observe)
    try:
        yield data
    finally:
        handle.remove()


def decode(adapter, name, image, quantized=True):
    if name != "editguard":
        return decode_float(adapter, name, image)
    torch = adapter.torch
    value = rgb_tensor(torch, image, adapter.device)[:, [2, 1, 0]]
    with torch.no_grad():
        if quantized:
            value = adapter.model.Quantization(value)
        value = adapter.net(x=value, rev=True)
    value = value[-1] if isinstance(value, (tuple, list)) else value
    logits = value.detach().float().cpu().numpy().reshape(-1)
    if logits.size != 64 or not np.isfinite(logits).all():
        raise EvidenceError("Invalid EditGuard copyright logits")
    return {"bits": (logits > 0).astype(int).tolist(), "logits": logits.tolist(),
            "decoder_quantization": quantized}


def direct_latent(adapter, name, source, payload):
    """Native inference graph at a predeclared 256px diagnostic geometry."""
    import torch
    if name not in ("rosteals", "sepwater"):
        raise ValueError(name)
    network = adapter.wrapper.model if name == "rosteals" else adapter._load()
    cover = rgb_tensor(torch, source.astype(np.float32) / 255, adapter.device) * 2 - 1
    with torch.no_grad():
        latent = network.encode_first_stage(cover)
        marked, _ = network(latent, None, torch.tensor([payload], device=adapter.device, dtype=torch.float32))
        raw = pixels((network.decode_first_stage(marked) + 1) / 2)
        ae = reconstruct_same_latent(adapter, name, latent)
    return {"raw": raw, "ae": ae}


def editguard_checkpoint(adapter, checkpoint):
    """Require all active parameters and buffers to equal the selected state."""
    torch = adapter.torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise EvidenceError("EditGuard checkpoint is not a tensor state dictionary")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    network = getattr(adapter.net, "module", adapter.net)
    active = network.state_dict()
    wrong = [k for k, value in active.items() if k not in state or
             not isinstance(state[k], torch.Tensor) or state[k].shape != value.shape or
             not torch.equal(value.detach().cpu(), state[k].cpu())]
    if wrong or not active:
        raise EvidenceError(f"Unfilled EditGuard inference weights: {wrong[:8]}")
    return {"checked": len(active), "all_equal": True, "tensors": tensor_digest(network)}


def load_clean_checkpoint(adapter, checkpoint):
    # Validate using the restricted tensor loader before the upstream loading path.
    state = adapter.torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    network = getattr(adapter.net, "module", adapter.net)
    network.load_state_dict(state, strict=True)
    network.eval()
    return editguard_checkpoint(adapter, checkpoint)
