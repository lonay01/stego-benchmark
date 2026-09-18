"""Restricted author-checkpoint compatibility; never fall back to arbitrary pickle."""

from __future__ import annotations

from contextlib import nullcontext


def load_tensor_checkpoint(path, *, numpy_metadata=False, namespace_metadata=False):
    import torch

    allowed = []
    if namespace_metadata:
        from argparse import Namespace

        allowed.append(Namespace)
    if numpy_metadata:
        import numpy as np

        # Known inert metadata types in the published SSL checkpoint. Unknown
        # globals still fail. This list is scoped to one load, not process-wide.
        from numpy.core.multiarray import scalar, _reconstruct

        allowed.extend([scalar, _reconstruct, np.ndarray, np.dtype])
        allowed.extend(
            type(np.dtype(dtype))
            for dtype in (
                np.float32,
                np.float64,
                np.int32,
                np.int64,
                np.uint32,
                np.uint64,
                np.bool_,
            )
        )
    if allowed and not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("This checkpoint needs the isolated PyTorch >=2.4 safe-globals loader")
    context = torch.serialization.safe_globals(allowed) if allowed else nullcontext()
    with context:
        return torch.load(path, map_location="cpu", weights_only=True)
