"""Thin file interfaces around the audited author-path implementations.

No search over serializers, checkpoints, bit positions, or output messages.
The author reconstruction/export is retained; only the benchmark's source crop
and independently assigned full random message are supplied from outside.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.io import write_json
from wmbench.matching import save_rgb


class AuthorAdapter:
    def __init__(self, name, source, root):
        import torch
        self.torch, self.name = torch, name
        if name == "distseal":
            from authornight.distseal import AuthorDistSeal
            self.reference = AuthorDistSeal(Path(source), Path(root))
            self.native_bits = 64
        else:
            from authornight.latent import AuthorLatent
            self.reference = AuthorLatent(name, Path(source), Path(root))
            self.native_bits = 100

    def embed(self, input_path, output_path, payload, metadata_path):
        if len(payload) != self.native_bits or any(b not in (0, 1) for b in payload):
            raise ValueError("Wrong full author payload")
        torch, ref = self.torch, self.reference
        with Image.open(input_path) as file:
            image = file.convert("RGB")
        message = torch.tensor([payload], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            if self.name == "distseal":
                from authornight.metrics import pixels
                cover = ref.tensor(np.array(image))
                latent, size = ref.ae.encode_pre_quant(cover)
                baseline = ref.ae.decode(latent, size).clamp(0, 1)
                marked = ref.ae.decode(ref.wm.blender(latent, ref.wm.embedder(latent, message)), size)
                result = pixels(marked.clamp(0, 1), "floor")
                # Reconstruction-only control is independent of message and saved
                # once per image by the worker, never substituted for source PSNR.
                self.last_reconstruction = pixels(baseline, "floor")
            else:
                cover = ref.tensor(image)
                latent = ref.model.encode_first_stage(cover)
                marked_latent, _ = ref.model(latent, None, message)
                stego = ref.model.decode_first_stage(marked_latent)
                result = ref.export(stego, cover, image)
        save_rgb(output_path, np.asarray(result, dtype=np.uint8))
        write_json(metadata_path, {"native_bits": self.native_bits, "author_method": self.name,
                                  "expected_bits": payload, "ecc": "none"})

    def decode(self, input_path, metadata_path):
        # No expected message or original carrier enters either author decoder.
        with Image.open(input_path) as file:
            image = file.convert("RGB")
        ref = self.reference
        logits = ref.decode(ref.tensor(np.array(image) if self.name == "distseal" else image))
        return {"bits": (np.asarray(logits).reshape(-1) > 0).astype(int).tolist(), "ecc": "none"}
