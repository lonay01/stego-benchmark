"""Optional real-checkpoint CPU smoke; not the historical COCO recheck.

Usage: python diagnostics/smoke_fin.py --author-repo PATH --output NEW_DIRECTORY
The source checkout must be the pinned FIN revision and contain HEAVY weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]

from bank_diagnostics.worker import Stages, fin_stages
from wmbench.adapters.bank_additions import FINAdapter
from wmbench.embedding_bank import bank_registry, message_for_model
from wmbench.io import digest, sha256, write_json
from wmbench.worker import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.author_repo.resolve(), args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "rev-parse", "HEAD"],
        text=True).strip()
    if revision != "f77222ffe9089a5a204e0fd37bd1e33a905be1ed":
        raise ValueError("Unexpected FIN source revision")
    if output.exists():
        raise FileExistsError("Use a new smoke output directory")
    output.mkdir(parents=True)

    import torch
    from PIL import Image

    torch.set_num_threads(4)
    seed_everything(0)
    example = repo / "test_images/1.png"
    source = output / "author_example_512.png"
    with Image.open(example) as image:
        original_size = image.size
        image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC).save(source)
    model = bank_registry(ROOT.parent)["fin"]
    payload, indices = message_for_model(model, "285")
    options = dict(repo_root=str(repo), asset_root=str(repo / "experiments/HEAVY"),
                   weights="FED.pt", inl_weights="INL.pt", device="cpu", model_id="fin",
                   resolution_policy="residual_128_to_512")
    started = time.monotonic()
    adapter = FINAdapter(options)
    evidence = {"kind": "real FIN HEAVY checkpoint CPU smoke, NOT historical COCO results",
                "revision": revision, "source": str(example), "source_sha256": sha256(example),
                "source_original_size": original_size, "checkpoint_hashes": adapter.weight_hashes,
                "torch": torch.__version__, "payload_note": "Bank message for ID285; image is FIN's author example, not COCO285"}
    stages = Stages(output, digest(evidence), payload, indices)
    fin_stages(adapter, source, stages)
    evidence.update(seconds=time.monotonic() - started, stages=stages.rows)
    write_json(output / "smoke_summary.json", evidence)
    if len(stages.rows) != 10 or any(r["status"] != "measured" for r in stages.rows):
        raise RuntimeError("FIN smoke did not complete all ten diagnostic stages")
    print(f"REAL FIN SMOKE: 10 stages measured in {evidence['seconds']:.1f}s; {output}")


if __name__ == "__main__":
    main()
