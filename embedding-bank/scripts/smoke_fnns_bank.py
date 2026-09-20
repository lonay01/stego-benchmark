"""Offline FNNS checkpoint + image smoke, including matched modes and resume.

Use an environment with the registry's torch/torchvision versions. Supply the
pinned OmniSealBench checkout, the downloaded official 48-bit TorchScript decoder
and one RGB512 PNG. This test never downloads files or writes to a production
bank. It is an interface check, not a COCO robustness experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from wmbench.adapters.bank_additions import FNNSAdapter
from wmbench.embedding_bank import POLICY, bank_registry, process_image, valid_record
from wmbench.io import atomic_bytes, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    model = bank_registry(project)["fnns_omniseal"]
    revision = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={args.repository.resolve().as_posix()}",
            "-C",
            str(args.repository),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if revision != model["source"]["revision"]:
        raise ValueError("Use the exact registered OmniSealBench revision.")
    with Image.open(args.image) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("Supply an RGB512 PNG; the test does not resize the input.")
    args.output.mkdir(parents=True, exist_ok=False)

    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("The FNNS smoke test forbids network access.")

    sys.addaudithook(offline)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    adapter = FNNSAdapter(
        {
            **model["options"],
            "model_id": model["id"],
            "device": "cpu",
            "repo_root": str(args.repository.resolve()),
            "weights": str(args.checkpoint.resolve()),
        }
    )
    assert all(not p.requires_grad for p in adapter.model.model.parameters())
    bank = args.output / "bank"
    relative = "_inputs/images/000000000139.png"
    atomic_bytes(bank / relative, args.image.read_bytes())
    fixture = {"image_id": "139", "path": relative, "sha256": sha256(args.image)}
    provenance = {
        "id": "fnns_cpu_smoke",
        "torch": torch.__version__,
        "device": "cpu",
        "source_revision": revision,
        "checkpoint_sha256": sha256(args.checkpoint),
    }
    records = process_image(adapter, model, bank, fixture, POLICY, provenance)
    assert len(records) == 4
    for record in records:
        assert record["status"] in ("ok", "match_not_found_within_policy")
        if record["status"] == "ok":
            assert record["identity_check"]["status"] == "measured"
    files = {p.relative_to(bank): sha256(p) for p in bank.rglob("*") if p.is_file()}

    def no_inference(*_args, **_kwargs):
        raise AssertionError("Resume repeated embedding or decoding")

    adapter.embed = no_inference
    adapter.decode = no_inference
    process_image(adapter, model, bank, fixture, POLICY, provenance)
    assert all(sha256(bank / p) == value for p, value in files.items())
    assert all(
        valid_record(bank, p) is not None
        for p in (bank / model["id"] / "_meta/records").glob("*.json")
    )
    report = {
        "purpose": "Real pretrained CPU interface smoke, not COCO100/GPU validation",
        "status": "passed",
        "runtime": provenance,
        "image_sha256": fixture["sha256"],
        "records": records,
        "resume_repeated_inference": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output / "report.json", report)
    print(f"FNNS passed: four mode records, verified resume; {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
