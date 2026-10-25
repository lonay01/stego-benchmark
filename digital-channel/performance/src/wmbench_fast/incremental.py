"""Checkpoint successfully audited bank images without weakening pending-input checks."""

from pathlib import Path
import tempfile

from wmbench.io import read_json
from wmbench_full_payload.storage import EvidenceError
from .cache import pack, unpack, validate_catalogue, original_bank


def collect_images(ctx, model, full_audit=False):
    cases, coverage = [], []
    scratch = Path(ctx["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    # The original collector writes its per-call catalogue here, never over the
    # canonical complete catalogue. Only successful image audits are committed.
    with tempfile.TemporaryDirectory(prefix="bank-audit-", dir=scratch) as temporary:
        for image in ctx["images"]:
            single = dict(ctx, images=[image])
            path = ctx["output"] / model["id"] / "_fast/bank-images" / (str(image["image_id"]) + ".json.gz")
            pair = None
            if not full_audit and path.exists():
                value = unpack(path)
                if value["identity"] != ctx["identity"]:
                    raise EvidenceError("Per-image bank checkpoint belongs to another run")
                pair = validate_catalogue(value["cases"], value["coverage"], single, model)
                if pair is None:
                    raise EvidenceError("Incomplete per-image bank checkpoint")
                print(f"BANK IMAGE REUSED {model['id']} {image['image_id']}", flush=True)
            if pair is None:
                staging = dict(single, output=Path(temporary))
                rows = original_bank.collect(staging, model)
                covered = read_json(Path(temporary) / model["id"] / "coverage.json")
                pair = validate_catalogue(rows, covered, single, model)
                if pair is None:
                    raise EvidenceError("Fresh per-image bank audit incomplete")
                pack(path, dict(identity=ctx["identity"], cases=pair[0], coverage=pair[1]))
            cases.extend(pair[0])
            coverage.extend(pair[1])
    return cases, coverage
