"""Read-only bank probe: replay a healthy carrier before recovering empty ones."""

import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from wmbench_digital.worker import load_adapter
from wmbench.worker import seed_everything
from stegastamp_cuda_hotfix import install_worker_guard
from stegastamp_byte_recovery import TARGETS, sha


def main(job_path, destination):
    job = json.loads(Path(job_path).read_text())
    bank = Path(job["bank"])
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    install_worker_guard()
    seed_everything(0)
    adapter = load_adapter(job)
    # Same initial carrier and order as the original bank, then a healthy control.
    stems = ["000000000285_m01", "000000106563_m01", *TARGETS]
    results = []
    for stem in stems:
        row = json.loads((bank / f"stegastamp/_meta/records/{stem}_native.json").read_text())
        seed_everything(int(row["signature"][:8], 16))
        target = out / (stem + ".png")
        adapter.embed(str(bank / row["source_crop"]), str(target), row["assignment"]["payload"],
                      str(out / (stem + ".metadata.json")))
        expected = next(a["sha256"] for a in row["artifacts"] if a["path"] == row["output"])
        result = dict(stem=stem, exact=sha(target) == expected, actual=sha(target))
        old = bank / row["output"]
        if old.stat().st_size:
            a = np.asarray(Image.open(target)).astype(int)
            b = np.asarray(Image.open(old)).astype(int)
            result.update(changed_values=int(np.count_nonzero(a != b)),
                          max_pixel_difference=int(np.abs(a-b).max()))
        results.append(result)
        print(json.dumps(result), flush=True)
    (out / "probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
