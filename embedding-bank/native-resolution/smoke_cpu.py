"""Real FIN/DWT smoke on an author example; explicitly NOT the COCO canary."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("fin", "dwt_dct_svd"), required=True)
    parser.add_argument("--deps", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new smoke output directory")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={args.author_repo.as_posix()}",
        "-C", str(args.author_repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != "f77222ffe9089a5a204e0fd37bd1e33a905be1ed":
        raise ValueError("Unexpected FIN source revision")
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="native-geometry-smoke-") as temporary:
        project = Path(temporary)
        with zipfile.ZipFile(root / "dist/Embedding_Bank_NATIVE_RESOLUTION_CANARY_source.zip") as archive:
            archive.extractall(project)
        sys.path.insert(0, str(project / "src"))
        if args.deps:
            sys.path.insert(0, str(args.deps.resolve()))
        import torch
        from PIL import Image
        from wmbench.io import sha256, write_json
        from wmbench.worker import seed_everything
        from wmbench_native_canary.contracts import models_for_trial
        from wmbench_native_canary.worker import process_pair, complete

        torch.set_num_threads(4)
        seed_everything(0)
        args.output.mkdir(parents=True)
        source = args.output / "_inputs/author_example_512.png"
        source.parent.mkdir()
        example = args.author_repo / "test_images/1.png"
        with Image.open(example) as image:
            image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC).save(source)
        descriptor = {"image_id": 1, "path": source.relative_to(args.output).as_posix(), "sha256": sha256(source)}
        model = models_for_trial(project)[args.method]
        started = time.monotonic()
        if args.method == "fin":
            from wmbench_native_canary.adapters import FINNative
            adapter = FINNative(dict(repo_root=str(args.author_repo), asset_root=str(args.author_repo / "experiments/HEAVY"),
                weights="FED.pt", inl_weights="INL.pt", device="cpu", model_id="fin", resolution_policy="author_128_direct"))
        else:
            from wmbench.adapters.legacy import InvisibleWatermarkAdapter
            adapter = InvisibleWatermarkAdapter({"method": "dwtDctSvd"})
        execution = {"kind": "real-checkpoint CPU smoke on FIN author example, NOT COCO",
                     "torch": torch.__version__, "source_revision": revision,
                     "original_image_sha256": sha256(example)}
        rows = [process_pair(adapter, model, args.output, descriptor, rep, execution) for rep in range(1, 6)]
        repeated = process_pair(adapter, model, args.output, descriptor, 1, execution)
        assert rows[0] == repeated and complete(args.output, model, [descriptor])
        evidence = {**execution, "method": args.method, "seconds": time.monotonic() - started,
                    "size": model["native_geometry"]["width"], "adapter_bits": model["native_bits"],
                    "messages": [{k: r.get(k) for k in ("replicate", "status", "ber", "exact")} for r in rows],
                    "resume_identical": True}
        write_json(args.output / "smoke_summary.json", evidence)
        print(evidence)
        if any(r["status"] == "error" for r in rows):
            raise RuntimeError("Smoke had technical errors; inspect preserved records")


if __name__ == "__main__":
    main()
