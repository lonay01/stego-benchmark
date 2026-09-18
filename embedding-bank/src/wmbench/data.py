"""Import existing COCO PNGs without changing pixels or silently replacing the corpus."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

from .io import (
    atomic_bytes,
    digest,
    download,
    read_json,
    safe_extract,
    sha256,
    write_csv,
    write_json,
)
from .protocol import payload_bank


def import_dataset(
    folder: Path,
    output: Path,
    captions_path: Path | None = None,
    expected_manifest: Path | None = None,
) -> dict:
    if folder.suffix.lower() == ".zip":
        extracted = output / "imported_bundle"
        safe_extract(folder, extracted)
        folder = extracted
    captions = {}
    if captions_path:
        if captions_path.suffix.lower() == ".csv":
            with captions_path.open(encoding="utf-8-sig", newline="") as handle:
                captions = {str(int(r["coco_id"])): r["prompt"] for r in csv.DictReader(handle)}
        else:
            data = read_json(captions_path)
            if "annotations" in data:
                for annotation in sorted(data["annotations"], key=lambda r: r["id"]):
                    captions.setdefault(str(annotation["image_id"]), annotation["caption"])
            else:
                captions = {str(int(k)): v for k, v in data.items()}
    expected = None
    if expected_manifest:
        if expected_manifest.suffix == ".csv":
            with expected_manifest.open(encoding="utf-8-sig", newline="") as handle:
                expected = list(csv.DictReader(handle))
        else:
            value = read_json(expected_manifest)
            expected = value if isinstance(value, list) else value["images"]
    rows, seen = [], {}
    for path in sorted(folder.rglob("*.png")):
        # Only canonical COCO filename stems. Never import display cards or rectified photos.
        match = re.fullmatch(r"(?:\d{2}_)?(\d{1,12})", path.stem)
        if not match:
            continue
        image_id = str(int(match.group(1)))
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                continue
        file_hash = sha256(path)
        if image_id in seen:
            if seen[image_id] != file_hash:
                raise ValueError(
                    f"Conflicting versions for COCO {image_id}; select the exact PNG folder"
                )
            continue
        seen[image_id] = file_hash
        rows.append(
            {
                "image_id": image_id,
                "coco_id": int(image_id),
                "path": str(path.resolve()),
                "sha256": file_hash,
                "prompt": captions.get(image_id, ""),
                "width": 512,
                "height": 512,
                "split": "exploratory_coco100",
            }
        )
    rows.sort(key=lambda row: row["coco_id"])
    if not rows:
        raise ValueError(
            "No canonical RGB 512x512 COCO PNGs found; choose the original corpus folder"
        )
    if expected is not None:
        expected_hashes = {
            str(int(r.get("coco_id", r.get("image_id")))): r.get(
                "sha256", r.get("sha256_png", r.get("png_sha256", r.get("file_sha256")))
            )
            for r in expected
        }
        actual_hashes = {r["image_id"]: r["sha256"] for r in rows}
        if actual_hashes != expected_hashes:
            raise ValueError("Imported PNG IDs/hashes differ from the supplied legacy manifest")
    legacy_semantic_hash = None
    columns = ["position", "image_id", "filename", "source_file", "source_url", "sha256_png"]
    if expected and all(k in expected[0] for k in columns):
        import hashlib

        semantic = "\n".join(
            "\t".join(str(r[k]) for k in columns)
            for r in sorted(expected, key=lambda r: int(r["position"]))
        )
        legacy_semantic_hash = hashlib.sha256(semantic.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "wmbench.dataset.v1",
        "images": rows,
        "content_hash": digest([{k: r[k] for k in ("coco_id", "sha256")} for r in rows]),
        "legacy_verified": expected is not None,
        "legacy_semantic_hash": legacy_semantic_hash,
        "legacy_manifest_sha256": sha256(expected_manifest) if expected_manifest else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "dataset.json", manifest)
    write_csv(output / "dataset_manifest.csv", rows)
    write_csv(
        output / "payload_manifest.csv",
        [{"payload_id": i, "bits": "".join(map(str, b))} for i, b in enumerate(payload_bank())],
    )
    return manifest


def synthetic_fixture(output: Path, count: int = 5) -> dict:
    """Only for plumbing tests. Never labelled COCO or used as scientific measurements."""
    import numpy as np
    from .matching import save_rgb

    folder = output / "synthetic_images"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(3100 + i)
        image = rng.integers(20, 235, (512, 512, 3), dtype=np.uint8)
        save_rgb(folder / f"{i + 1:012d}.png", image)
    manifest = import_dataset(folder, output)
    manifest["schema"] = "wmbench.synthetic_fixture.v1"
    for row in manifest["images"]:
        row["split"] = "synthetic_fixture_NOT_COCO"
        row["prompt"] = "synthetic noise test pattern"
    write_json(output / "dataset.json", manifest)
    return manifest


def coco_captions(cache: Path) -> Path:
    """Fetch official annotations once; no new source images or payloads are selected here."""
    import zipfile

    path = cache / "captions_val2017.json"
    if not path.exists():
        archive = cache / "annotations_trainval2017.zip"
        download("https://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as handle:
            atomic_bytes(path, handle.read("annotations/captions_val2017.json"))
    return path
