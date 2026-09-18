"""Freeze the historical COCO100 corpus and disjoint detector-calibration inputs.

Only ``test_count`` changes the test prefix: the complete historical 100-image
pool is always reserved. Persist this entire input directory when moving from
Colab to a server; absolute paths are reconstructed, never used as identities.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageOps, __version__ as pillow_version

from .io import atomic_bytes, digest, download, read_json, sha256, write_json
from .protocol import LEGACY_DATASET_HASH

SCHEMA = "wmbench.portable_inputs.v1"
COCO_S3 = "https://s3.amazonaws.com/images.cocodataset.org"
LEGACY_COLUMNS = ("position", "image_id", "filename", "source_file", "source_url", "sha256_png")
PREPROCESS = {
    "name": "coco_calibration_exif_rgb_short_edge_center_crop_v1",
    "size": 512,
    "resampling": "Pillow.BILINEAR",
    "resize_rounding": "integer_floor_long_edge",
    "crop_rounding": "integer_floor_top_left",
    "png_compress_level": 6,
}


def _legacy_rows(path: Path) -> tuple[list[dict], str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or any(not set(LEGACY_COLUMNS) <= row.keys() for row in rows):
        raise ValueError("Need the exact historical 100-row manifest.csv with legacy columns")
    if len({int(row["image_id"]) for row in rows}) != 100:
        raise ValueError("Duplicate image IDs in the historical manifest")
    if len({int(row["position"]) for row in rows}) != 100:
        raise ValueError("Duplicate positions in the historical manifest")
    semantic = "\n".join(
        "\t".join(str(row[key]) for key in LEGACY_COLUMNS)
        for row in sorted(rows, key=lambda row: int(row["position"]))
    )
    semantic_hash = hashlib.sha256(semantic.encode("utf-8")).hexdigest()
    if semantic_hash != LEGACY_DATASET_HASH:
        raise ValueError(
            "Historical COCO100 semantic hash mismatch: "
            f"expected {LEGACY_DATASET_HASH}, received {semantic_hash}. "
            "Do not replace missing test images with newly downloaded JPEGs."
        )
    return rows, semantic_hash


def _source_images(folder: Path, expected: list[dict]) -> dict[int, Path]:
    image_folder = folder / "test"
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Need historical fixed512 PNG directory: {image_folder}")
    paths = {}
    for path in sorted(image_folder.rglob("*.png")):
        match = re.fullmatch(r"(?:\d{2}_)?(\d{1,12})", path.stem)
        if not match:
            continue
        image_id = int(match.group(1))
        if image_id in paths:
            raise ValueError(f"Duplicate canonical PNG for COCO {image_id}")
        paths[image_id] = path
    expected_hashes = {int(row["image_id"]): row["sha256_png"] for row in expected}
    if set(paths) != set(expected_hashes):
        raise ValueError("Historical PNG IDs differ from the full reserved 100-image manifest")
    for image_id, path in paths.items():
        if sha256(path) != expected_hashes[image_id]:
            raise ValueError(f"Historical PNG hash mismatch: COCO {image_id}: {path}")
        _check_fixed_rgb(path)
    return paths


def _check_fixed_rgb(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"Expected byte-preserved RGB 512x512 PNG: {path}")
        image.load()


def _captions(base: Path, supplied: Path | None) -> tuple[Path, dict[int, dict]]:
    destination = base / "annotations/captions_val2017.json"
    if supplied is not None:
        if not supplied.is_file():
            raise FileNotFoundError(f"COCO captions file does not exist: {supplied}")
        if destination.exists() and sha256(destination) != sha256(supplied):
            raise ValueError("Supplied captions differ from the frozen input-bundle annotations")
        if not destination.exists():
            atomic_bytes(destination, supplied.read_bytes())
    elif not destination.exists():
        archive = base / "annotations/annotations_trainval2017.zip"
        download(f"{COCO_S3}/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as handle:
            atomic_bytes(destination, handle.read("annotations/captions_val2017.json"))
    value = read_json(destination)
    if not isinstance(value, dict) or not isinstance(value.get("images"), list):
        raise ValueError("Need official captions_val2017 JSON containing images and annotations")
    if not isinstance(value.get("annotations"), list):
        raise ValueError("COCO captions JSON has no annotations list")
    images = {}
    for row in value["images"]:
        image_id = int(row["id"])
        filename = f"{image_id:012d}.jpg"
        if row.get("file_name") != filename or image_id in images:
            raise ValueError("Invalid/duplicate image ID or noncanonical COCO JPEG filename")
        images[image_id] = {"image_id": image_id, "filename": filename, "prompt": ""}
    for annotation in sorted(value["annotations"], key=lambda row: int(row["id"])):
        image_id = int(annotation["image_id"])
        if image_id in images and not images[image_id]["prompt"]:
            caption = annotation["caption"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("Empty or non-string official COCO caption")
            images[image_id]["prompt"] = caption
    if any(not row["prompt"] for row in images.values()):
        raise ValueError("At least one COCO image has no usable caption")
    return destination, images


def _freeze_json(path: Path, value: dict) -> dict:
    if path.exists():
        previous = read_json(path)
        if previous != value:
            raise ValueError(f"Frozen input provenance changed: {path}")
        return previous
    write_json(path, value)
    return value


def _dataset(rows: list[dict], *, legacy_hash: str | None = None) -> dict:
    manifest = {
        "schema": "wmbench.dataset.v1",
        "images": rows,
        "content_hash": digest([{k: row[k] for k in ("coco_id", "sha256")} for row in rows]),
        "prompt_hash": digest([{k: row[k] for k in ("coco_id", "prompt")} for row in rows]),
        "legacy_verified": legacy_hash is not None,
        "legacy_semantic_hash": legacy_hash,
    }
    return manifest


def _calibration_image(source: Path, target: Path) -> None:
    """Explicit calibration-only transform; historical test PNGs are never transformed."""
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        if min(width, height) <= 0:
            raise ValueError(f"Invalid source dimensions: {source}")
        if width <= height:
            dimensions = (512, max(512, height * 512 // width))
        else:
            dimensions = (max(512, width * 512 // height), 512)
        resized = image.resize(dimensions, Image.Resampling.BILINEAR)
        left, top = (dimensions[0] - 512) // 2, (dimensions[1] - 512) // 2
        cropped = resized.crop((left, top, left + 512, top + 512))
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG", compress_level=6)
        atomic_bytes(target, buffer.getvalue())


def prepare_inputs(
    source_root: Path | None,
    persistent_root: Path,
    test_count: int,
    calibration_count: int = 1000,
    *,
    captions_path: Path | None = None,
) -> dict:
    """Prepare byte-preserved tests plus a fixed, disjoint clean-calibration pool.

    ``source_root`` initially contains ``test/*.png`` and ``manifest.csv`` from
    the historical digital experiment. After moving the entire persistent input
    bundle to another machine, pass ``None`` or the old unavailable path: verified
    reserved PNGs and frozen metadata suffice. No replacement corpus is allowed.

    Captions use the first official annotation by annotation ID. Calibration IDs
    are sorted numerically after excluding *all 100* test IDs, independently of
    ``test_count``. Source JPEG and generated PNG hashes are recorded per image;
    interrupted downloads resume, but corrupted completed images fail closed.
    """
    if type(test_count) is not int or not 1 <= test_count <= 100:
        raise ValueError("test_count must be an integer from 1 to 100")
    if type(calibration_count) is not int or calibration_count < 1:
        raise ValueError("calibration_count must be a positive integer")
    source_root = Path(source_root) if source_root is not None else None
    base = Path(persistent_root).resolve() / "inputs"
    reserved = base / "reserved_coco100"
    local_legacy = reserved / "manifest.csv"
    source_paths: dict[int, Path] = {}
    if source_root is not None and source_root.is_dir():
        source_manifest = source_root / "manifest.csv"
        expected, legacy_hash = _legacy_rows(source_manifest)
        source_paths = _source_images(source_root, expected)
        if local_legacy.exists() and sha256(local_legacy) != sha256(source_manifest):
            raise ValueError("Historical source manifest differs from the persistent copy")
        if not local_legacy.exists():
            atomic_bytes(local_legacy, source_manifest.read_bytes())
    elif local_legacy.exists():
        expected, legacy_hash = _legacy_rows(local_legacy)
    else:
        raise FileNotFoundError(
            "Need the complete historical coco_fixed_512/{test,manifest.csv} directory, "
            "or the previously prepared persistent inputs directory."
        )
    expected.sort(key=lambda row: int(row["image_id"]))
    # Validate or restore the reserved files before downloading any calibration data.
    for row in expected:
        image_id = int(row["image_id"])
        target = reserved / "images" / f"{image_id:012d}.png"
        if not target.exists():
            if image_id not in source_paths:
                raise FileNotFoundError(
                    f"Missing reserved test PNG; restore original bytes: {target}"
                )
            atomic_bytes(target, source_paths[image_id].read_bytes())
        if sha256(target) != row["sha256_png"]:
            raise ValueError(f"Corrupted reserved test PNG; restore original bytes: {target}")
        _check_fixed_rgb(target)
    captions_file, caption_rows = _captions(
        base, Path(captions_path) if captions_path is not None else None
    )
    reserved_ids = {int(row["image_id"]) for row in expected}
    if not reserved_ids <= caption_rows.keys():
        raise ValueError("Official captions do not contain every historical test image")
    annotation_hash = sha256(captions_file)
    pool = {
        "schema": SCHEMA,
        "legacy_semantic_hash": legacy_hash,
        "legacy_manifest_sha256": sha256(local_legacy),
        "captions_sha256": annotation_hash,
        "caption_selection": "lowest_annotation_id",
        "selection": "ascending_coco_id; reserve_all_100_independent_of_test_count",
        "images": [
            {
                "image_id": str(int(row["image_id"])),
                "coco_id": int(row["image_id"]),
                "filename": f"{int(row['image_id']):012d}.png",
                "sha256": row["sha256_png"],
                "prompt": caption_rows[int(row["image_id"])]["prompt"],
            }
            for row in expected
        ],
    }
    _freeze_json(reserved / "pool.json", pool)
    all_rows = [
        {
            **{key: row[key] for key in ("image_id", "coco_id", "sha256", "prompt")},
            "path": str((reserved / "images" / row["filename"]).resolve()),
            "width": 512,
            "height": 512,
            "split": "exploratory_coco100",
        }
        for row in pool["images"]
    ]
    reserved_manifest = reserved / "dataset.json"
    write_json(reserved_manifest, _dataset(all_rows, legacy_hash=legacy_hash))
    test_manifest = base / "manifests" / f"test_{test_count:03d}.json"
    write_json(test_manifest, _dataset(all_rows[:test_count], legacy_hash=legacy_hash))

    available = sorted(set(caption_rows) - reserved_ids)
    if calibration_count > len(available):
        raise ValueError(
            f"Requested {calibration_count} calibration images, only {len(available)} disjoint IDs"
        )
    calibration = base / "calibration"
    calibration_plan = {
        "schema": SCHEMA,
        "reserved_pool_hash": digest(pool),
        "captions_sha256": annotation_hash,
        "selection": "ascending_coco_id_excluding_complete_reserved_coco100",
        "candidate_ids": available,
        "source_url_template": f"{COCO_S3}/val2017/{{coco_id:012d}}.jpg",
        "preprocessing": PREPROCESS,
        "pillow_version": pillow_version,
        "source_authenticity": "official_public_S3_bucket_over_verified_TLS; first_observed_SHA256",
    }
    # Freeze the entire ordered candidate pool before the first network image request.
    _freeze_json(calibration / "plan.json", calibration_plan)
    plan_hash = digest(calibration_plan)
    calibration_rows = []
    for number, image_id in enumerate(available[:calibration_count], start=1):
        name = f"{image_id:012d}"
        target = calibration / "images" / f"{name}.png"
        record_path = calibration / "records" / f"{name}.json"
        url = f"{COCO_S3}/val2017/{name}.jpg"
        if record_path.exists():
            record = read_json(record_path)
            if (
                record.get("plan_hash") != plan_hash
                or record.get("coco_id") != image_id
                or record.get("source_url") != url
                or record.get("filename") != f"{name}.png"
                or record.get("prompt") != caption_rows[image_id]["prompt"]
                or not target.is_file()
                or sha256(target) != record.get("sha256")
            ):
                raise ValueError(f"Corrupt/incompatible completed calibration image: {record_path}")
            _check_fixed_rgb(target)
        else:
            jpeg = calibration / "source_jpeg" / f"{name}.jpg"
            download(url, jpeg)
            _calibration_image(jpeg, target)
            record = {
                "schema": SCHEMA,
                "plan_hash": plan_hash,
                "coco_id": image_id,
                "source_url": url,
                "source_sha256": sha256(jpeg),
                "sha256": sha256(target),
                "filename": f"{name}.png",
                "prompt": caption_rows[image_id]["prompt"],
            }
            write_json(record_path, record)
        calibration_rows.append(
            {
                "image_id": str(image_id),
                "coco_id": image_id,
                "path": str(target.resolve()),
                "sha256": record["sha256"],
                "prompt": record["prompt"],
                "width": 512,
                "height": 512,
                "split": "disjoint_clean_detector_calibration",
            }
        )
        if number == 1 or number % 25 == 0 or number == calibration_count:
            print(f"Calibration input verified: {number}/{calibration_count}", flush=True)
    calibration_manifest = base / "manifests" / f"calibration_{calibration_count:04d}.json"
    write_json(calibration_manifest, _dataset(calibration_rows))
    provenance = {
        "schema": SCHEMA,
        "test_count": test_count,
        "calibration_count": calibration_count,
        "reserved_test_count": 100,
        "legacy_semantic_hash": legacy_hash,
        "reserved_pool_hash": digest(pool),
        "calibration_plan_hash": plan_hash,
        "captions_sha256": annotation_hash,
        "test_ids": [row["coco_id"] for row in all_rows[:test_count]],
        "calibration_ids": available[:calibration_count],
        "test_content_hash": _dataset(all_rows[:test_count])["content_hash"],
        "calibration_content_hash": _dataset(calibration_rows)["content_hash"],
        "calibration_preprocessing": PREPROCESS,
        "calibration_pillow_version": pillow_version,
        "note": "Test PNGs are byte-identical to legacy; calibration images are newly transformed.",
    }
    provenance_path = (
        base / "manifests" / f"provenance_test{test_count}_cal{calibration_count}.json"
    )
    _freeze_json(provenance_path, provenance)
    return {
        "test_manifest": str(test_manifest),
        "calibration_manifest": str(calibration_manifest),
        "reserved_manifest": str(reserved_manifest),
        "provenance_path": str(provenance_path),
        "provenance": provenance,
        "inputs_root": str(base),
    }
