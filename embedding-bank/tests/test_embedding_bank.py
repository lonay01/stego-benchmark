"""Offline contracts for the persistent embedding bank; no model downloads.

The fake adapter checks orchestration and serialization, not watermark quality.
Synthetic RGB images are explicitly admitted only with require_historical=False.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.embedding_bank import (
    common_message,
    freeze_inputs,
    message_for_model,
    process_image,
    valid_record,
)
from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, psnr, save_rgb


POLICY = {
    "seed": 20260811,
    "targets": [39, 42, 45],
    "tolerance": 0.05,
    "max_alpha": 4,
    "iterations": 64,
}
MODEL = {"id": "fake_test_only", "native_bits": 32}
PROVENANCE = {"test_only": True, "adapter": "deterministic fake; not a research method"}
MANIFEST_FIELDS = (
    "position",
    "image_id",
    "filename",
    "source_file",
    "source_url",
    "sha256_png",
)


class FakeAdapter:
    """Save a deterministic residual and persist the supplied message for a mock decode."""

    def __init__(self, unchanged: bool = False, require_grad: bool = False):
        self.unchanged = unchanged
        self.require_grad = require_grad
        self.embed_calls = 0
        self.decode_calls = 0

    def embed(self, input_path, output_path, payload, metadata_path):
        self.embed_calls += 1
        if self.require_grad:
            import torch

            if not torch.is_grad_enabled():
                raise AssertionError("The controller disabled image optimization gradients")
        source = load_rgb(input_path)
        noise = np.random.default_rng(104).normal(0, 5, source.shape)
        output = (
            source
            if self.unchanged
            else np.rint(np.clip(source.astype(np.float64) + noise, 0, 255)).astype(np.uint8)
        )
        save_rgb(output_path, output)
        write_json(metadata_path, {"expected_bits": list(payload), "test_only": True})

    def decode(self, input_path, metadata_path):
        self.decode_calls += 1
        # This is a mock of the IO contract, not a watermark extractor. A real
        # adapter must decode the image without using these reference bits.
        payload = read_json(metadata_path)["expected_bits"]
        return {"bits": list(payload), "test_only": True}


class MessageTests(unittest.TestCase):
    def test_one_balanced_message_is_stable_per_image(self):
        messages = [common_message(str(index)) for index in range(1, 101)]
        for index, message in enumerate(messages, 1):
            self.assertEqual(len(message), 30)
            self.assertEqual(sum(message), 15)
            self.assertTrue(all(type(bit) is int and bit in (0, 1) for bit in message))
            self.assertEqual(message, common_message(str(index)))
        self.assertGreater(len({tuple(bits) for bits in messages}), 90)
        self.assertNotEqual(common_message("139"), common_message("139", seed=1))

    def test_common_segment_is_identical_across_models(self):
        expected = common_message("139")
        for size in (30, 32, 64, 100, 128, 256, 1024):
            model = {"id": f"fake_{size}", "native_bits": size}
            payload, indices = message_for_model(model, "139")
            self.assertEqual(len(payload), size)
            self.assertEqual(indices, list(range(size - 30, size)))
            self.assertEqual([payload[index] for index in indices], expected)
            self.assertEqual((payload, indices), message_for_model(model, "139"))

    def test_reserved_bits_remain_fixed_without_changing_message(self):
        reserved = {"48": 0, "49": 1, "50": 0, "51": 0, "64": 1, "65": 0}
        payload, indices = message_for_model(
            {"id": "invismark", "native_bits": 100, "reserved_bits": reserved}, "139"
        )
        self.assertEqual([payload[index] for index in indices], common_message("139"))
        for position, value in reserved.items():
            self.assertEqual(payload[int(position)], value)

    def test_reserved_bit_may_not_overwrite_common_message(self):
        model = {"id": "bad", "native_bits": 32, "reserved_bits": {"2": 0}}
        with self.assertRaises(ValueError):
            message_for_model(model, "139")

    def test_short_or_detection_only_payloads_are_rejected(self):
        for length in (0, 10, 24, 29):
            with self.subTest(length=length), self.assertRaises(ValueError):
                message_for_model({"id": "ineligible", "native_bits": length}, "139")


class InputFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wm_bank_inputs_test_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic_coco_layout"
        self.bank = self.root / "bank"
        (self.source / "test").mkdir(parents=True)
        rows = []
        for image_id in range(1, 101):
            path = self.source / "test" / f"{image_id:012d}.png"
            Image.new("RGB", (512, 512), (image_id, 120, 150)).save(path)
            rows.append(
                {
                    "position": image_id - 1,
                    "image_id": image_id,
                    "filename": path.name,
                    "source_file": f"{image_id:012d}.jpg",
                    "source_url": f"https://test.invalid/synthetic/{image_id}.jpg",
                    "sha256_png": sha256(path),
                }
            )
        with (self.source / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def test_synthetic_fixture_is_not_accepted_as_historical_coco(self):
        with self.assertRaises(ValueError):
            freeze_inputs(self.source, self.bank, require_historical=True)

    def test_explicit_test_fixture_freezes_identical_bytes(self):
        manifest = freeze_inputs(self.source, self.bank, require_historical=False)
        self.assertEqual(len(manifest["images"]), 100)
        for image in manifest["images"]:
            relative = Path(image["path"])
            self.assertFalse(relative.is_absolute())
            self.assertTrue((self.bank / relative).resolve().is_relative_to(self.bank.resolve()))
            self.assertEqual(sha256(self.bank / relative), image["sha256"])
            original = self.source / "test" / f"{int(image['image_id']):012d}.png"
            self.assertEqual(sha256(original), image["sha256"])

    def test_changed_source_bytes_are_rejected(self):
        Image.new("RGB", (512, 512), "red").save(self.source / "test/000000000001.png")
        with self.assertRaises(ValueError):
            freeze_inputs(self.source, self.bank, require_historical=False)

    def test_missing_source_image_is_not_silently_skipped(self):
        (self.source / "test/000000000100.png").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            freeze_inputs(self.source, self.bank, require_historical=False)


class BankExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wm_bank_execution_test_")
        self.bank = Path(self.temporary.name) / "bank"
        source = self.bank / "_originals/000000000139.png"
        source.parent.mkdir(parents=True)
        rng = np.random.default_rng(501)
        save_rgb(source, rng.integers(35, 220, (512, 512, 3), dtype=np.uint8))
        self.image = {
            "image_id": "139",
            "path": source.relative_to(self.bank).as_posix(),
            "sha256": sha256(source),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_image(self, adapter, **kwargs):
        return process_image(
            adapter,
            MODEL,
            self.bank,
            self.image,
            POLICY,
            PROVENANCE,
            **kwargs,
        )

    def record_paths(self):
        return sorted(
            path
            for path in (self.bank / MODEL["id"] / "_meta/records").glob("*.json")
            if not path.stem.endswith("_identity")
        )

    def test_single_embedding_produces_four_mode_records_and_resumes(self):
        adapter = FakeAdapter()
        results = self.run_image(adapter)
        self.assertEqual(len(results), 4)
        self.assertEqual(adapter.embed_calls, 1)
        native = self.bank / MODEL["id"] / "native/000000000139.png"
        self.assertTrue(native.is_file())
        records = self.record_paths()
        self.assertEqual(len(records), 4)
        for record_path in records:
            self.assertIsNotNone(valid_record(self.bank, record_path))
        before = {path.relative_to(self.bank): path.read_bytes() for path in records}
        png_hash = sha256(native)
        calls = adapter.decode_calls
        self.run_image(adapter)
        self.assertEqual(adapter.embed_calls, 1, "Valid Native must not be embedded again")
        self.assertEqual(adapter.decode_calls, calls, "Valid identity checks are reusable")
        self.assertEqual(sha256(native), png_hash)
        for relative, contents in before.items():
            self.assertEqual((self.bank / relative).read_bytes(), contents)

    def test_saved_matched_pixels_are_exactly_scalar_residual(self):
        results = self.run_image(FakeAdapter())
        source = load_rgb(self.bank / self.image["path"])
        native = load_rgb(self.bank / MODEL["id"] / "native/000000000139.png")
        for record in results:
            if record["mode"] == "native":
                continue
            self.assertEqual(record["status"], "ok")
            output = load_rgb(self.bank / record["output"])
            expected = np.rint(
                np.clip(
                    source.astype(np.float64)
                    + record["alpha"] * (native.astype(np.float64) - source),
                    0,
                    255,
                )
            ).astype(np.uint8)
            np.testing.assert_array_equal(output, expected)
            self.assertLessEqual(abs(psnr(source, output) - record["target"]), POLICY["tolerance"])

    def test_corrupted_native_raises_instead_of_overwriting(self):
        adapter = FakeAdapter()
        self.run_image(adapter)
        native = self.bank / MODEL["id"] / "native/000000000139.png"
        native.write_bytes(b"intentional corrupted fixture")
        native_record = next(path for path in self.record_paths() if path.stem.endswith("_native"))
        with self.assertRaises(ValueError):
            valid_record(self.bank, native_record)
        with self.assertRaises(ValueError):
            self.run_image(adapter)
        self.assertEqual(native.read_bytes(), b"intentional corrupted fixture")
        self.assertEqual(adapter.embed_calls, 1)

    def test_no_residual_has_diagnostic_records_but_no_matched_pngs(self):
        adapter = FakeAdapter(unchanged=True)
        records = self.run_image(adapter)
        self.assertEqual(len(records), 4)
        self.assertTrue((self.bank / MODEL["id"] / "native/000000000139.png").is_file())
        for target in (39, 42, 45):
            folder = self.bank / MODEL["id"] / f"matched_{target}"
            self.assertEqual(list(folder.glob("*.png")), [])
        self.assertEqual(len(self.record_paths()), 4)
        for path in self.record_paths():
            self.assertIsNotNone(valid_record(self.bank, path))
        self.run_image(adapter)
        self.assertEqual(adapter.embed_calls, 1)

    def test_portable_records_validate_after_copy_to_another_root(self):
        self.run_image(FakeAdapter())
        copied = self.bank.parent / "other_ssd_bank"
        shutil.copytree(self.bank, copied)
        for record in self.record_paths():
            self.assertIsNotNone(valid_record(copied, copied / record.relative_to(self.bank)))

    def test_absent_record_is_not_a_completed_result(self):
        self.assertIsNone(valid_record(self.bank, self.bank / "no_record.json"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional local torch is unavailable")
    def test_optimization_adapter_keeps_gradients_enabled(self):
        adapter = FakeAdapter(require_grad=True)
        self.run_image(adapter)
        self.assertEqual(adapter.embed_calls, 1)


if __name__ == "__main__":
    unittest.main()
