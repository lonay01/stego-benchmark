"""Offline resume/provenance regressions; synthetic adapters are not models."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wmbench.embedding_bank import (
    POLICY,
    _complete,
    freeze_inputs,
    inference_identity,
    process_image,
)
from wmbench.io import digest, read_json, sha256, write_json
from wmbench.maintenance import ARCHIVE_RULES, INVISMARK_CHECKPOINT
from wmbench.matching import load_rgb, save_rgb


WIBE_REVISION = "acf94ed81db4243d22fb7a7e075aff1671bc58e0"
MODEL = {"id": "synthetic_resume_fixture", "native_bits": 32}
PROVENANCE = {"test_only": True, "id": "offline_fixture"}


def model_identity(model_id: str, weights: dict[str, str]) -> dict:
    return {
        "registry": {
            "id": model_id,
            "adapter": "wmbench.adapters.legacy:WibeAdapter",
            "source": {
                "url": "https://github.com/ispras/wibe.git",
                "revision": WIBE_REVISION,
            },
        },
        "weights": dict(weights),
    }


class InferenceIdentityTests(unittest.TestCase):
    def complete_archive_identity(self, model_id="trustmark_q"):
        namespace, filenames = ARCHIVE_RULES[model_id]
        weights = {
            f"model_files/{namespace}/{name}": f"checkpoint_hash_{index}"
            for index, name in enumerate(filenames)
        }
        weights[f"downloads/{namespace}.zip"] = "transport_hash"
        return model_identity(model_id, weights)

    def test_known_complete_transport_archive_is_not_an_inference_input(self):
        for model_id in ARCHIVE_RULES:
            with self.subTest(model_id=model_id):
                original = self.complete_archive_identity(model_id)
                before = copy.deepcopy(original)
                normalized = inference_identity(original)
                namespace, filenames = ARCHIVE_RULES[model_id]
                self.assertNotIn(f"downloads/{namespace}.zip", normalized["weights"])
                for name in filenames:
                    key = f"model_files/{namespace}/{name}"
                    self.assertEqual(normalized["weights"][key], original["weights"][key])
                self.assertEqual(original, before, "Normalization must not erase provenance")

    def test_changed_checkpoint_hash_remains_a_different_identity(self):
        original = self.complete_archive_identity()
        changed = copy.deepcopy(original)
        key = "model_files/trustmark/encoder_Q.ckpt"
        changed["weights"][key] = "different_checkpoint_hash"
        self.assertNotEqual(inference_identity(original), inference_identity(changed))

    def test_repacked_zip_alone_does_not_change_inference_identity(self):
        original = self.complete_archive_identity()
        changed = copy.deepcopy(original)
        changed["weights"]["downloads/trustmark.zip"] = "different_transport_hash"
        self.assertEqual(inference_identity(original), inference_identity(changed))

    def test_incomplete_archive_content_does_not_enable_exemption(self):
        original = self.complete_archive_identity()
        del original["weights"]["model_files/trustmark/decoder_Q.ckpt"]
        self.assertIn("downloads/trustmark.zip", inference_identity(original)["weights"])

    def test_other_source_revision_does_not_enable_archive_exemption(self):
        original = self.complete_archive_identity()
        original["registry"]["source"]["revision"] = "0" * 40
        self.assertIn("downloads/trustmark.zip", inference_identity(original)["weights"])

    def test_unknown_archives_and_event_files_remain_strict(self):
        original = self.complete_archive_identity()
        unknown = {
            "downloads/other.zip": "unknown_archive",
            "other/model.zip": "model_archive",
            "runs2/events.out.tfevents.123.host.456.0": "not_invismark",
            "logs/events.out.tfevents.123.host.456.0": "other_event",
        }
        original["weights"].update(unknown)
        normalized = inference_identity(original)["weights"]
        for key, value in unknown.items():
            self.assertEqual(normalized[key], value)

    def test_only_known_invismark_event_pattern_is_ignored(self):
        known = "runs2/events.out.tfevents.123.host-name.456.0"
        unknown = "runs2/events.out.tfevents.123.host-name.456.0.extra"
        original = model_identity(
            "invismark",
            {
                "model_files/invismark/invismark.ckpt": INVISMARK_CHECKPOINT,
                known: "write_only_event",
                unknown: "unknown_file",
                "runs2/checkpoint.pth": "inference_candidate",
            },
        )
        normalized = inference_identity(original)["weights"]
        self.assertNotIn(known, normalized)
        self.assertEqual(normalized[unknown], "unknown_file")
        self.assertEqual(normalized["runs2/checkpoint.pth"], "inference_candidate")
        self.assertEqual(normalized["model_files/invismark/invismark.ckpt"], INVISMARK_CHECKPOINT)

    def test_other_invismark_checkpoint_or_revision_keeps_event(self):
        event = "runs2/events.out.tfevents.123.host.456.0"
        for change in ("checkpoint", "revision"):
            with self.subTest(change=change):
                original = model_identity(
                    "invismark",
                    {
                        "model_files/invismark/invismark.ckpt": INVISMARK_CHECKPOINT,
                        event: "event_hash",
                    },
                )
                if change == "checkpoint":
                    original["weights"]["model_files/invismark/invismark.ckpt"] = "changed"
                else:
                    original["registry"]["source"]["revision"] = "1" * 40
                self.assertIn(event, inference_identity(original)["weights"])

    def test_windows_asset_separators_normalize_without_changing_hashes(self):
        original = self.complete_archive_identity()
        windows = copy.deepcopy(original)
        windows["weights"] = {
            key.replace("/", "\\"): value for key, value in original["weights"].items()
        }
        self.assertEqual(inference_identity(original), inference_identity(windows))


class FakeAdapter:
    """Exercise persistence; decoding fixture metadata is not real extraction."""

    def __init__(self, fail_decode_once=False, poor_bits=False, interrupt_embed_once=False):
        self.fail_decode_once = fail_decode_once
        self.poor_bits = poor_bits
        self.interrupt_embed_once = interrupt_embed_once
        self.embeddings = 0
        self.decodes = 0
        self.decode_draws = []
        self.embed_hashes = []

    def embed(self, input_path, output_path, payload, metadata_path):
        self.embeddings += 1
        pixels = load_rgb(input_path).astype(np.float64)
        # Global RNG intentionally tests the controller's per-image seeding.
        pixels += np.random.normal(0, 5, pixels.shape)
        save_rgb(output_path, np.rint(np.clip(pixels, 0, 255)).astype(np.uint8))
        write_json(metadata_path, {"test_only": True, "payload": payload})
        self.embed_hashes.append(sha256(output_path))
        if self.interrupt_embed_once and self.embeddings == 1:
            raise RuntimeError("Synthetic interruption before sealing")

    def decode(self, input_path, metadata_path):
        self.decodes += 1
        self.decode_draws.append(float(np.random.random()))
        if self.fail_decode_once and self.decodes == 1:
            raise OSError("Synthetic transient decode failure")
        payload = read_json(metadata_path)["payload"]
        return {"bits": [1 - bit for bit in payload] if self.poor_bits else payload}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wm_bank_resume_fixture_")
        self.bank = Path(self.temporary.name) / "bank"
        self.source = self.bank / "_inputs/images/000000000139.png"
        pixels = np.random.default_rng(505).integers(30, 225, (512, 512, 3), dtype=np.uint8)
        save_rgb(self.source, pixels)
        self.image = {
            "image_id": "139",
            "path": "_inputs/images/000000000139.png",
            "sha256": sha256(self.source),
        }
        self.records = self.bank / MODEL["id"] / "_meta/records"
        self.native_record = self.records / "000000000139_native.json"
        self.native_check = self.records / "000000000139_native_identity.json"

    def tearDown(self):
        self.temporary.cleanup()

    def run_image(self, adapter, *, full_modes=False, identity_check=True):
        policy = POLICY if full_modes else {**POLICY, "targets": []}
        return process_image(
            adapter, MODEL, self.bank, self.image, policy, PROVENANCE, identity_check=identity_check
        )

    def test_complete_rejects_changed_image_record_signature(self):
        self.run_image(FakeAdapter(), full_modes=True)
        saved = read_json(self.native_record)
        saved["signature"] = "wrong_signature"
        write_json(self.native_record, saved)
        with self.assertRaisesRegex(ValueError, "signature"):
            _complete(self.bank, MODEL, [self.image], identity_check=True)

    def test_complete_rejects_changed_identity_check_signature(self):
        self.run_image(FakeAdapter(), full_modes=True)
        saved = read_json(self.native_check)
        saved["signature"] = "wrong_signature"
        write_json(self.native_check, saved)
        with self.assertRaisesRegex(ValueError, "signature"):
            _complete(self.bank, MODEL, [self.image], identity_check=True)

    def test_decode_error_retries_without_reembedding_and_with_same_seed(self):
        adapter = FakeAdapter(fail_decode_once=True)
        self.run_image(adapter)
        self.assertEqual(read_json(self.native_check)["status"], "decode_error")
        native_before = self.native_record.read_bytes()
        self.run_image(adapter)
        self.assertEqual(adapter.embeddings, 1)
        self.assertEqual(adapter.decodes, 2)
        self.assertEqual(adapter.decode_draws[0], adapter.decode_draws[1])
        self.assertEqual(self.native_record.read_bytes(), native_before)
        self.assertEqual(read_json(self.native_check)["status"], "measured")
        archived = list((self.bank / MODEL["id"] / "_meta/decode_errors").glob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(read_json(archived[0])["status"], "decode_error")

    def test_measured_poor_ber_is_not_retried(self):
        adapter = FakeAdapter(poor_bits=True)
        self.run_image(adapter)
        before = self.native_check.read_bytes()
        self.assertEqual(read_json(self.native_check)["common30_ber"], 1.0)
        self.run_image(adapter)
        self.assertEqual(adapter.embeddings, 1)
        self.assertEqual(adapter.decodes, 1)
        self.assertEqual(self.native_check.read_bytes(), before)

    def test_unsealed_native_retry_preserves_seed_and_archives_partial_keys(self):
        adapter = FakeAdapter(interrupt_embed_once=True)
        with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
            self.run_image(adapter, identity_check=False)
        self.assertFalse(self.native_record.exists())
        self.run_image(adapter, identity_check=False)
        self.assertEqual(adapter.embeddings, 2)
        self.assertEqual(adapter.embed_hashes[0], adapter.embed_hashes[1])
        unsealed = self.bank / MODEL["id"] / "_meta/unsealed"
        self.assertEqual(len(list(unsealed.glob("*.png"))), 1)
        self.assertEqual(len(list(unsealed.glob("*_keys/metadata.json"))), 1)

    def test_frozen_manifest_content_hash_corruption_is_rejected(self):
        rows = [self.image]
        csv_path = self.bank / "_inputs/legacy_manifest.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("image_id,sha256_png\n139," + self.image["sha256"] + "\n")
        manifest = {
            "images": rows,
            "legacy_semantic_hash": "synthetic_NOT_COCO100",
            "content_hash": digest([{"coco_id": 139, "sha256": self.image["sha256"]}]),
            "manifest_sha256": sha256(csv_path),
        }
        path = self.bank / "_inputs/manifest.json"
        write_json(path, manifest)
        self.assertEqual(
            freeze_inputs(Path("not_needed"), self.bank, require_historical=False), manifest
        )
        manifest["content_hash"] = "corrupt_hash"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "manifest content/order"):
            freeze_inputs(Path("not_needed"), self.bank, require_historical=False)


if __name__ == "__main__":
    unittest.main()
