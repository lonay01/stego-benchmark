"""Replacement identity, carrier geometry, blind extraction and real CPU smoke."""

import ast
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from wmbench.adapters.meta_hidden import ASSET_HASHES, MetaHiddenAdapter, verify_assets
from wmbench.embedding_bank import bank_registry, model_carrier, process_image
from wmbench.io import read_json, sha256
from wmbench.matching import load_rgb, save_rgb, psnr

PROJECT = Path(__file__).resolve().parents[1]


class MetaContractTests(unittest.TestCase):
    def test_replaces_one_entry_without_relabelling_old_results(self):
        models = bank_registry(PROJECT)
        self.assertEqual(len(models), 32)
        self.assertNotIn("hidden_benchmark", models)
        model = models["meta_hidden_revisited"]
        self.assertEqual(model["native_bits"], 48)
        self.assertEqual(model["native_size"], 256)
        self.assertEqual(model["options"], {"profile": "checkpoint_256"})
        self.assertEqual({a["path"]: a["sha256"] for a in model["assets"]}, ASSET_HASHES)
        for asset in model["assets"]:
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertIn(model["upstream"]["revision"], asset["url"])

    def test_missing_or_changed_asset_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_assets(Path(directory))

    def test_blind_decoder_does_not_read_expected_message(self):
        source = (PROJECT / "src/wmbench/adapters/meta_hidden.py").read_text("utf-8")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        decode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == "metadata_path"
                             for n in ast.walk(decode)))

    def test_central_crop_is_literal_persistent_and_corruption_checked(self):
        model = bank_registry(PROJECT)["meta_hidden_revisited"]
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory)
            source = bank / "_inputs/000000000139.png"
            image = np.random.default_rng(5).integers(0, 256, (512, 512, 3), dtype=np.uint8)
            save_rgb(source, image)
            original = sha256(source)
            crop = model_carrier(bank, model, source)
            np.testing.assert_array_equal(load_rgb(crop), image[128:384, 128:384])
            self.assertEqual(model_carrier(bank, model, source), crop)
            self.assertEqual(sha256(source), original)
            save_rgb(crop, np.zeros((256, 256, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "carrier differs"):
                model_carrier(bank, model, source)

    def test_historical_512_methods_keep_original_carrier(self):
        source = Path("unused.png")
        self.assertEqual(model_carrier(Path("bank"), {"id": "other"}, source), source)


class MetaPretrainedTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("WMBENCH_META_FIXTURES"), "Set official local fixtures")
    def test_real_cpu_embed_png_decode_and_legacy_controller(self):
        import torch

        torch.set_num_threads(4)
        fixture = Path(os.environ["WMBENCH_META_FIXTURES"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            for relative in ASSET_HASHES:
                target = assets / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fixture / Path(relative).name, target)
            adapter = MetaHiddenAdapter({"asset_root": str(assets), "device": "cpu"})
            with Image.open(fixture / "demo.png") as image:
                image = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
                source = np.array(image.crop((128, 128, 384, 384)))
            carrier, output, metadata = root / "carrier.png", root / "wm.png", root / "meta.json"
            save_rgb(carrier, source)
            payload = [int(b) for b in "111010110101000001010111010011010100010000100111"]
            adapter.embed(str(carrier), str(output), payload, str(metadata))
            decoded = adapter.decode(str(output), str(root / "nonexistent.json"))
            self.assertEqual(len(decoded["bits"]), 48)
            self.assertTrue(np.isfinite(decoded["logits"]).all())
            self.assertEqual(load_rgb(output).shape, (256, 256, 3))
            self.assertEqual(read_json(metadata)["expected_bits"], payload)
            # Test consistency with direct execution of the loaded author neural core.
            with torch.inference_mode():
                x = adapter.normalize(adapter.to_tensor(Image.fromarray(source))).unsqueeze(0)
                bits = 2 * torch.tensor([payload], dtype=torch.float32) - 1
                rgb = adapter.unnormalize(adapter.encoder(x, bits))
                direct = np.array(adapter.to_pil((torch.round(rgb.clamp(0, 1)*255)/255)[0]))
            np.testing.assert_array_equal(load_rgb(output), direct)
            print("META_CPU_SMOKE", {"bit_errors": sum(a != b for a, b in zip(payload, decoded["bits"])),
                                     "bits": 48, "psnr": psnr(source, direct)})
            # Recovery quality is measured, never forced to be zero BER.
            with self.assertRaisesRegex(ValueError, "exactly 48"):
                adapter.embed(str(carrier), str(output), payload[:30], str(metadata))
            model = bank_registry(PROJECT)["meta_hidden_revisited"]
            bank = root / "bank"
            original = bank / "_inputs/139.png"
            save_rgb(original, np.array(image))
            entry = {"image_id": "139", "path": "_inputs/139.png", "sha256": sha256(original)}
            policy = {"seed": 20260811, "targets": [42], "tolerance": .05,
                      "max_alpha": 4, "iterations": 64}
            rows = process_image(adapter, model, bank, entry, policy, {"test": "cpu"})
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["identity_check"]["status"], "measured")
                self.assertEqual(load_rgb(bank / row["output"]).shape, (256, 256, 3))
                self.assertAlmostEqual(row["psnr"], psnr(load_rgb(bank / row["carrier"]),
                                                       load_rgb(bank / row["output"])))
            frozen = output.read_bytes()
            again = process_image(adapter, model, bank, entry, policy, {"test": "cpu"})
            self.assertEqual([r["artifacts"] for r in rows], [r["artifacts"] for r in again])
            self.assertEqual(output.read_bytes(), frozen)
            changed = assets / "hidden/models.py"
            changed.write_bytes(changed.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_assets(assets)


if __name__ == "__main__":
    unittest.main()
