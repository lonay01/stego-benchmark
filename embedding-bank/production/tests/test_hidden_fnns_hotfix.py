"""Repair tests against the unchanged production ZIP; no cloud mutation."""

import copy
import importlib.util
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from wmbench.io import read_json, write_json, sha256
from wmbench.matching import save_rgb, load_rgb
from wmbench_full_payload.storage import EvidenceError, freeze
from wmbench_bank.contracts import models_for_run, MODES
from wmbench_bank import worker, runner
from wmbench import colab_canary
from wmbench.adapters import meta_hidden

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scoped_repair", ROOT / "hotfix_hidden_fnns.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
PROJECT = Path(os.environ["WMBENCH_TEST_RELEASE"])


class ToyFNNS:
    """Storage fixture only, not an FNNS accuracy claim."""
    def __init__(self):
        self.embeds = 0
        self.decodes = 0

    def embed(self, source, png, payload, metadata):
        self.embeds += 1
        image = load_rgb(source).astype(int)
        residual = np.arange(image.size).reshape(image.shape) % 19 - 9
        save_rgb(png, np.clip(image + residual, 0, 255).astype(np.uint8))
        write_json(metadata, {"expected_bits": payload, "fixture": True})

    def decode(self, source, metadata):
        self.decodes += 1
        assert not any(read_json(metadata)["expected_bits"])
        return {"bits": [0] * 48}  # Deliberately nonzero BER; never a retry reason.


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = models_for_run(PROJECT)[fix.FNNS]
        path = self.root / "_inputs/images/000000000139.png"
        save_rgb(path, np.random.default_rng(12).integers(20, 230, (512, 512, 3), dtype=np.uint8))
        self.image = {"image_id": 139, "path": path.relative_to(self.root).as_posix(), "sha256": sha256(path)}
        self.adapter = ToyFNNS()
        self.stem = "000000000139_m01"

    def tearDown(self):
        self.temp.cleanup()

    def native(self):
        return self.root / fix.FNNS / "native" / f"{self.stem}.png"

    def prepare_pair(self):
        worker.encode_pair(self.adapter, self.model, self.root, self.image, 1, {})
        worker.decode_pair(self.adapter, self.model, self.root, self.image, 1, {})

    def identity(self):
        return {"registry": {"id": fix.META}, "weights": {**meta_hidden.ASSET_HASHES, fix.PYC: "old-cache"}}

    def test_meta_cache_only_change_preserves_original_identity(self):
        old = self.identity()
        path = self.root / fix.META / "_meta/model_identity.json"
        freeze(path, old)
        original = path.read_bytes()
        current = copy.deepcopy(old)
        current["weights"][fix.PYC] = "new-cache"
        fix.freeze_meta(path, current, freeze)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(list((path.parent / "repairs").glob("*.json"))), 1)

    def test_meta_does_not_need_pyc(self):
        old = self.identity()
        current = copy.deepcopy(old)
        del current["weights"][fix.PYC]
        self.assertEqual(fix.canonical_meta(current), fix.canonical_meta(old))

    def test_meta_source_weights_and_unknown_files_remain_strict(self):
        for name in [*meta_hidden.ASSET_HASHES, "hidden/__pycache__/another.pyc"]:
            current = self.identity()
            current["weights"][name] = "changed"
            with self.subTest(name=name), self.assertRaises(EvidenceError):
                fix.canonical_meta(current)

    def test_meta_registry_and_dependencies_remain_strict(self):
        path = self.root / fix.META / "_meta/model_identity.json"
        old = self.identity()
        freeze(path, old)
        current = copy.deepcopy(old)
        current["registry"]["strength"] = 2
        with self.assertRaises(EvidenceError):
            fix.freeze_meta(path, current, freeze)
        dependency = path.with_name("dependency_identity.json")
        freeze(dependency, {"lock": "old"})
        with self.assertRaises(EvidenceError):
            fix.freeze_meta(dependency, {"lock": "changed"}, freeze)

    def test_other_method_identity_has_no_cache_exception(self):
        path = self.root / "rosteals/_meta/model_identity.json"
        freeze(path, {"weights": {"x.pyc": "old"}})
        with self.assertRaises(EvidenceError):
            fix.freeze_meta(path, {"weights": {"x.pyc": "new"}}, freeze)

    @unittest.skipUnless(Path(r"D:\GIT\tmp\meta-hidden-reference/hidden_replicate.pth").is_file(),
                         "Local official Meta weights unavailable")
    def test_real_meta_source_loader_preserves_png_and_logits(self):
        import torch
        from PIL import Image
        torch.set_num_threads(4)
        fixture = Path(r"D:\GIT\tmp\meta-hidden-reference")
        assets = self.root / "assets"
        for relative in meta_hidden.ASSET_HASHES:
            destination = assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture / Path(relative).name, destination)
        source = self.root / "meta_input.png"
        with Image.open(fixture / "demo.png") as file:
            save_rgb(source, np.array(file.convert("RGB").resize((256, 256))))
        options = {"asset_root": str(assets), "device": "cpu"}
        before = meta_hidden.MetaHiddenAdapter(options)
        with patch.object(meta_hidden, "load_classes", fix.meta_source_classes):
            after = meta_hidden.MetaHiddenAdapter(options)
        payload = np.random.default_rng(3).integers(0, 2, 48).tolist()
        first, second = self.root / "before.png", self.root / "after.png"
        metadata = self.root / "meta_message.json"
        before.embed(str(source), str(first), payload, str(metadata))
        after.embed(str(source), str(second), payload, str(metadata))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(before.decode(str(first), str(metadata)), after.decode(str(second), str(metadata)))

    def test_intact_fnns_nonzero_ber_is_never_retried(self):
        self.prepare_pair()
        snapshot = {str(p): sha256(p) for p in (self.root / fix.FNNS).rglob("*") if p.is_file()}
        self.assertIsNone(fix.repair_pair(self.root, self.model, self.image, 1))
        worker.encode_pair(self.adapter, self.model, self.root, self.image, 1, {})
        self.assertEqual(self.adapter.embeds, 1)
        self.assertTrue(all(sha256(p) == checksum for p, checksum in snapshot.items()))

    def test_empty_native_quarantines_only_its_pair_and_modes(self):
        self.prepare_pair()
        worker.encode_pair(self.adapter, self.model, self.root, self.image, 2, {})
        other = {str(p): sha256(p) for p in (self.root / fix.FNNS).rglob("*")
                 if p.is_file() and "m02" in str(p)}
        self.native().write_bytes(b"")
        folder = fix.repair_pair(self.root, self.model, self.image, 1)
        self.assertTrue((folder / "retired.json").exists())
        self.assertEqual((folder / "original" / self.native().relative_to(self.root)).stat().st_size, 0)
        self.assertFalse(self.native().exists())
        for mode in MODES:
            self.assertFalse((self.root / fix.FNNS / "_meta/records" / f"{self.stem}_{mode}.json").exists())
        self.assertTrue(all(sha256(p) == value for p, value in other.items()))
        worker.encode_pair(self.adapter, self.model, self.root, self.image, 1, {})
        worker.decode_pair(self.adapter, self.model, self.root, self.image, 1, {})
        self.assertEqual(fix.inspect_pair(self.root, self.model, self.image, 1)[2], {})

    def test_missing_carrier_can_be_retired(self):
        self.prepare_pair()
        self.native().unlink()
        self.assertIsNotNone(fix.repair_pair(self.root, self.model, self.image, 1))

    def test_exact_backup_restores_bytes_without_touching_records(self):
        self.prepare_pair()
        original = self.native().read_bytes()
        checksum = sha256(self.native())
        fix.verified_copy(self.native(), fix.backup_path(self.root, checksum), checksum)
        record = self.root / fix.FNNS / "_meta/records" / f"{self.stem}_native.json"
        original_record = record.read_bytes()
        self.native().write_bytes(b"")
        fix.repair_pair(self.root, self.model, self.image, 1)
        self.assertEqual(self.native().read_bytes(), original)
        self.assertEqual(record.read_bytes(), original_record)
        self.assertEqual(self.adapter.embeds, 1)

    def test_noncarrier_corruption_stops_without_retiring_png(self):
        self.prepare_pair()
        row = read_json(self.root / fix.FNNS / "_meta/records" / f"{self.stem}_native.json")
        (self.root / row["decoder_metadata"]).write_text("{}")
        with self.assertRaises(EvidenceError):
            fix.repair_pair(self.root, self.model, self.image, 1)
        self.assertTrue(self.native().is_file())

    def test_payload_change_stops_even_if_png_is_empty(self):
        self.prepare_pair()
        record = self.root / fix.FNNS / "_meta/records" / f"{self.stem}_native.json"
        row = read_json(record)
        row["assignment"]["payload"][-1] ^= 1
        write_json(record, row)
        self.native().write_bytes(b"")
        with self.assertRaises(EvidenceError):
            fix.repair_pair(self.root, self.model, self.image, 1)

    def test_interrupted_quarantine_resumes(self):
        self.prepare_pair()
        self.native().write_bytes(b"")
        original_unlink = Path.unlink
        removed = []

        def interrupted(path, *args, **kwargs):
            if path.parent.name == "records":
                if removed:
                    raise OSError("synthetic disconnect")
                removed.append(str(path))
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", interrupted), self.assertRaises(OSError):
            fix.repair_pair(self.root, self.model, self.image, 1)
        fix.repair_pair(self.root, self.model, self.image, 1)
        plans = list((self.root / fix.FNNS / "_meta/repairs").glob("*/*/plan.json"))
        self.assertEqual(len(plans), 1)
        self.assertTrue((plans[0].parent / "retired.json").exists())

    def test_invalid_backup_prevents_any_retirement(self):
        self.prepare_pair()
        self.native().write_bytes(b"")
        with patch.object(fix, "finish_retirement"):
            folder = fix.repair_pair(self.root, self.model, self.image, 1)
        entry = read_json(folder / "plan.json")["files"][0]
        (folder / "original" / entry["path"]).write_bytes(b"broken archive")
        with self.assertRaises(EvidenceError):
            fix.finish_retirement(self.root, folder)
        self.assertTrue(self.native().exists())

    def test_unsafe_plan_path_is_rejected(self):
        folder = self.root / "event"
        write_json(folder / "plan.json", {"stem": self.stem,
            "files": [{"path": "../outside", "sha256": "a" * 64}]})
        with self.assertRaises(EvidenceError):
            fix.finish_retirement(self.root, folder)

    def test_recovery_cannot_target_other_method(self):
        with self.assertRaises(EvidenceError):
            fix.repair_pair(self.root, {"id": "rosteals"}, self.image, 1)

    def test_installed_overlay_cpu_smoke_and_process_routing(self):
        saved = {"freeze": worker.freeze, "encode_pair": worker.encode_pair, "sealed": worker.sealed, "work": worker.work,
                 "complete": runner.complete, "monitor": colab_canary.monitored_process,
                 "classes": meta_hidden.load_classes}
        calls = []
        try:
            colab_canary.monitored_process = lambda command, *a, **kw: calls.append(command)
            fix.install()
            self.prepare_pair()
            before = self.adapter.embeds
            self.native().write_bytes(b"")
            self.assertFalse(runner.complete(self.root, self.model, [self.image]))
            # New files already have backups: exact restore, no extra embedding.
            worker.encode_pair(self.adapter, self.model, self.root, self.image, 1, {})
            self.assertEqual(self.adapter.embeds, before)
            for method in (fix.META, fix.FNNS, "rosteals", "sepwater", "trustmark_q"):
                path = self.root / f"{method}.json"
                write_json(path, {"model": {"id": method}})
                command = ["controller", "-m", "wmbench.process_guard", "--", "worker", "-m",
                           "wmbench_bank.runner", "work", "--settings", str(path)]
                colab_canary.monitored_process(command)
                if method in fix.TARGETS:
                    self.assertIn(str(Path(fix.__file__).resolve()), calls[-1])
                else:
                    self.assertEqual(command, calls[-1])
        finally:
            for name in ("freeze", "encode_pair", "sealed", "work"):
                setattr(worker, name, saved[name])
            runner.complete = saved["complete"]
            colab_canary.monitored_process = saved["monitor"]
            meta_hidden.load_classes = saved["classes"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
