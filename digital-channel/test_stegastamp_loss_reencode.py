"""CPU transaction tests; no claims of GPU/model validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent


class ReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(ROOT / "dist/COCO_Digital_Channel_RECOVERY_FAST_RESUME_source.zip") as z:
            z.extractall(cls.code.name)
        sys.path[:0] = [str(Path(cls.code.name) / "src"), str(Path(cls.code.name) / "production/src"), str(ROOT)]
        spec = importlib.util.spec_from_file_location("replacement_test_module", ROOT / "stegastamp_loss_reencode.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.code.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bank, self.work, self.results = [self.root / n for n in ("bank", "recovery", "results")]
        self.stage = self.work / "replacement_bank"
        self.old, self.new = {}, {}
        for stem in self.module.TARGETS:
            name = f"stegastamp/native/{stem}.png"
            self.put(self.bank / name, b"")
            self.put(self.stage / name, b"new-" + stem.encode())
            self.old[name] = self.module.sha256(self.bank / name)
            self.new[name] = self.module.sha256(self.stage / name)
        self.put(self.bank / "sepmark/native/untouched.png", b"other")
        self.put(self.results / "stegastamp/cases.json", b"old-catalogue")
        self.put(self.results / "stegastamp/records/000000000285_m01_native.json.gz", b"unrelated")
        self.plan = dict(revision=self.module.REVISION, bank=str(self.bank), stage=str(self.stage),
                         old_files=self.old, new_files=self.new, execution={"recovery": True})
        self.save_plan()

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_plan(self):
        self.module.write_json(self.work / "replacement_manifest.json", self.plan)

    def test_commit_and_idempotence(self):
        self.module.commit(self.work, self.results)
        for name, digest in self.new.items():
            self.assertEqual(self.module.sha256(self.bank / name), digest)
            self.assertEqual((self.work / "previous_bank" / name).read_bytes(), b"")
        self.assertEqual((self.bank / "sepmark/native/untouched.png").read_bytes(), b"other")
        self.assertFalse((self.results / "stegastamp/cases.json").exists())
        self.assertEqual((self.work / "previous_digital/stegastamp/cases.json").read_bytes(), b"old-catalogue")
        self.assertEqual((self.results / "stegastamp/records/000000000285_m01_native.json.gz").read_bytes(), b"unrelated")
        self.module.commit(self.work, self.results)

    def test_staged_tamper_leaves_bank_empty(self):
        name = next(iter(self.new))
        self.put(self.stage / name, b"tampered")
        with self.assertRaises(RuntimeError):
            self.module.commit(self.work, self.results)
        self.assertTrue(all((self.bank / n).stat().st_size == 0 for n in self.old))

    def test_original_change_refused(self):
        name = next(iter(self.old))
        self.put(self.bank / name, b"external-change")
        with self.assertRaises(RuntimeError):
            self.module.commit(self.work, self.results)

    def test_other_method_cannot_enter_plan(self):
        name = "sepmark/native/untouched.png"
        self.put(self.stage / name, b"replacement")
        self.plan["new_files"][name] = self.module.sha256(self.stage / name)
        self.save_plan()
        with self.assertRaises(RuntimeError):
            self.module.commit(self.work, self.results)
        self.assertEqual((self.bank / name).read_bytes(), b"other")

    def test_partial_commit_resumes_from_verified_backup(self):
        for name in self.old:
            self.module.copy_checked(self.bank / name, self.work / "previous_bank" / name)
        self.module.write_json(self.work / "commit_journal.json", dict(revision=self.module.REVISION, digital={}, state="backed_up"))
        name = next(iter(self.new))
        self.put(self.bank / name, (self.stage / name).read_bytes())
        self.module.commit(self.work, self.results)
        self.assertTrue((self.work / "replacement_receipt.json").is_file())

    def test_missing_backup_prevents_commit(self):
        self.module.write_json(self.work / "commit_journal.json", dict(revision=self.module.REVISION, digital={}, state="backed_up"))
        with self.assertRaises(RuntimeError):
            self.module.commit(self.work, self.results)
        self.assertTrue(all((self.bank / n).stat().st_size == 0 for n in self.old))


if __name__ == "__main__":
    unittest.main()
