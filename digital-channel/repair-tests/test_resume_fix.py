"""Lost-Colab-cache regression tests; no real model results are modified."""

import ast
import contextlib
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import nbformat
from wmbench.io import write_json, read_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_digital import storage, runner
from wmbench_digital.spec import ATTACKS
import test_digital as baseline
import digital_resume_hotfix as fix

ROOT = Path(fix.__file__).parent


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.fixture = baseline.Pipeline()
        self.fixture.setUp()
        self.old_storage, self.old_runner = storage.remove_packages, runner.remove_packages
        fix.install_cleanup_patch()
        self.manifest = self.fixture.packages()
        self.archive = Path(self.manifest["archive"])

    def tearDown(self):
        storage.remove_packages, runner.remove_packages = self.old_storage, self.old_runner
        self.fixture.tearDown()

    def commit(self):
        f = self.fixture
        rows = {a["id"]: dict(status="measured", ber=0., exact=True, seconds=.1, decoded_bits=[0, 1]) for a in ATTACKS}
        storage.save_result(storage.result_path(f.root, f.case), dict(case_key=f.case["key"], case=f.case, rows=rows))

    def lose_cache(self):
        # Only these exact files in the test-created cache are removed.
        self.assertTrue(self.archive.resolve().is_relative_to(self.fixture.scratch.resolve()))
        self.archive.unlink(missing_ok=True)
        (self.archive.parent/"owner.json").unlink()
        self.archive.parent.rmdir()

    def cleanup(self, manifests=None):
        f = self.fixture
        storage.remove_packages(f.root, [f.case], [self.manifest] if manifests is None else manifests,
                                [f.scratch, f.root/"_temporary"], "identity", f.method)

    def test_missing_cache_is_idempotent_and_results_unchanged(self):
        self.commit()
        self.lose_cache()
        f = self.fixture
        result = storage.result_path(f.root, f.case)
        before = (sha256(result), sha256(f.bank/"carrier.png"))
        self.cleanup()
        self.cleanup()
        self.assertEqual(before, (sha256(result), sha256(f.bank/"carrier.png")))
        self.assertFalse(self.archive.parent.exists())
        self.assertTrue(list((f.root/f.method/"commits").glob("*.json")))

    def test_incomplete_results_still_block_even_without_cache(self):
        self.lose_cache()
        with self.assertRaisesRegex(EvidenceError, "durable"):
            self.cleanup()

    def test_present_zip_requires_owner(self):
        self.commit()
        (self.archive.parent/"owner.json").unlink()
        before = sha256(self.archive)
        with self.assertRaises(FileNotFoundError):
            self.cleanup()
        self.assertEqual(before, sha256(self.archive))

    def test_present_zip_requires_correct_owner(self):
        self.commit()
        write_json(self.archive.parent/"owner.json", {"identity": "someone-else", "method": self.fixture.method})
        with self.assertRaisesRegex(EvidenceError, "owner"):
            self.cleanup()
        self.assertTrue(self.archive.exists())

    def test_present_zip_requires_original_hash(self):
        self.commit()
        self.archive.write_bytes(b"altered disposable test archive")
        with self.assertRaisesRegex(EvidenceError, "changed"):
            self.cleanup()
        self.assertEqual(self.archive.read_bytes(), b"altered disposable test archive")

    def test_absent_archive_outside_allowed_location_still_rejected(self):
        self.commit()
        foreign = {**self.manifest, "archive": str(self.fixture.bank/self.archive.name)}
        with self.assertRaisesRegex(EvidenceError, "Unsafe"):
            self.cleanup([foreign])

    def test_wrong_case_filename_still_rejected_when_missing(self):
        self.commit()
        self.lose_cache()
        foreign = {**self.manifest, "archive": str(self.archive.with_name("unrelated.zip"))}
        with self.assertRaisesRegex(EvidenceError, "filename"):
            self.cleanup([foreign])

    def test_normal_cleanup_and_then_lost_owner_resume(self):
        self.commit()
        self.cleanup()
        self.assertFalse(self.archive.exists())
        self.lose_cache()
        self.cleanup()

    def test_complete_method_resumes_without_any_decoder_or_attack_stage(self):
        from wmbench import colab_canary

        f = self.fixture
        self.commit()
        self.lose_cache()
        write_json(f.root/f.method/"quality"/(f.case["id"]+".json"),
                   dict(case_key=f.case["key"], ssim=1., lpips=0.))
        ctx = dict(identity="identity", bank=f.bank, output=f.root, scratch=f.scratch,
                   models={f.method: {"id": f.method}}, selected=[f.method], images=[{"image_id": 139}])
        @contextlib.contextmanager
        def lease(_):
            yield lambda: None
        result_path = storage.result_path(f.root, f.case)
        before = sha256(result_path)
        with patch.object(runner, "os", types.SimpleNamespace(name="posix")), \
             patch.object(runner, "context", return_value=ctx), \
             patch.object(runner, "collect", return_value=[f.case]), \
             patch.object(runner, "invoke") as invoke, \
             patch.object(colab_canary, "controller_lease", lease):
            self.assertTrue(runner.run(baseline.PROJECT/"production",
                dict(output=str(f.root), max_hours=1, ssd_reserve_gib=60, pairwise=False)))
        invoke.assert_not_called()
        self.assertEqual(sha256(result_path), before)
        self.assertEqual(read_json(f.root/f.method/"status.json")["status"], "complete")

    def test_source_stays_frozen(self):
        self.assertEqual(sha256(Path(storage.__file__)), fix.STORAGE_SHA)
        self.assertIs(runner.remove_packages, storage.remove_packages)


class ResumeNotebook(unittest.TestCase):
    def test_only_entrypoint_and_repair_cell_added(self):
        old = nbformat.read(ROOT/"dist/COCO_Digital_Channel_STEGASTAMP_FIX.ipynb", as_version=4)
        new = nbformat.read(ROOT/"dist/COCO_Digital_Channel_RESUME_FIX.ipynb", as_version=4)
        nbformat.validate(new)
        by_id = {c.id: c for c in new.cells}
        changed = []
        for cell in old.cells:
            revised = by_id[cell.id]
            if revised.source != cell.source:
                self.assertEqual(revised.source, cell.source.replace('"stegastamp_clean_hotfix", "run"',
                                                                    '"digital_resume_hotfix", "run"'))
                changed.append(cell.id)
        self.assertEqual(changed, ["digital-08"])
        for cell in new.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            namespace = dict(Path=Path, hashlib=hashlib, os=os, SCRATCH_ROOT=base/"scratch",
                             OUTPUT_ROOT=base/"results", ENV={"PYTHONPATH": "existing"})
            for _ in range(2):
                for name in ("stegastamp-repair-setup", "resume-cache-setup"):
                    exec(compile(by_id[name].source, name, "exec"), namespace)
            target = base/"scratch/repairs/digital-missing-cache-v1/digital_resume_hotfix.py"
            self.assertEqual(target.read_text("utf-8"), (ROOT/"digital_resume_hotfix.py").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
