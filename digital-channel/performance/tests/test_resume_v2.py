"""Synthetic regression tests: unchanged fast-mode trust boundary, less I/O."""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from wmbench.io import write_json
from wmbench_fast import cache, completed, incremental
from wmbench_fast.runner import mark_finished, TABLES
from wmbench_digital.storage import save_result, result_path
from wmbench_full_payload.storage import EvidenceError
import test_performance


class ResumeV2(unittest.TestCase):
    setUp = test_performance.Performance.setUp
    tearDown = test_performance.Performance.tearDown
    document = test_performance.Performance.document
    def seal_fixture(self):
        save_result(result_path(self.root, self.case), self.document())
        folder = self.root / self.method
        for name in TABLES:
            p = folder / "tables" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture")
        write_json(folder / "tables/counts.json", dict(carriers=1, complete_carriers=1))
        write_json(folder / "coverage.json", [])
        cache.pack(folder / "_fast/catalogue.json.gz", {"synthetic": True})
        mark_finished(self.root, self.method, [self.case])
        self.ctx = dict(identity="run", output=self.root, images=[{"image_id": 139}])
        self.model = dict(id=self.method)
        completed.seal(self.ctx, self.model)
        return folder

    def test_small_receipt_skips_large_catalogue_and_raw_results(self):
        folder = self.seal_fixture()
        before = {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}
        original = completed.read_json
        def read(path):
            self.assertEqual(Path(path).name, "counts.json")
            return original(path)
        with patch.object(completed, "read_json", side_effect=read):
            self.assertEqual(completed.check(self.ctx, self.model)["complete_carriers"], 1)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def test_new_selection_and_full_audit_do_not_skip(self):
        self.seal_fixture()
        self.assertIsNone(completed.check(self.ctx, self.model, True))
        for key, value in (("identity", "other"), ("images", [{"image_id": 285}])):
            self.assertIsNone(completed.check(dict(self.ctx, **{key: value}), self.model))
        self.assertIsNone(completed.check(self.ctx, dict(self.model, native_bits=999)))

    def test_drive_mtime_drift_requires_identical_content(self):
        folder = self.seal_fixture()
        for relative in ("cases.json", "tables/attack_summary.csv"):
            path = folder / relative
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 8_000_000_000))
        self.assertEqual(completed.check(self.ctx, self.model)["carriers"], 1)
        (folder / "tables/attack_summary.csv").write_text("altered")
        self.assertIsNone(completed.check(self.ctx, self.model))

    def test_legacy_receipt_does_not_accept_unhashed_mtime_drift(self):
        folder = self.seal_fixture()
        path = folder / "_fast/complete.json.gz"
        old = cache.unpack(path)
        old["snapshot"].pop("table_sha256")
        cache.pack(path, old)
        completed.seal(self.ctx, self.model)
        table = folder / "tables/attack_summary.csv"
        stat = table.stat()
        os.utime(table, ns=(stat.st_atime_ns, stat.st_mtime_ns + 8_000_000_000))
        self.assertIsNone(completed.check(self.ctx, self.model))

    def test_changed_or_missing_result_invalidates_compact_receipt(self):
        self.seal_fixture()
        p = result_path(self.root, self.case)
        save_result(p, dict(case_key=self.case["key"], rows={}))
        self.assertIsNone(completed.check(self.ctx, self.model))
        p.unlink()
        self.assertIsNone(completed.check(self.ctx, self.model))

    def test_changed_catalogue_or_report_invalidates_compact_receipt(self):
        for relative in ("cases.json", "coverage.json", "_fast/catalogue.json.gz", "tables/attack_summary.csv"):
            folder = self.seal_fixture()
            (folder / relative).write_bytes(b"changed")
            self.assertIsNone(completed.check(self.ctx, self.model))

    def test_incremental_audit_retains_completed_image_after_next_failure(self):
        calls = []
        ctx = dict(output=self.root, scratch=self.scratch, identity="run",
                   images=[{"image_id": 1}, {"image_id": 2}])
        model = {"id": self.method}
        fail = True
        def fake_collect(single, model):
            i = single["images"][0]["image_id"]
            calls.append(i)
            if i == 2 and fail:
                raise EvidenceError("damaged second image")
            write_json(single["output"] / self.method / "coverage.json", [{"image_id": i}])
            return [{"image_id": i}]
        with patch.object(incremental.original_bank, "collect", side_effect=fake_collect), \
             patch.object(incremental, "validate_catalogue", side_effect=lambda a, b, *_: (a, b)):
            with self.assertRaises(EvidenceError):
                incremental.collect_images(ctx, model)
            fail = False
            cases, _ = incremental.collect_images(ctx, model)
            self.assertEqual(calls, [1, 2, 2])
            self.assertEqual(cases, [{"image_id": 1}, {"image_id": 2}])
            incremental.collect_images(ctx, model, full_audit=True)
            self.assertEqual(calls, [1, 2, 2, 1, 2])

    def test_catalogue_reuse_does_not_rewrite_cache(self):
        ctx = dict(output=self.root, identity="test")
        folder = self.root / self.method
        rows, coverage = [self.case], [{"fixture": True}]
        cache.pack(folder / "_fast/catalogue.json.gz", dict(identity="test", cases=rows, coverage=coverage))
        write_json(folder / "coverage.json", coverage)
        with patch.object(cache, "validate_catalogue", return_value=(rows, coverage)), \
             patch.object(cache, "pack", side_effect=AssertionError("Unexpected rewrite")):
            self.assertEqual(cache.collect(ctx, {"id": self.method}), rows)
