"""Report-only regression and interrupted-publication tests (synthetic CPU data)."""

import gzip
from pathlib import Path
import unittest
from unittest.mock import patch

from wmbench.io import read_json, sha256, write_json
from wmbench_digital import report
from wmbench_digital.storage import result_path, save_result
from wmbench_fast.cache import ResultCache
from wmbench_fast import report_io
from wmbench_full_payload.storage import EvidenceError
import test_performance as fixtures


class ReportIO(unittest.TestCase):
    def setUp(self):
        fixtures.Performance.setUp(self)
        self.record = result_path(self.root, self.case)
        save_result(self.record, fixtures.Performance.document(self))
        self.index = ResultCache(self.root, self.method, [self.case])
        self.index.warm()
        self.tables = self.root / self.method / "tables"
        self.before_record = sha256(self.record)
        self.before_bank = {p: sha256(p) for p in self.bank.rglob("*") if p.is_file()}

    tearDown = fixtures.Performance.tearDown

    def run_report(self):
        return report_io.method_report(self.root, self.method, self.scratch / "reports",
                                       self.index, [self.case])

    def contents(self):
        return {n: (gzip.decompress((self.tables / n).read_bytes()) if n.endswith(".gz")
                    else (self.tables / n).read_bytes()) for n in report_io.TABLES}

    def assert_evidence_unchanged(self):
        self.assertEqual(sha256(self.record), self.before_record)
        self.assertEqual(self.before_bank, {p: sha256(p) for p in self.before_bank})

    def test_original_numerical_output_and_previous_tables_preserved(self):
        report.method_report(self.root, self.method)
        expected = self.contents()
        old_hashes = {n: sha256(self.tables / n) for n in report_io.TABLES}
        counts = self.run_report()
        self.assertEqual(counts["complete_carriers"], 1)
        self.assertEqual(self.contents(), expected)
        receipt = read_json(self.root / self.method / "_fast/report-publication.json")
        archive = Path(receipt["previous_tables"])
        self.assertEqual({n: sha256(archive / n) for n in old_hashes}, old_hashes)
        self.assert_evidence_unchanged()

    def test_no_report_rename_on_durable_storage(self):
        original = Path.replace
        def refuse_drive(path, target):
            if path.is_relative_to(self.root):
                raise FileNotFoundError("Simulated Drive .partial rename failure")
            return original(path, target)
        with patch.object(Path, "replace", refuse_drive):
            self.run_report()
        self.assert_evidence_unchanged()

    def test_failed_staging_keeps_previous_canonical_tables(self):
        report.method_report(self.root, self.method)
        before = {n: sha256(self.tables / n) for n in report_io.TABLES}
        original = report_io.copy_verified
        def fail(source, destination, *args, **kwargs):
            if "_report_versions" in str(destination):
                raise OSError("Simulated Drive disconnect")
            return original(source, destination, *args, **kwargs)
        with patch.object(report_io, "copy_verified", fail), self.assertRaises(OSError):
            self.run_report()
        self.assertEqual(before, {n: sha256(self.tables / n) for n in before})
        self.assert_evidence_unchanged()

    def test_partial_publication_resumes_without_recomputing_metrics(self):
        report.method_report(self.root, self.method)
        expected = self.contents()
        original = report_io.copy_verified
        def fail(source, destination, *args, **kwargs):
            if destination == self.tables / "attack_summary.csv":
                raise OSError("Simulated interrupted canonical copy")
            return original(source, destination, *args, **kwargs)
        with patch.object(report_io, "copy_verified", fail), self.assertRaises(OSError):
            self.run_report()
        self.assertTrue((self.root / self.method / "_fast/report-staged.json").is_file())
        with patch.object(report, "method_report", side_effect=AssertionError("No recomputation")):
            self.run_report()
        self.assertEqual(self.contents(), expected)
        self.assert_evidence_unchanged()

    def test_corrupt_staged_version_refused(self):
        self.run_report()
        manifest = read_json(self.root / self.method / "_fast/report-staged.json")
        target = self.root / self.method / "_report_versions" / manifest["version"] / "counts.json"
        write_json(target, {"tampered": True})
        with self.assertRaises(EvidenceError):
            self.run_report()

    def test_incomplete_results_refused(self):
        save_result(self.record, {"case_key": self.case["key"], "rows": {}})
        self.index = ResultCache(self.root, self.method, [self.case])
        with self.assertRaises(EvidenceError):
            self.run_report()
        self.assertFalse(self.tables.exists())

    def test_foreign_quality_refused(self):
        write_json(self.root / self.method / "quality" / (self.case["id"] + ".json"),
                   {"case_key": "foreign", "ssim": 1, "lpips": 0})
        with self.assertRaises(EvidenceError):
            self.run_report()
        self.assertFalse(self.tables.exists())

    def test_source_hash_mismatch_refused(self):
        with self.assertRaises(EvidenceError):
            report_io.copy_verified(self.record, self.scratch / "copy", "bad", lambda: None)

    def test_changed_selection_blocks_publication(self):
        original = report.method_report
        def change_selection(*args):
            result = original(*args)
            write_json(self.root / "selection.json", {"changed": True})
            return result
        with patch.object(report, "method_report", change_selection), self.assertRaises(EvidenceError):
            self.run_report()
        self.assertFalse(self.tables.exists())


if __name__ == "__main__":
    unittest.main()
