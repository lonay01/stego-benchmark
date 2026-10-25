"""Regressions for the StegaStamp-only replay exception and notebook delivery."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import nbformat
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_full_payload.contracts import score
from wmbench_digital import worker
from wmbench_digital.storage import result_path, read_result, complete
from wmbench_digital.report import comparison
import stegastamp_clean_hotfix as fix
import test_digital as baseline

ROOT = Path(fix.__file__).parent


class CleanReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        (self.bank / "carrier.png").write_bytes(b"unchanged-test-carrier")
        self.assignment = dict(payload=[0]*100, capacity=dict(information_indices=list(range(100)),
            reserved_bits={}, information_bits=100, interface="raw_message"))
        old = [1]*48 + [0]*52
        fresh = old.copy()
        fresh[47] = 0
        self.old, self.fresh = self.row(old), self.row(fresh)
        self.case = dict(method="stegastamp", mode="matched_42", id="case", key="key",
            assignment=self.assignment, clean=self.old, output="carrier.png",
            carrier_sha256=sha256(self.bank/"carrier.png"))
        self.job = dict(bank=str(self.bank), deadline=time.time()+60)
        self.execution = dict(session="test", runtime={})
        self.decode_function = worker.decode

    def tearDown(self):
        worker.decode = self.decode_function
        self.temp.cleanup()

    def row(self, bits):
        result = score({"bits": bits, "details": {"implementation": "wibe_onnx", "ecc": "none_raw100"}}, self.assignment)
        result.pop("expected_bits")
        result["seconds"] = 0.1
        return result

    def check(self, fresh=None, prior=None, repeats=None):
        value = copy.deepcopy(self.fresh if fresh is None else fresh)
        repeats = [copy.deepcopy(value)]*2 if repeats is None else repeats
        with patch.object(worker, "measured", side_effect=repeats) as measured:
            fix.check_clean(value, self.case, object(), "unused", 1, self.root, self.execution,
                            {} if prior is None else prior, self.job)
        return value, measured.call_count

    def test_exact_match_unchanged_no_retries(self):
        value, calls = self.check(self.old)
        self.assertEqual(value, self.old)
        self.assertEqual(calls, 0)

    def test_stable_one_bit_records_first_fresh_and_preserves_bank(self):
        original = copy.deepcopy(self.case)
        value, calls = self.check()
        self.assertEqual(calls, 2)
        self.assertEqual(value["ber"], .47)
        self.assertEqual(self.case, original)
        receipt = read_json(self.root / value["clean_replay"]["receipt"])
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["bank"]["ber"], .48)
        self.assertEqual(receipt["first_fresh"]["ber"], .47)
        self.assertEqual(comparison({"status": "measured", "ber": .50, "exact": False}, value)["delta_ber_pp"],
                         (.50-.47)*100)

    def test_worse_first_fresh_is_not_replaced_with_better_bank(self):
        self.case["clean"], self.fresh = self.fresh, self.old
        result, _ = self.check()
        self.assertEqual(result["ber"], .48)

    def test_two_bits_still_stop(self):
        self.fresh = self.row([1]*46 + [0]*54)
        with self.assertRaises(EvidenceError):
            self.check()

    def test_native_and_other_methods_still_stop(self):
        for method, mode in [("stegastamp", "native"), ("sepwater", "matched_42")]:
            with self.subTest(method=method, mode=mode):
                self.case.update(method=method, mode=mode)
                with self.assertRaises(EvidenceError):
                    self.check()

    def test_unstable_repeat_stops_no_best_of_three(self):
        with self.assertRaisesRegex(EvidenceError, "unstable"):
            self.check(repeats=[self.old, self.fresh])
        receipt = read_json(self.root / "stegastamp/clean-replay/case-test.json")
        self.assertFalse(receipt["accepted"])

    def test_partial_case_cannot_change_clean(self):
        with self.assertRaisesRegex(EvidenceError, "partial"):
            self.check(prior={"clean": self.old})
        value, calls = self.check(prior={"clean": self.fresh})
        self.assertEqual(value["ber"], .47)
        self.assertEqual(calls, 2)

    def test_changed_png_is_not_tolerated(self):
        (self.bank/"carrier.png").write_bytes(b"changed")
        with self.assertRaisesRegex(EvidenceError, "checksum"):
            self.check()

    def test_score_and_status_must_be_consistent(self):
        for key, value in [("ber", .01), ("status", "error"), ("exact", True), ("information_bits", 30)]:
            with self.subTest(key=key):
                bad = {**self.fresh, key: value}
                with self.assertRaises(EvidenceError):
                    self.check(bad)

    def test_preserves_original_drift_evidence(self):
        path = self.root / "stegastamp/clean-drift/case.json"
        write_json(path, {"historical_diagnostic": True})
        self.check()
        self.assertEqual(read_json(path), {"historical_diagnostic": True})

    def test_patch_changes_only_one_comparison_in_memory(self):
        before = Path(worker.__file__).read_bytes()
        fix.install_worker_patch()
        self.assertNotEqual(worker.decode, self.decode_function)
        self.assertEqual(Path(worker.__file__).read_bytes(), before)
        self.assertIn("_check_stegastamp_clean", worker.decode.__code__.co_names)

    def test_only_stegastamp_decode_command_is_redirected(self):
        path = self.root/"job.json"
        env = {"PYTHONPATH": "production"}
        for name in ("stegastamp", "sepwater", "trustmark_q"):
            write_json(path, {"model": {"id": name}})
            for stage in ("decode", "prepare", "attacks", "quality"):
                command = ["controller", "-m", "wmbench.process_guard", "--", "isolated-python",
                           "-m", "wmbench_digital.runner", stage, "--settings", str(path)]
                new, new_env = fix.redirected_command(command, env)
                if name == "stegastamp" and stage == "decode":
                    self.assertIn("stegastamp_clean_hotfix", new)
                    self.assertIn(str(ROOT), new_env["PYTHONPATH"])
                    self.assertEqual(new[4], "isolated-python")
                else:
                    self.assertEqual(new, command)
                    self.assertEqual(new_env, env)

    def test_all_79_conditions_commit_and_resume_without_touching_bank(self):
        fixture = baseline.Pipeline()
        fixture.setUp()
        try:
            fixture.case.update(method="stegastamp", mode="matched_42", assignment=self.assignment,
                                clean=self.old, id="case_matched_42")
            fixture.method = "stegastamp"
            fixture.job["model"]["id"] = "stegastamp"
            write_json(fixture.root/"stegastamp/cases.json", [fixture.case])
            fixture.packages()
            before = {p.relative_to(fixture.bank).as_posix(): sha256(p) for p in fixture.bank.rglob("*") if p.is_file()}
            fix.install_worker_patch()
            bits = self.fresh["decoded_bits"]
            class Adapter:
                calls = 0
                def decode(self, *_):
                    self.calls += 1
                    return {"bits": bits, "details": {"implementation": "wibe_onnx"}}
            adapter = Adapter()
            with patch.object(worker, "load_adapter", return_value=adapter):
                worker.decode(fixture.job)
                self.assertEqual(adapter.calls, 81)  # Clean + 2 diagnostics + 78 attacks.
                document = read_result(result_path(fixture.root, fixture.case), fixture.case)
                self.assertTrue(complete(document))
                self.assertEqual(document["rows"]["clean"]["ber"], .47)
                worker.decode(fixture.job)
                self.assertEqual(adapter.calls, 81)
            after = {p.relative_to(fixture.bank).as_posix(): sha256(p) for p in fixture.bank.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
        finally:
            fixture.tearDown()


class Notebook(unittest.TestCase):
    def test_only_run_entry_changed_and_setup_executes(self):
        old = nbformat.read(ROOT/"dist/COCO_Digital_Channel_RESUMABLE.ipynb", as_version=4)
        new = nbformat.read(ROOT/"dist/COCO_Digital_Channel_STEGASTAMP_FIX.ipynb", as_version=4)
        nbformat.validate(new)
        by_id = {c.id: c for c in new.cells}
        changes = []
        for cell in old.cells:
            revised = by_id[cell.id]
            if revised.source != cell.source:
                changes.append(cell.id)
                self.assertEqual(revised.source, cell.source.replace('"wmbench_digital.runner", "run"',
                                                                   '"stegastamp_clean_hotfix", "run"'))
        self.assertEqual(changes, ["digital-08"])
        for cell in new.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            namespace = dict(Path=Path, hashlib=hashlib, os=os, SCRATCH_ROOT=base/"scratch",
                             OUTPUT_ROOT=base/"results", ENV={"PYTHONPATH": "existing"})
            source = by_id["stegastamp-repair-setup"].source
            exec(compile(source, "notebook repair cell", "exec"), namespace)
            exec(compile(source, "notebook repair cell resumed", "exec"), namespace)
            saved = base/"scratch/repairs/stegastamp-clean-replay-v1/stegastamp_clean_hotfix.py"
            self.assertEqual(saved.read_text("utf-8"), (ROOT/"stegastamp_clean_hotfix.py").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
