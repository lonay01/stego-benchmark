"""Offline scope, runtime, preflight, archival and resume regressions."""

import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import nbformat
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_digital import runner, worker, storage
from wmbench_digital.spec import ATTACKS
from wmbench import colab_canary
from wmbench.adapters.physical import StegaStampONNX
import test_digital as baseline
import stegastamp_cuda_hotfix as fix

ROOT = Path(fix.__file__).parent


def cuda_runtime():
    return {"stegastamp_cuda": {"fix": fix.VERSION, "actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}}


class Runtime(unittest.TestCase):
    def session(self, providers):
        return Mock(get_providers=Mock(return_value=providers), get_provider_options=Mock(return_value={"CUDAExecutionProvider": {"device_id": "0"}}))

    def test_cpu_provider_rejected_even_if_package_supports_cuda(self):
        session = self.session(["CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "CPU-only fallback"):
            fix.require_cuda(session)
        session.disable_fallback.assert_not_called()

    def test_empty_or_cpu_first_rejected(self):
        for providers in ([], ["CPUExecutionProvider", "CUDAExecutionProvider"]):
            with self.assertRaises(RuntimeError):
                fix.require_cuda(self.session(providers))

    def test_actual_cuda_is_recorded_and_fallback_disabled(self):
        session = self.session(["CUDAExecutionProvider", "CPUExecutionProvider"])
        evidence = fix.require_cuda(session)
        self.assertEqual(evidence["actual_providers"][0], "CUDAExecutionProvider")
        self.assertFalse(evidence["automatic_fallback"])
        session.disable_fallback.assert_called_once()

    def test_missing_libraries_fail_before_any_mutation(self):
        with patch.object(fix.importlib.metadata, "distribution", side_effect=fix.importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "libcublasLt"):
                fix.library_environment({"LD_LIBRARY_PATH": "old"})

    def test_existing_library_linkage_is_local_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, filenames in (("cublas", fix.REQUIRED_LIBRARIES[:2]), ("cudnn", ["libcudnn.so.9"]), ("cuda_runtime", ["libcudart.so.12"])):
                folder = root / "nvidia" / name / "lib"
                folder.mkdir(parents=True)
                for filename in filenames:
                    (folder / filename).write_bytes(filename.encode())
            dist = types.SimpleNamespace(version="test-version", locate_file=lambda p: root / p)
            original = {"LD_LIBRARY_PATH": "unchanged", "CUSTOM": "value"}
            with patch.object(fix.importlib.metadata, "distribution", return_value=dist):
                result = fix.library_environment(original)
            self.assertEqual(original, {"LD_LIBRARY_PATH": "unchanged", "CUSTOM": "value"})
            self.assertTrue(result["LD_LIBRARY_PATH"].endswith("unchanged"))
            receipt = json.loads(result["WM_STEGA_CUDA_LIBRARIES"])
            self.assertEqual(set(receipt["libraries"]), set(fix.REQUIRED_LIBRARIES))
            for item in receipt["libraries"].values():
                self.assertEqual(item["sha256"], sha256(item["path"]))

    def test_worker_guard_supersedes_old_tolerance_without_changing_disk(self):
        import stegastamp_clean_hotfix as previous
        old_decode, old_runtime, old_load = worker.decode, worker.runtime_info, StegaStampONNX._load
        try:
            previous.install_worker_patch()
            self.assertIn("_check_stegastamp_clean", worker.decode.__code__.co_names)
            fix.install_worker_guard()
            self.assertNotIn("_check_stegastamp_clean", worker.decode.__code__.co_names)
            self.assertEqual(sha256(worker.__file__), fix.WORKER_SHA)
            self.assertTrue(StegaStampONNX._load._cuda_guard)
        finally:
            worker.decode, worker.runtime_info, StegaStampONNX._load = old_decode, old_runtime, old_load

    def test_redirect_only_stegastamp_prepare_and_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            env = {"PYTHONPATH": "original", "LD_LIBRARY_PATH": "old"}
            linked = {"LD_LIBRARY_PATH": "cuda-libraries", "WM_STEGA_CUDA_LIBRARIES": "{}"}
            for name in ("stegastamp", "rosteals"):
                for stage in ("prepare", "decode", "attacks", "quality"):
                    write_json(path, {"model": {"id": name}})
                    command = ["python", "-m", "wmbench.process_guard", "--", "python", "-m", "wmbench_digital.runner", stage, "--settings", str(path)]
                    new, updated = fix.redirect(command, env, linked)
                    if name == "stegastamp" and stage in {"prepare", "decode"}:
                        self.assertEqual(updated["LD_LIBRARY_PATH"], "cuda-libraries")
                        self.assertEqual(new[6], "stegastamp_cuda_hotfix" if stage == "decode" else "wmbench_digital.runner")
                    else:
                        self.assertEqual(new, command)
                        self.assertIs(updated, env)
            self.assertEqual(env["LD_LIBRARY_PATH"], "old")
            write_json(path, {"model": {"id": "stegastamp"}, "stegastamp_cuda_preflight": True})
            cmd = ["python", "-m", "wmbench_digital.runner", "decode", "--settings", str(path)]
            self.assertEqual(fix.redirect(cmd, env, linked)[0][3], "preflight")


class Records(unittest.TestCase):
    def setUp(self):
        self.f = baseline.Pipeline()
        self.f.setUp()
        self.f.method = fix.METHOD
        self.f.case["method"] = fix.METHOD
        self.f.job["model"] = {"id": fix.METHOD}
        self.f.job["slot"] = str(self.f.scratch / "slots" / fix.METHOD)
        write_json(self.f.root / fix.METHOD / "cases.json", [self.f.case])
        write_json(self.f.bank / "decoder.json", {"payload": [0, 0], "runtime_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]})
        self.case, self.job = self.f.case, self.f.job
        self.path = storage.result_path(self.f.root, self.case)

    def tearDown(self):
        self.f.tearDown()

    def commit(self, cuda=False, partial=False):
        execution = {"runtime": cuda_runtime() if cuda else {}, "session": "old"}
        rows = {a["id"]: {**self.case["clean"], "execution": execution} for a in ATTACKS[:3] if partial}
        if not partial:
            rows = {a["id"]: {**self.case["clean"], "execution": execution} for a in ATTACKS}
        document = dict(case_key=self.case["key"], case=self.case, rows=rows)
        storage.save_result(self.path, document)
        return document

    def passed(self):
        report = dict(fix=fix.VERSION, status="passed", session=self.job["session"], identity=self.job["identity"],
                      case_keys=[self.case["key"]], execution=cuda_runtime(),
                      checks=[{"case_key": self.case["key"], "passed": True}])
        write_json(fix.receipt_path(self.job), report)

    def call_preflight(self, row=None):
        with patch.object(fix, "install_worker_guard"), patch.object(worker, "load_adapter", return_value=object()), \
             patch.object(worker, "runtime_info", return_value=cuda_runtime()), \
             patch.object(worker, "measured", return_value=self.case["clean"] if row is None else row):
            fix.preflight(self.job)

    def test_preflight_requires_exact_bits_even_when_ber_is_same(self):
        self.commit()
        before = sha256(self.path)
        with self.assertRaisesRegex(EvidenceError, "still differs"):
            self.call_preflight({**self.case["clean"], "decoded_bits": [1, 0]})
        self.assertEqual(read_json(fix.receipt_path(self.job))["status"], "failed")
        self.assertEqual(sha256(self.path), before)
        with self.assertRaisesRegex(RuntimeError, "must pass"):
            fix.quarantine_legacy(self.job, [self.case])

    def test_preflight_requires_all_cases(self):
        self.commit()
        second = {**self.case, "id": "second", "key": "second"}
        write_json(self.f.root / fix.METHOD / "cases.json", [self.case, second])
        with patch.object(fix, "install_worker_guard"), patch.object(worker, "load_adapter", return_value=object()), \
             patch.object(worker, "runtime_info", return_value=cuda_runtime()), \
             patch.object(worker, "measured", side_effect=[self.case["clean"], {**self.case["clean"], "ber": .5}]):
            with self.assertRaises(EvidenceError):
                fix.preflight(self.job)
        self.assertEqual(len(read_json(fix.receipt_path(self.job))["checks"]), 2)
        self.assertTrue(self.path.exists())

    def test_preflight_rejects_non_cuda_bank(self):
        write_json(self.f.bank / "decoder.json", {"payload": [0, 0], "runtime_providers": ["CPUExecutionProvider"]})
        with self.assertRaisesRegex(EvidenceError, "Bank carrier"):
            self.call_preflight()

    def test_preflight_rejects_payload_leak(self):
        write_json(self.f.bank / "decoder.json", {"payload": [0, 1], "runtime_providers": ["CUDAExecutionProvider"]})
        with self.assertRaisesRegex(EvidenceError, "Reference payload"):
            self.call_preflight()

    def test_preflight_stops_at_budget_before_withdrawal(self):
        self.commit()
        self.job["deadline"] = 0
        with self.assertRaises(TimeoutError):
            self.call_preflight()
        self.assertTrue(self.path.exists())

    def test_preflight_passes_without_mutating_records_or_bank(self):
        self.commit()
        before = (sha256(self.path), sha256(self.f.bank / "carrier.png"))
        self.call_preflight()
        report = read_json(fix.receipt_path(self.job))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["case_keys"], [self.case["key"]])
        self.assertEqual(before, (sha256(self.path), sha256(self.f.bank / "carrier.png")))

    def test_legacy_backup_is_exact_idempotent_and_other_method_unchanged(self):
        document = self.commit()
        before = self.path.read_bytes()
        other = {**self.case, "method": "rosteals"}
        other_path = storage.result_path(self.f.root, other)
        storage.save_result(other_path, {**document, "case": other})
        other_hash = sha256(other_path)
        self.passed()
        fix.quarantine_legacy(self.job, [self.case])
        fix.quarantine_legacy(self.job, [self.case])
        self.assertFalse(self.path.exists())
        backups = list((self.f.root / fix.METHOD / "_preserved_cpu_records").rglob("*.json.gz"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertEqual(storage.read_result(backups[0]), document)
        self.assertEqual(sha256(other_path), other_hash)
        self.assertEqual(read_json(backups[0].with_suffix(".receipt.json"))["state"], "withdrawn_from_active_tables")

    def test_new_cuda_rows_including_partial_remain_active(self):
        self.passed()
        for partial in (False, True):
            self.commit(cuda=True, partial=partial)
            before = sha256(self.path)
            fix.quarantine_legacy(self.job, [self.case])
            self.assertEqual(sha256(self.path), before)

    def test_mixed_backend_record_must_be_recomputed(self):
        doc = self.commit(cuda=True)
        doc["rows"]["clean"]["execution"] = {"runtime": {}}
        self.assertFalse(fix.cuda_record(doc))

    def test_failed_backup_readback_never_deletes_active_record(self):
        self.commit()
        self.passed()
        before = sha256(self.path)
        with patch.object(fix, "checksum", return_value="bad-readback"):
            with self.assertRaisesRegex(RuntimeError, "backup readback"):
                fix.quarantine_legacy(self.job, [self.case])
        self.assertEqual(sha256(self.path), before)

    def test_crash_after_backup_before_unlink_recovers(self):
        self.commit()
        self.passed()
        before = self.path.read_bytes()
        with patch.object(Path, "unlink", side_effect=OSError("interrupted before unlink")):
            with self.assertRaises(OSError):
                fix.quarantine_legacy(self.job, [self.case])
        self.assertEqual(self.path.read_bytes(), before)
        fix.quarantine_legacy(self.job, [self.case])
        self.assertFalse(self.path.exists())

    def test_changed_selection_cannot_use_old_preflight(self):
        self.commit()
        self.passed()
        with self.assertRaises(RuntimeError):
            fix.quarantine_legacy(self.job, [{**self.case, "key": "different"}])
        self.assertTrue(self.path.exists())

    def test_preflight_summary_without_all_checks_cannot_withdraw(self):
        self.commit()
        self.passed()
        report = read_json(fix.receipt_path(self.job))
        report["checks"] = []
        write_json(fix.receipt_path(self.job), report)
        with self.assertRaisesRegex(RuntimeError, "must pass"):
            fix.quarantine_legacy(self.job, [self.case])
        self.assertTrue(self.path.exists())

    def test_outside_record_target_rejected(self):
        self.commit()
        malicious = {**self.case, "id": "../../outside"}
        self.passed()
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            fix.quarantine_legacy(self.job, [malicious])
        self.assertTrue(self.path.exists())

    def test_existing_archives_regenerate_and_all_79_rows_commit_then_resume(self):
        # Real attack PNG/ZIP creation and checkpoint persistence, fake model only.
        manifest = self.f.packages()
        self.commit()
        Path(manifest["archive"]).unlink()
        self.call_preflight()
        fix.quarantine_legacy(self.job, [self.case])
        regenerated = self.f.packages()
        self.assertEqual(manifest["sha256"], regenerated["sha256"])
        adapter = Mock(decode=Mock(return_value={"bits": [0, 1]}))
        with patch.object(worker, "load_adapter", return_value=adapter), \
             patch.object(worker, "runtime_info", return_value=cuda_runtime()):
            worker.decode(self.job)
            worker.decode(self.job)
        self.assertEqual(adapter.decode.call_count, 79)
        document = storage.read_result(self.path, self.case)
        self.assertTrue(storage.complete(document))
        self.assertTrue(fix.cuda_record(document))
        before = sha256(self.path)
        fix.quarantine_legacy(self.job, [self.case])
        self.assertEqual(sha256(self.path), before)


class Controller(unittest.TestCase):
    @contextlib.contextmanager
    def installed(self):
        old = (runner.run, runner.invoke, colab_canary.monitored_process, runner.remove_packages, storage.remove_packages)
        try:
            fix.install_controller_patch()
            yield
        finally:
            runner.run, runner.invoke, colab_canary.monitored_process, runner.remove_packages, storage.remove_packages = old

    def test_only_one_hook_inserted_and_source_file_unchanged(self):
        with self.installed():
            self.assertIn("_stega_cuda_before", runner.run.__code__.co_names)
            self.assertEqual(sha256(runner.__file__), fix.RUNNER_SHA)

    def test_other_model_hook_is_noop(self):
        with self.installed(), patch.object(fix, "library_environment") as libraries:
            runner._stega_cuda_before({"model": {"id": "rosteals"}}, None, lambda: None, [])
            libraries.assert_not_called()

    def test_preparation_then_preflight_then_quarantine_before_original_attacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(model={"id": "stegastamp"}, session="abc", results=str(root), identity="id")
            calls = []
            def invoke(j, stage, path, owned):
                calls.append((stage, j.get("stegastamp_cuda_preflight", False)))
            with patch.object(runner, "invoke", side_effect=invoke), self.installed(), \
                 patch.object(fix, "library_environment", return_value={"LD_LIBRARY_PATH": "libs"}), \
                 patch.object(fix, "quarantine_legacy", side_effect=lambda *a: calls.append(("quarantine", True))):
                case = {"method": "stegastamp", "id": "one", "key": "key"}
                runner._stega_cuda_before(job, root / "job.json", lambda: None, [case])
                runner.invoke(job, "prepare", root / "job.json", lambda: None)
                runner.invoke(job, "attacks", root / "job.json", lambda: None)
            self.assertEqual(calls, [("prepare", False), ("decode", True), ("quarantine", True), ("attacks", False)])

    def test_failed_preflight_cannot_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(model={"id": "stegastamp"}, session="abc", results=str(root), identity="id")
            with patch.object(runner, "invoke", side_effect=[None, RuntimeError("CUDA preflight failed")]), self.installed(), \
                 patch.object(fix, "library_environment", return_value={"LD_LIBRARY_PATH": "libs"}), \
                 patch.object(fix, "quarantine_legacy") as archive:
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    runner._stega_cuda_before(job, root / "job.json", lambda: None,
                                             [{"method": "stegastamp", "id": "one", "key": "key"}])
                archive.assert_not_called()


class Notebook(unittest.TestCase):
    def test_only_run_entry_and_two_new_cells_differ(self):
        old = nbformat.read(ROOT / "dist/COCO_Digital_Channel_RESUME_FIX.ipynb", as_version=4)
        new = nbformat.read(ROOT / "dist/COCO_Digital_Channel_STEGASTAMP_CUDA_FIX.ipynb", as_version=4)
        nbformat.validate(new)
        by_id = {c.id: c for c in new.cells}
        changed = []
        for cell in old.cells:
            revised = by_id[cell.id]
            if revised.source != cell.source:
                self.assertEqual(revised.source, cell.source.replace('"digital_resume_hotfix", "run"', '"stegastamp_cuda_hotfix", "run"'))
                changed.append(cell.id)
        self.assertEqual(changed, ["digital-08"])
        self.assertEqual(len(new.cells), len(old.cells) + 2)
        for cell in new.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertFalse(cell.outputs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = dict(Path=Path, hashlib=hashlib, os=os, SCRATCH_ROOT=root / "scratch", OUTPUT_ROOT=root / "results", ENV={"PYTHONPATH": "existing"})
            for _ in range(2):
                for name in ("stegastamp-repair-setup", "resume-cache-setup", "stegastamp-cuda-setup"):
                    exec(compile(by_id[name].source, name, "exec"), namespace)
            self.assertEqual((root / "scratch/repairs/stegastamp-cuda-v1/stegastamp_cuda_hotfix.py").read_text("utf-8"),
                             (ROOT / "stegastamp_cuda_hotfix.py").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
