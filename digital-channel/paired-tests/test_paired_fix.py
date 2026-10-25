"""Tests with synthetic model responses; actual CPU inference is separate."""

import ast
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import nbformat
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_digital import worker, runner, storage
from wmbench_digital.spec import ATTACKS
from wmbench_digital.report import comparison
from wmbench import colab_canary
from wmbench.adapters.physical import StegaStampONNX
import test_digital as baseline
import stegastamp_paired_hotfix as fix

ROOT = Path(fix.__file__).parent


def profile():
    identity = {"policy": fix.POLICY, "actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"], "fixture": "simulated-runtime"}
    return {"version": fix.VERSION, "identity": identity, "fingerprint": fix.digest(identity)}


def runtime():
    return {"stegastamp_paired": profile()}


class Paired(unittest.TestCase):
    def setUp(self):
        self.f = baseline.Pipeline()
        self.f.setUp()
        self.f.method = fix.METHOD
        self.f.case["method"] = fix.METHOD
        self.f.job["model"] = {"id": fix.METHOD}
        self.f.job["slot"] = str(self.f.scratch / "slots" / fix.METHOD)
        self.case, self.job = self.f.case, self.f.job
        write_json(self.f.root / fix.METHOD / "cases.json", [self.case])
        self.path = storage.result_path(self.f.root, self.case)
        self.first = {"status": "measured", "decoded_bits": [1, 1], "ber": .5, "exact": False, "seconds": 1.}

    def tearDown(self):
        self.f.tearDown()

    def preflight(self, values=None):
        rows = [copy.deepcopy(self.first) for _ in range(4)] if values is None else values
        with patch.object(fix, "install_numerical_profile"), patch.object(worker, "load_adapter", side_effect=[Mock(), Mock()]), \
             patch.object(worker, "runtime_info", side_effect=[runtime(), runtime()]), \
             patch.object(worker, "measured", side_effect=rows):
            fix.preflight(self.job)

    def legacy(self):
        record = {"case_key": self.case["key"], "case": self.case, "rows": {a["id"]: self.case["clean"] for a in ATTACKS}}
        storage.save_result(self.path, record)
        return record

    def clean_gate(self, clean=None, prior=None, execution=None, case=None):
        clean = copy.deepcopy(self.first if clean is None else clean)
        fix.check_clean(clean, self.case if case is None else case, None, None, None, self.f.root,
                        {"runtime": runtime(), "session": "test"} if execution is None else execution,
                        {} if prior is None else prior, self.job)
        return clean

    def test_stable_historical_difference_is_audited_not_a_crash(self):
        before = sha256(self.f.bank / "carrier.png")
        self.preflight()
        control = read_json(fix.baseline_path(self.job, self.case))
        self.assertEqual(control["first"], self.first)
        self.assertEqual(control["historical"], self.case["clean"])
        self.assertEqual(sha256(self.f.bank / "carrier.png"), before)
        report = read_json(fix.receipt_path(self.job))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checks"][0]["difference"]["different_bit_positions"], [0])
        self.assertEqual(report["checks"][0]["difference"]["ber_difference_pp"], 50.)

    def test_observed_100_bit_51_to_50_case_is_not_rejected_if_repeatable(self):
        old, fresh = [1] * 51 + [0] * 49, [1] * 51 + [0] * 49
        fresh[24] = 0
        self.case["clean"] = {"status": "measured", "decoded_bits": old, "ber": .51, "exact": False}
        self.first = {"status": "measured", "decoded_bits": fresh, "ber": .50, "exact": False}
        write_json(self.f.root / fix.METHOD / "cases.json", [self.case])
        self.preflight()
        difference = read_json(fix.receipt_path(self.job))["checks"][0]["difference"]
        self.assertEqual(difference["different_bit_positions"], [24])
        self.assertAlmostEqual(difference["ber_difference_pp"], -1.)
        self.clean_gate()

    def test_first_observation_not_fastest_repeat_is_preserved(self):
        rows = [{**self.first, "seconds": t} for t in (10., 2., 3., 4.)]
        self.preflight(rows)
        self.assertEqual(read_json(fix.baseline_path(self.job, self.case))["first"]["seconds"], 10.)

    def test_same_session_instability_is_retained_and_blocks(self):
        self.legacy()
        before = sha256(self.path)
        with self.assertRaisesRegex(RuntimeError, "not repeatable"):
            self.preflight([self.first, self.case["clean"], self.first])
        report = read_json(fix.receipt_path(self.job))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["checks"][0]["extractions"]), 3)
        self.assertFalse(fix.baseline_path(self.job, self.case).exists())
        self.assertEqual(sha256(self.path), before)

    def test_new_session_instability_blocks_without_selecting_better(self):
        with self.assertRaisesRegex(RuntimeError, "not repeatable"):
            self.preflight([self.first] * 3 + [self.case["clean"]])
        self.assertFalse(fix.baseline_path(self.job, self.case).exists())

    def test_technical_error_never_treated_as_stable_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "not repeatable"):
            self.preflight([{"status": "error", "reason": "failure"}] * 3)

    def test_budget_interrupt_retains_legacy(self):
        self.legacy()
        self.job["deadline"] = 0
        with self.assertRaises(TimeoutError):
            self.preflight()
        self.assertTrue(self.path.exists())

    def test_resume_keeps_original_control_bytes_and_time(self):
        self.preflight()
        control = fix.baseline_path(self.job, self.case)
        before = control.read_bytes()
        self.job["session"] = "next-session"
        self.preflight([{**self.first, "seconds": .1}] * 4)
        self.assertEqual(control.read_bytes(), before)

    def test_resume_changed_bits_cannot_replace_committed_control(self):
        self.preflight()
        control = fix.baseline_path(self.job, self.case)
        before = control.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "changed on resume"):
            self.preflight([self.case["clean"]] * 4)
        self.assertEqual(control.read_bytes(), before)

    def test_leaked_payload_and_changed_carrier_block(self):
        write_json(self.f.bank / "decoder.json", {"payload": [1, 0]})
        with self.assertRaisesRegex(EvidenceError, "Reference payload"):
            self.preflight()
        write_json(self.f.bank / "decoder.json", {"payload": [0, 0]})
        self.case["carrier_sha256"] = "bad-hash"
        with self.assertRaisesRegex(EvidenceError, "hash changed"):
            fix.validated_input(self.job, self.case)

    def test_gate_pairs_to_fresh_control_not_historical(self):
        self.preflight()
        clean = self.clean_gate()
        self.assertEqual(clean["ber"], .5)
        self.assertEqual(clean["paired_control"]["historical_bank_ber"], 0.)
        self.assertEqual(comparison({**clean, "ber": .75}, clean)["delta_ber_pp"], 25.)

    def test_gate_still_requires_exact_repeat_of_committed_control(self):
        self.preflight()
        with self.assertRaisesRegex(EvidenceError, "committed paired Clean"):
            self.clean_gate(clean=self.case["clean"])

    def test_gate_rejects_other_method(self):
        self.preflight()
        with self.assertRaises(EvidenceError):
            self.clean_gate(case={**self.case, "method": "rosteals"})

    def test_gate_rejects_different_runtime_fingerprint(self):
        self.preflight()
        changed = runtime()
        changed["stegastamp_paired"]["identity"]["fixture"] = "other-GPU"
        changed["stegastamp_paired"]["fingerprint"] = fix.digest(changed["stegastamp_paired"]["identity"])
        with self.assertRaises(EvidenceError):
            self.clean_gate(execution={"runtime": changed})

    def test_legacy_backed_up_byte_for_byte_and_idempotent(self):
        self.legacy()
        before = self.path.read_bytes()
        self.preflight()
        fix.archive_legacy(self.job, [self.case])
        fix.archive_legacy(self.job, [self.case])
        backups = list((self.f.root / fix.METHOD / "_preserved_previous_records").rglob("*.json.gz"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertFalse(self.path.exists())

    def test_bad_backup_readback_never_deletes_active_record(self):
        self.legacy()
        self.preflight()
        before = sha256(self.path)
        with patch.object(fix.cuda, "checksum", return_value="corrupt-copy"):
            with self.assertRaisesRegex(RuntimeError, "Backup verification"):
                fix.archive_legacy(self.job, [self.case])
        self.assertEqual(sha256(self.path), before)

    def test_interrupted_withdrawal_recovers_from_verified_copy(self):
        self.legacy()
        self.preflight()
        with patch.object(Path, "unlink", side_effect=OSError("interrupt")):
            with self.assertRaises(OSError):
                fix.archive_legacy(self.job, [self.case])
        self.assertTrue(self.path.exists())
        fix.archive_legacy(self.job, [self.case])
        self.assertFalse(self.path.exists())

    def test_cross_method_and_parent_path_targets_rejected(self):
        self.legacy()
        self.preflight()
        for case in ({**self.case, "method": "rosteals"}, {**self.case, "id": "../../foreign"}):
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                fix.archive_legacy(self.job, [case])
        self.assertTrue(self.path.exists())

    def test_decode_real_attack_files_all_79_rows_and_resume(self):
        self.preflight()
        self.f.packages()
        old_decode = worker.decode
        old_runtime, old_load = worker.runtime_info, StegaStampONNX._load
        adapter = Mock(decode=Mock(return_value={"bits": [1, 1]}))
        try:
            with patch.object(fix, "install_numerical_profile"):
                fix.install_worker_patch()
            with patch.object(worker, "load_adapter", return_value=adapter), patch.object(worker, "runtime_info", return_value=runtime()):
                worker.decode(self.job)
                worker.decode(self.job)
            self.assertEqual(adapter.decode.call_count, 79)
            record = storage.read_result(self.path)
            self.assertTrue(storage.complete(record))
            self.assertTrue(fix.current_record(record))
            self.assertEqual(record["case"]["clean"]["ber"], 0.)
            self.assertEqual(record["rows"]["clean"]["ber"], .5)
            before = sha256(self.path)
            fix.archive_legacy(self.job, [self.case])
            self.assertEqual(sha256(self.path), before)
        finally:
            worker.decode, worker.runtime_info, StegaStampONNX._load = old_decode, old_runtime, old_load

    def test_partial_case_cannot_mix_old_backend_rows(self):
        self.preflight()
        with self.assertRaisesRegex(EvidenceError, "Partial case"):
            self.clean_gate(prior={"clean": self.first})

    def test_audit_exposes_both_controls_and_difference(self):
        self.preflight()
        path = self.f.root / fix.METHOD / "tables/stegastamp_bank_vs_paired_clean.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row["bank_ber_percent"]), 0.)
        self.assertEqual(float(row["paired_clean_ber_percent"]), 50.)
        self.assertEqual(float(row["difference_pp"]), 50.)


class Scope(unittest.TestCase):
    def test_only_entrypoint_and_two_added_cells_change(self):
        old = nbformat.read(ROOT / "dist/COCO_Digital_Channel_STEGASTAMP_CUDA_FIX.ipynb", as_version=4)
        new = nbformat.read(ROOT / "dist/COCO_Digital_Channel_STEGASTAMP_PAIRED_FIX.ipynb", as_version=4)
        nbformat.validate(new)
        by_id = {c.id: c for c in new.cells}
        changed = []
        for cell in old.cells:
            if cell.source != by_id[cell.id].source:
                self.assertEqual(by_id[cell.id].source, cell.source.replace('"stegastamp_cuda_hotfix", "run"', '"stegastamp_paired_hotfix", "run"'))
                changed.append(cell.id)
        self.assertEqual(changed, ["digital-08"])
        self.assertEqual(len(new.cells), len(old.cells) + 2)
        for cell in new.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ns = dict(Path=Path, hashlib=hashlib, os=os, SCRATCH_ROOT=root / "scratch", OUTPUT_ROOT=root / "results", ENV={"PYTHONPATH": "old"})
            for _ in range(2):
                for name in ("stegastamp-repair-setup", "resume-cache-setup", "stegastamp-cuda-setup", "stegastamp-paired-setup"):
                    exec(compile(by_id[name].source, name, "exec"), ns)
            self.assertEqual((root / "scratch/repairs/stegastamp-paired-cuda-v1/stegastamp_paired_hotfix.py").read_text("utf-8"),
                             (ROOT / "stegastamp_paired_hotfix.py").read_text("utf-8"))

    def test_controller_redirect_and_noop_for_other_methods(self):
        old = (runner.run, runner.invoke, runner.remove_packages, storage.remove_packages,
               colab_canary.monitored_process, fix.cuda.redirect, fix.cuda.cuda_record, fix.cuda.quarantine_legacy)
        try:
            fix.install_controller_patch()
            with patch.object(fix.cuda, "library_environment") as libraries:
                runner._stega_cuda_before({"model": {"id": "rosteals"}}, None, None, [])
                libraries.assert_not_called()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "job.json"
                env = {"PYTHONPATH": "original"}
                linked = {"LD_LIBRARY_PATH": "libs", "WM_STEGA_CUDA_LIBRARIES": "{}"}
                cmd = ["python", "-m", "wmbench_digital.runner", "decode", "--settings", str(path)]
                for method in ("stegastamp", "rosteals"):
                    write_json(path, {"model": {"id": method}})
                    command, updated = fix.cuda.redirect(cmd, env, linked)
                    if method == "stegastamp":
                        self.assertEqual(command[2], "stegastamp_paired_hotfix")
                        self.assertEqual(updated["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                    else:
                        self.assertEqual(command, cmd)
                        self.assertIs(updated, env)
            self.assertEqual(sha256(worker.__file__), fix.cuda.WORKER_SHA)
            self.assertEqual(sha256(runner.__file__), fix.cuda.RUNNER_SHA)
        finally:
            (runner.run, runner.invoke, runner.remove_packages, storage.remove_packages,
             colab_canary.monitored_process, fix.cuda.redirect, fix.cuda.cuda_record, fix.cuda.quarantine_legacy) = old


class NumericalProfile(unittest.TestCase):
    def exercise(self, provider="CUDAExecutionProvider", options=None):
        session = Mock()
        session.get_providers.return_value = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
        session.get_provider_options.return_value = {"CUDAExecutionProvider": options or {"use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC"}, "CPUExecutionProvider": {}}
        session.get_inputs.return_value = [types.SimpleNamespace(name=n) for n in ("image", "secret")]
        session.get_outputs.return_value = [types.SimpleNamespace(name=n) for n in ("stegastamp", "residual", "decoded")]
        ort = types.SimpleNamespace(__version__="test", SessionOptions=types.SimpleNamespace,
                                    ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                                    InferenceSession=Mock(return_value=session))
        old_load, old_runtime = StegaStampONNX._load, worker.runtime_info
        try:
            with tempfile.TemporaryDirectory() as directory:
                weight = Path(directory) / "weights.onnx"
                weight.write_bytes(b"test-file-not-a-real-model")
                linked = json.dumps({"packages": {"nvidia-cudnn-cu12": "9.x-test"}, "libraries": {"cudnn": {"sha256": "test"}}})
                with patch.dict(sys.modules, {"onnxruntime": ort}), patch.dict(os.environ, {"WM_STEGA_CUDA_LIBRARIES": linked}), \
                     patch.object(fix.subprocess, "run", return_value=types.SimpleNamespace(stdout="A100,driver-test")), \
                     patch.object(worker, "runtime_info", return_value={"python": "test"}):
                    fix.install_numerical_profile()
                    adapter = StegaStampONNX({"device": "cuda:0", "weights": str(weight)})
                    adapter.initialize()
                    self.assertIs(adapter._load(), session)
                    ort.InferenceSession.assert_called_once()
                    opts = ort.InferenceSession.call_args.kwargs["sess_options"]
                    self.assertTrue(opts.use_deterministic_compute)
                    self.assertEqual(opts.intra_op_num_threads, 1)
                    self.assertEqual(opts.inter_op_num_threads, 1)
                    providers = ort.InferenceSession.call_args.kwargs["providers"]
                    self.assertEqual(providers[0][1]["use_tf32"], "0")
                    self.assertEqual(providers[0][1]["cudnn_conv_algo_search"], "HEURISTIC")
                    evidence = fix.execution_profile(worker.runtime_info())
                    self.assertEqual(evidence["identity"]["onnx_sha256"], sha256(weight))
                    self.assertTrue(session.disable_fallback.called)
        finally:
            StegaStampONNX._load, worker.runtime_info = old_load, old_runtime

    def test_fixed_cuda_options_and_runtime_receipt(self):
        self.exercise()

    def test_real_loader_path_cannot_accept_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "CPU-only fallback"):
            self.exercise(provider="CPUExecutionProvider")

    def test_rejected_numerical_options_stop_before_output(self):
        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            self.exercise(options={"use_tf32": "1", "cudnn_conv_algo_search": "EXHAUSTIVE"})


if __name__ == "__main__":
    unittest.main()
