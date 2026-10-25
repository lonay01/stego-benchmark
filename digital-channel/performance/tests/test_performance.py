"""CPU contract tests. These do not claim real CUDA-model validation."""

import copy
import contextlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch

from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_digital.storage import save_result, read_result, result_path, complete
from wmbench_digital.spec import ATTACKS
from wmbench_fast import cache
from wmbench_fast.runner import finished, mark_finished, TABLES, progress_report, verify_release
from wmbench_fast.scheduler import shards, can_admit
import test_digital as legacy


class Performance(unittest.TestCase):
    def setUp(self):
        legacy.Pipeline.setUp(self)
        self.job["settings"].update(attack_workers=2, decode_workers=2, full_audit=False)
        self.case["clean"]["artifacts"] = [dict(path=n, sha256=sha256(self.bank/n))
                                              for n in ("carrier.png", "crop.png", "decoder.json")]
        write_json(self.root/self.method/"cases.json", [self.case])

    def tearDown(self):
        legacy.Pipeline.tearDown(self)

    def document(self):
        return dict(case_key=self.case["key"], rows={a["id"]: dict(self.case["clean"]) for a in ATTACKS})

    def test_cached_resume_no_raw_reads(self):
        path = result_path(self.root, self.case)
        save_result(path, self.document())
        index = cache.ResultCache(self.root, self.method, [self.case])
        index.warm()
        with patch.object(cache, "RAW_READ", side_effect=AssertionError("Unexpected raw read")):
            resumed = cache.ResultCache(self.root, self.method, [self.case])
            self.assertTrue(complete(resumed.read(path, self.case)))

    def test_changed_raw_and_missing_index_recovered(self):
        path = result_path(self.root, self.case)
        save_result(path, self.document())
        index = cache.ResultCache(self.root, self.method, [self.case])
        index.warm()
        updated = self.document()
        updated["rows"].pop(ATTACKS[-1]["id"])
        save_result(path, updated)
        resumed = cache.ResultCache(self.root, self.method, [self.case])
        self.assertFalse(complete(resumed.read(path, self.case)))
        path.unlink()  # Only this temporary synthetic test result.
        resumed = cache.ResultCache(self.root, self.method, [self.case])
        self.assertIsNone(resumed.read(path, self.case))

    def test_full_audit_bypasses_index(self):
        path = result_path(self.root, self.case)
        save_result(path, self.document())
        cache.ResultCache(self.root, self.method, [self.case]).warm()
        with patch.object(cache, "RAW_READ", wraps=read_result) as raw:
            index = cache.ResultCache(self.root, self.method, [self.case], full_audit=True)
            index.read(path, self.case)
            self.assertEqual(raw.call_count, 1)

    def test_foreign_checkpoint_cannot_overwrite_result(self):
        path = result_path(self.root, self.case)
        save_result(path, self.document())
        before = sha256(path)
        index = cache.ResultCache(self.root, self.method, [self.case])
        with self.assertRaises(EvidenceError):
            index.save(path, {"case_key": "foreign", "rows": {}})
        self.assertEqual(sha256(path), before)

    def test_cache_corruption_fails_closed(self):
        p = self.root/"cache.gz"
        cache.pack(p, {"ok": True})
        envelope = json.loads(gzip.decompress(p.read_bytes()))
        envelope["value"]["ok"] = False
        p.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        with self.assertRaises(EvidenceError):
            cache.unpack(p)

    def test_input_hashes_still_checked(self):
        cache.verify_inputs(self.job, self.case)
        write_json(self.bank/"decoder.json", {"expected_bits": [1, 1]})
        with self.assertRaises(EvidenceError):
            cache.verify_inputs(self.job, self.case)

    def test_receipt_invalidated_by_new_prefix_or_result(self):
        path = result_path(self.root, self.case)
        save_result(path, self.document())
        for name in TABLES:
            target = self.root/self.method/"tables"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture")
        mark_finished(self.root, self.method, [self.case])
        self.assertTrue(finished(self.root, self.method, [self.case]))
        self.assertFalse(finished(self.root, self.method, [self.case], True))
        extra = dict(self.case, id="other", key="other")
        self.assertFalse(finished(self.root, self.method, [self.case, extra]))
        save_result(path, dict(case_key=self.case["key"], rows={}))
        self.assertFalse(finished(self.root, self.method, [self.case]))

    def test_whole_image_shards_disjoint(self):
        cases = [dict(image_id=i, id=f"{i}-{j}") for i in range(7) for j in range(20)]
        groups = shards(cases, 4)
        flattened = [i for g in groups for i in g]
        self.assertEqual(sorted(flattened), list(range(7)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(shards([], 2), [])

    def test_resource_admission(self):
        sample = dict(gpu_gib=2, rss_gib=3)
        self.assertTrue(can_admit(sample, 20, 20))
        self.assertFalse(can_admit(sample, 3, 20))
        self.assertFalse(can_admit(sample, 20, 3))
        self.assertFalse(can_admit(dict(sample, gpu_gib=None), 20, 20))

    def test_progress_does_not_promote_old_canary(self):
        write_json(self.root/self.method/"status.json", {"status": "complete"})
        progress_report(self.root, [self.method], 100, {})
        result = read_json(self.root/"_reports/fast_progress.json")[0]
        self.assertEqual(result["status"], "not_checked_this_session")
        self.assertEqual(result["requested_images"], 100)

    def test_parallel_packages_identical_to_serial(self):
        from wmbench_fast.packages import prepare
        from wmbench_digital.packages import prepare_packages
        prepare_packages(self.job)
        path = self.root/self.method/"packages"/(self.case["id"]+".json")
        serial = read_json(path)
        Path(serial["archive"]).unlink()  # Simulated loss of disposable test ZIP.
        prepare(self.job)
        parallel = read_json(path)
        self.assertEqual(serial["entries"], parallel["entries"])
        self.assertEqual(serial["sha256"], parallel["sha256"])

    def test_two_process_decode_and_resume_smoke(self):
        from wmbench_fast.packages import prepare
        from wmbench_fast.worker import prepare_probes
        cases = [self.case, dict(self.case, id="000000000285_m01_native", image_id=285, key="second-key")]
        write_json(self.root/self.method/"cases.json", cases)
        prepare(self.job)
        inputs = prepare_probes(self.job, self.case)
        jobs = []
        for index, case in enumerate(cases):
            job = dict(self.job, worker_index=index, shard_images=[case["image_id"]],
                       probe_case=self.case, probe_inputs=inputs, session="smoke"+str(index),
                       probe_reference=str(self.scratch/"probe.json"), ready_path=str(self.scratch/f"ready-{index}.json"),
                       cancel_path=str(self.scratch/"cancel.json"))
            path = self.root/f"job{index}.json"
            write_json(path, job)
            jobs.append(path)
        script = Path(__file__).with_name("fake_decode.py")
        first = subprocess.Popen([sys.executable, str(script), str(jobs[0])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+45
        while not (self.scratch/"ready-0.json").exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        second = subprocess.Popen([sys.executable, str(script), str(jobs[1])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for proc in (first, second):
            output, _ = proc.communicate(timeout=90)
            self.assertEqual(proc.returncode, 0, output.decode(errors="replace"))
        before = {}
        for case in cases:
            path = result_path(self.root, case)
            self.assertTrue(complete(read_result(path, case)))
            before[path] = sha256(path)
        for job in jobs:
            proc = subprocess.run([sys.executable, str(script), str(job)], capture_output=True, timeout=45)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
        self.assertEqual(before, {p: sha256(p) for p in before})
        self.assertTrue((Path(self.job["slot"])/"tmp/digital-decode-0/attack.png").exists())
        self.assertTrue((Path(self.job["slot"])/"tmp/digital-decode-1/attack.png").exists())

    def test_frozen_overlay_source_inventory(self):
        self.assertTrue(verify_release().is_dir())

    def test_real_bank_catalogue_reuse_and_full_audit(self):
        from wmbench_digital import bank
        ordinary = bank.collect
        calls = []
        def collect_once(ctx, model):
            cases = ordinary(ctx, model)
            calls.append(len(cases))
            coverage = read_json(ctx["output"]/model["id"]/"coverage.json")
            self.assertIsNotNone(cache.validate_catalogue(cases, coverage, ctx, model))
            with patch.object(cache.original_bank, "collect", side_effect=AssertionError("Old bank was rehashed")):
                reused = cache.collect(ctx, model)
                self.assertEqual(reused, cases)
                self.assertEqual(cache.collect(ctx, model), cases)
            bad = copy.deepcopy(cases)
            bad[0]["assignment"]["payload"][0] ^= 1
            with self.assertRaises(EvidenceError):
                cache.validate_catalogue(bad, coverage, ctx, model)
            return cases
        with patch.object(bank, "collect", side_effect=collect_once):
            legacy.BankValidation().test_real_production_record_contract_and_readonly_collection()
        self.assertEqual(calls, [20])

    def test_fast_controller_completed_method_never_invokes_gpu(self):
        from wmbench_fast import runner, scheduler
        from wmbench_digital import report
        import stegastamp_paired_hotfix
        save_result(result_path(self.root, self.case), self.document())
        for name in TABLES:
            target = self.root/self.method/"tables"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture")
        write_json(self.root/self.method/"tables/counts.json", {"carriers": 1, "complete_carriers": 1})
        mark_finished(self.root, self.method, [self.case])
        ctx = dict(bank=self.bank, output=self.root, scratch=self.scratch, selected=[self.method],
                   models={self.method: {"id": self.method}}, images=[{"image_id": 139}], identity="id")
        settings = dict(self.job["settings"], output=str(self.root), max_hours=1, ssd_reserve_gib=35,
                        n_images=1, pairwise=False)
        with patch.object(runner, "os", types.SimpleNamespace(name="posix")), \
             patch.object(runner, "context", return_value=ctx), patch.object(runner, "collect", return_value=[self.case]), \
             patch.object(stegastamp_paired_hotfix, "install_controller_patch"), \
             patch("wmbench.colab_canary.controller_lease", return_value=contextlib.nullcontext(lambda: None)), \
             patch.object(scheduler, "invoke", side_effect=AssertionError("GPU setup on complete method")), \
             patch.object(report, "method_report", side_effect=AssertionError("Rebuilding complete tables")), \
             patch.object(report, "final_report"):
            self.assertTrue(runner.run(Path("unused"), settings))

    def test_probe_mismatch_and_stegastamp_guard(self):
        from wmbench_fast.worker import probe, prepare_probes, decode
        class Adapter:
            def __init__(self):
                self.calls = 0
            def decode(self, *_):
                self.calls += 1
                return {"bits": [0, 1] if self.calls == 1 else [1, 1]}
        job = dict(self.job, worker_index=1, probe_case=self.case,
                   probe_inputs=prepare_probes(self.job, self.case),
                   probe_reference=str(self.root/"probe.json"), ready_path=str(self.root/"ready.json"))
        clean = {k: self.case["clean"][k] for k in ("status", "decoded_bits", "ber", "exact")}
        write_json(job["probe_reference"], {"case_key": self.case["key"], "rows": [clean]*4})
        with self.assertRaises(EvidenceError):
            probe(Adapter(), job)
        self.assertFalse(Path(job["ready_path"]).exists())
        with self.assertRaises(ValueError):
            decode(dict(job, model={"id": "stegastamp"}))


if __name__ == "__main__":
    unittest.main()
