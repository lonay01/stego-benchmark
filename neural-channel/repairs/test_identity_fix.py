"""Recovery-only regressions; all artifacts are synthetic and temporary."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import neural_sdxl_identity_fix as fix
from test_neural import Fixture, FakeChannel, FakeMetric
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import freeze, EvidenceError
from wmbench_neural import generate, runner, storage
from wmbench_neural.spec import ATTACKS, digest


def identity():
    return dict(engine="sdxl", revision="fixed-checkpoint", torch_gpu="fixed-gpu",
        libraries={"diffusers": "0.32.2"}, grid=[512, 512],
        checkpoint_files=[{"name": "model.safetensors", "sha256": "frozen"}],
        scheduler_config={"beta_start": .00085, "beta_end": .012,
            "_use_default_values": ["sigma_max", "sigma_min", "use_beta_sigmas"]})


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="neural-identity-repair-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "_execution/sdxl.json"
        self.saved = identity()
        freeze(self.path, self.saved)
        self.original = self.path.read_bytes()

    def test_exact_match_is_noop(self):
        self.assertEqual(fix.reconcile_identity(self.root, "sdxl", self.saved), self.saved)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse((self.root / "_repairs").exists())

    def test_permutation_preserves_legacy_file_hash_and_digest(self):
        observed = copy.deepcopy(self.saved)
        observed["scheduler_config"]["_use_default_values"].reverse()
        original_observed = copy.deepcopy(observed)
        result = fix.reconcile_identity(self.root, "sdxl", observed)
        self.assertEqual(result, self.saved)
        self.assertEqual(observed, original_observed)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(digest(result), digest(self.saved))
        receipt_path, = (self.root / "_repairs" / fix.VERSION).glob("sdxl-*.json")
        receipt = read_json(receipt_path)
        self.assertEqual(receipt["preserved_file_sha256"], sha256(self.path))
        self.assertEqual(receipt["observed_identity"], observed)
        fix.reconcile_identity(self.root, "sdxl", observed)
        self.assertEqual(len(list(receipt_path.parent.glob("*.json"))), 1)

    def test_real_parameter_change_reports_path_and_fails_closed(self):
        observed = copy.deepcopy(self.saved)
        observed["scheduler_config"]["beta_end"] = .013
        with self.assertRaisesRegex(EvidenceError, "scheduler_config.beta_end"):
            fix.reconcile_identity(self.root, "sdxl", observed)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse((self.root / "_repairs").exists())

    def test_other_execution_fields_remain_strict(self):
        for key, value in (("torch_gpu", "other"), ("revision", "changed"),
                           ("libraries", {"diffusers": "0.33.0"}),
                           ("checkpoint_files", [{"sha256": "other"}]),
                           ("grid", [256, 512])):
            with self.subTest(key=key), self.assertRaisesRegex(EvidenceError, key):
                fix.reconcile_identity(self.root, "sdxl", {**self.saved, key: value})
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_changed_membership_and_missing_field_rejected(self):
        for value in ([], ["different"]):
            observed = copy.deepcopy(self.saved)
            observed["scheduler_config"]["_use_default_values"] = value
            with self.assertRaises(EvidenceError):
                fix.reconcile_identity(self.root, "sdxl", observed)
        observed["scheduler_config"].pop("_use_default_values")
        with self.assertRaises(EvidenceError):
            fix.reconcile_identity(self.root, "sdxl", observed)

    def test_malformed_members_are_rejected(self):
        for value in (["duplicate", "duplicate"], [123], "not-a-list", None):
            observed = copy.deepcopy(self.saved)
            observed["scheduler_config"]["_use_default_values"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fix.reconcile_identity(self.root, "sdxl", observed)

    def test_other_lists_are_not_normalized(self):
        self.saved["schedule_order"] = [1, 2, 3]
        write_json(self.path, self.saved)
        with self.assertRaisesRegex(EvidenceError, "schedule_order"):
            fix.reconcile_identity(self.root, "sdxl", {**self.saved, "schedule_order": [3, 2, 1]})

    def test_new_identity_is_canonical_without_writing_execution(self):
        new = fix.reconcile_identity(self.root, "inpaint", self.saved)
        self.assertEqual(new["scheduler_config"]["_use_default_values"],
                         sorted(self.saved["scheduler_config"]["_use_default_values"]))
        self.assertFalse((self.root / "_execution/inpaint.json").exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_original_freeze_still_rejects_unapproved_changes(self):
        with self.assertRaises(EvidenceError):
            freeze(self.path, {**self.saved, "revision": "other"})

    def test_real_diffusers_hashseed_permutation_has_identical_timesteps(self):
        probe = '''import json, diffusers
from diffusers import EulerDiscreteScheduler
assert diffusers.__version__ == "0.32.2"
s = EulerDiscreteScheduler.from_config(dict(num_train_timesteps=1000,
    beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
    prediction_type="epsilon", interpolation_type="linear", use_karras_sigmas=False,
    timestep_spacing="leading", steps_offset=1))
config = dict(s.config)
s.set_timesteps(20)
print(json.dumps(dict(scheduler_config=config,
    timesteps=s.timesteps.tolist(), sigmas=s.sigmas.tolist())))
'''
        values = []
        for seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "HF_HUB_OFFLINE": "1"}
            result = subprocess.run([sys.executable, "-c", probe], env=env,
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
            values.append(json.loads(result.stdout.strip().splitlines()[-1]))
        self.assertNotEqual(values[0]["scheduler_config"][fix.DEFAULT_KEYS],
                            values[1]["scheduler_config"][fix.DEFAULT_KEYS])
        self.assertEqual(fix.canonical_identity(values[0]), fix.canonical_identity(values[1]))
        path = self.root / "_execution/real-scheduler.json"
        freeze(path, values[0])
        self.assertEqual(fix.reconcile_identity(self.root, "real-scheduler", values[1]), values[0])


class DispatchTests(unittest.TestCase):
    def test_child_wrapper_keeps_process_guard_and_propagates_hash(self):
        command = ["controller", "-m", "wmbench.process_guard", "--", "isolated-python",
                   "-m", "wmbench_neural.runner", "decode", "--settings", "job.json"]
        env = {"PYTHONPATH": "frozen-src"}
        result, result_env = fix.redirect_child(command, env, "expected")
        self.assertEqual(result[:6], command[:6])
        self.assertEqual(result[6], fix.ENTRYPOINT)
        self.assertEqual(result[7:], command[7:])
        self.assertEqual(result_env[fix.HASH_ENV], "expected")
        self.assertNotIn(fix.HASH_ENV, env)
        self.assertEqual(command[6], "wmbench_neural.runner")

    def test_unrelated_command_is_unchanged(self):
        command, env = ["python", "-m", "other.module"], {"original": "value"}
        result, result_env = fix.redirect_child(command, env, "expected")
        self.assertIs(result, command)
        self.assertIs(result_env, env)

    def test_missing_or_wrong_hash_stops_before_delegate(self):
        for checksum in ("", "wrong"):
            with patch.dict(os.environ, {fix.HASH_ENV: checksum}), \
                    patch.object(sys, "argv", ["repair", "run", "--settings", "missing.json"]), \
                    self.assertRaisesRegex(RuntimeError, "checksum"):
                fix.main()


class OverlayIntegration(Fixture):
    def call(self, action, config):
        with patch.dict(os.environ, {fix.HASH_ENV: fix.module_sha256()}), \
                patch.object(sys, "argv", ["repair", action, "--settings", str(config)]):
            fix.main()

    def test_actual_generate_freeze_package_resume_preserves_old_evidence(self):
        saved = identity()
        freeze(self.root / "_execution/sdxl.json", saved)
        old_bytes = (self.root / "_execution/sdxl.json").read_bytes()
        job = {**self.job, "engine": "sdxl"}
        config = self.base / "job.json"
        write_json(config, job)

        class LoadedChannel(FakeChannel):
            def __init__(self, engine, cache):
                super().__init__(engine, cache)
                self.identity = copy.deepcopy(saved)
                self.identity["scheduler_config"][fix.DEFAULT_KEYS].reverse()

        before_bank = {p.name: sha256(p) for p in self.bank.iterdir()}
        with patch.object(generate, "NeuralChannel", LoadedChannel), \
                patch.object(generate, "PerceptualMetric", FakeMetric):
            self.call("generate", config)
            calls = LoadedChannel.calls
            manifests = []
            for attack in ATTACKS:
                if attack["engine"] == "sdxl":
                    for group in (self.cases[:2], self.cases[2:]):
                        path = storage.manifest_path(self.root, "synthetic", attack, group[0]["image_id"])
                        manifests.append((path, sha256(path)))
                        for entry in read_json(path)["entries"].values():
                            self.assertEqual(entry["execution"]["identity_sha256"], digest(saved))
            self.call("generate", config)
            self.assertEqual(LoadedChannel.calls, calls)
            self.assertEqual(manifests, [(p, sha256(p)) for p, _ in manifests])
        self.assertEqual((self.root / "_execution/sdxl.json").read_bytes(), old_bytes)
        self.assertEqual(before_bank, {p.name: sha256(p) for p in self.bank.iterdir()})

    def test_controller_uses_overlay_for_actual_invoke_command(self):
        from wmbench import colab_canary
        config = self.base / "controller-settings.json"
        write_json(config, dict(output=str(self.root)))
        job = {**self.job, "engine": "sdxl", "project": str(self.base / "project")}

        def delegated_run(project, settings):
            runner.invoke(job, "generate", lambda: None)
            return True

        with patch.object(runner, "run", side_effect=delegated_run), \
                patch.object(colab_canary, "monitored_process",
                    return_value=dict(exit_code=0, reason=None)) as monitor:
            self.call("run", config)
        command, env = monitor.call_args.args[:2]
        self.assertIn(fix.ENTRYPOINT, command)
        self.assertNotIn("wmbench_neural.runner", command)
        self.assertEqual(env[fix.HASH_ENV], fix.module_sha256())


if __name__ == "__main__":
    unittest.main(verbosity=2)
