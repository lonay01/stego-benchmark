"""Transport/restart regressions against the actual immutable release.

Tiny synthetic bytes stand in for the 218 MiB checkpoint in offline tests.
The separate live audit only downloads and hashes the official weights.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import neural_videoseal_download_fix as fix
import neural_sdxl_identity_fix as identity_fix
import test_identity_fix
from wmbench.embedding_bank import inference_identity
from wmbench.io import read_json, write_json
from wmbench_neural import runner


DATA = b"synthetic checkpoint bytes: not real weights"
SHA = hashlib.sha256(DATA).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data=DATA, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}


class TransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="videoseal-transport-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank = self.root / "bank"
        self.results = self.root / "results"
        self.slot = self.root / "scratch/slots/videoseal"
        self.model = dict(id="videoseal", native_bits=256,
            source=dict(revision=fix.WIBE_REVISION), options=dict(model_card="unchanged.yaml"))
        self.identity = dict(registry=self.model, weights={fix.CHECKPOINT_KEY: SHA})
        self.identity_path = self.bank / "videoseal/_meta/model_identity.json"
        write_json(self.identity_path, self.identity)
        self.job = dict(model=self.model, bank=str(self.bank), results=str(self.results),
            slot=str(self.slot), session="12345", deadline=time.time() + 120)
        self.cache = self.results / "_repairs" / fix.VERSION / "assets" / SHA / "y_256b_img.pth"
        self.local = self.slot / "assets/videoseal" / fix.CHECKPOINT_KEY
        self.bank_before = self.identity_path.read_bytes()
        checksum = patch.object(fix, "CHECKPOINT_SHA256", SHA)
        checksum.start()
        self.addCleanup(checksum.stop)
        self.results.mkdir(parents=True)
        for name in ("neural-run.json", "_execution/sdxl.json", "videoseal/records/old.json",
                     "videoseal/packages/pending.zip", "_runtime/videoseal/_meta/prepared.json"):
            path = self.results / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing evidence: " + name.encode())
        self.old = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}

    def tearDown(self):
        self.assertEqual(self.identity_path.read_bytes(), self.bank_before)
        for path, content in self.old.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def put(self, path, content=DATA):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_direct_download_then_new_slot_resume_without_network(self):
        with patch.object(fix.urllib.request, "urlopen", return_value=Response()) as network:
            self.assertEqual(fix.ensure_videoseal_checkpoint(self.job), self.local)
        self.assertEqual(network.call_args.args[0].full_url, fix.CHECKPOINT_URL)
        self.assertNotIn("context", network.call_args.kwargs)  # No insecure TLS context.
        self.assertEqual(self.cache.read_bytes(), DATA)
        self.assertEqual(self.local.read_bytes(), DATA)
        receipt = self.results / "_repairs" / fix.VERSION / "receipts/12345.json"
        before = receipt.read_bytes()
        with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("network used")):
            fix.ensure_videoseal_checkpoint(self.job)  # Same-session receipt is idempotent.
            resumed = fix.ensure_videoseal_checkpoint({**self.job, "session": "67890",
                "slot": str(self.root / "fresh-runtime/slots/videoseal")})
        self.assertEqual(receipt.read_bytes(), before)
        self.assertEqual(resumed.read_bytes(), DATA)

    def test_valid_local_checkpoint_seeds_durable_cache(self):
        self.put(self.local)
        with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("network used")):
            fix.ensure_videoseal_checkpoint(self.job)
        self.assertEqual(self.cache.read_bytes(), DATA)

    def test_other_methods_are_noops_without_io_or_network(self):
        with patch.object(fix, "expected_checkpoint", side_effect=AssertionError("bank accessed")):
            for name in ("wam", "pixelseal", "chunkyseal", "stegastamp"):
                self.assertIsNone(fix.ensure_videoseal_checkpoint({"model": {"id": name}}))

    def test_conflicting_checkpoint_is_preserved_not_replaced(self):
        for path in (self.local, self.cache):
            with self.subTest(path=path):
                self.put(path, b"different bytes")
                with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("network used")), \
                        self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    fix.ensure_videoseal_checkpoint(self.job)
                self.assertEqual(path.read_bytes(), b"different bytes")
                path.unlink()  # Test-owned fixture, not user data.

    def test_wrong_saved_weights_or_registry_are_refused(self):
        for field in ("weights", "registry"):
            wrong = copy.deepcopy(self.identity)
            if field == "weights":
                wrong[field][fix.CHECKPOINT_KEY] = "0" * 64
            else:
                wrong[field]["options"]["model_card"] = "other.yaml"
            write_json(self.identity_path, wrong)
            with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("network used")), \
                    self.assertRaisesRegex(ValueError, "refused"):
                fix.ensure_videoseal_checkpoint(self.job)
        self.identity_path.write_bytes(self.bank_before)

    def test_missing_bank_identity_is_not_inferred_from_runtime(self):
        self.identity_path.unlink()
        with self.assertRaises(FileNotFoundError):
            fix.ensure_videoseal_checkpoint(self.job)
        self.identity_path.write_bytes(self.bank_before)

    def test_session_path_escape_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, "session"):
            fix.ensure_videoseal_checkpoint({**self.job, "session": "../../escape"})
        self.assertFalse((self.results / "_repairs").exists())

    def test_transient_timeout_and_short_response_retry_then_success(self):
        responses = [urllib.error.URLError("timed out"), Response(b"partial", length=99), Response()]
        with patch.object(fix.urllib.request, "urlopen", side_effect=responses) as network:
            fix.fetch_checkpoint(self.cache, SHA, self.job["deadline"])
        self.assertEqual(network.call_count, 3)
        self.assertEqual([call.args[0].full_url for call in network.call_args_list],
                         [fix.CHECKPOINT_URL, fix.MIRROR_URL, fix.MIRROR_URL])
        self.assertEqual(self.cache.read_bytes(), DATA)
        self.assertFalse(list(self.cache.parent.glob("*.part")))

    def test_mirror_bytes_still_require_exact_checkpoint_hash(self):
        with patch.object(fix.urllib.request, "urlopen", side_effect=[
                TimeoutError("Meta unreachable"), Response(b"wrong mirror bytes")]) as network, \
                self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            fix.fetch_checkpoint(self.cache, SHA, self.job["deadline"])
        self.assertEqual(network.call_count, 2)
        self.assertFalse(self.cache.exists())

    def test_network_failure_is_bounded_and_keeps_no_partial_final(self):
        with patch.object(fix.urllib.request, "urlopen", side_effect=TimeoutError("timeout")) as network, \
                self.assertRaisesRegex(RuntimeError, "No weights or results were substituted"):
            fix.fetch_checkpoint(self.cache, SHA, self.job["deadline"])
        self.assertEqual(network.call_count, 3)
        self.assertFalse(self.cache.exists())
        self.assertFalse(list(self.cache.parent.iterdir()))

    def test_wrong_complete_download_is_not_retried_or_published(self):
        with patch.object(fix.urllib.request, "urlopen", return_value=Response(b"wrong weights")) as network, \
                self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            fix.fetch_checkpoint(self.cache, SHA, self.job["deadline"])
        self.assertEqual(network.call_count, 1)
        self.assertFalse(self.cache.exists())
        self.assertFalse(list(self.cache.parent.iterdir()))

    def test_expired_deadline_never_starts_download(self):
        with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("network used")), \
                self.assertRaisesRegex(TimeoutError, "Session budget"):
            fix.fetch_checkpoint(self.cache, SHA, time.time() - 1)

    def test_old_archive_not_needed_for_frozen_inference_identity(self):
        previous = {**self.identity, "weights": {**self.identity["weights"],
            "downloads/videoseal.zip": "historical transport hash"}}
        self.assertEqual(inference_identity(previous), inference_identity(self.identity))
        # Unknown files/checkpoint changes are still scientific differences.
        changed = {**self.identity, "weights": {**self.identity["weights"], "unknown.pth": "other"}}
        self.assertNotEqual(inference_identity(previous), inference_identity(changed))
        changed["weights"] = {fix.CHECKPOINT_KEY: "other"}
        self.assertNotEqual(inference_identity(previous), inference_identity(changed))

    def test_actual_frozen_adapter_skips_nextcloud_for_preseeded_checkpoint(self):
        from wmbench.adapters.legacy import WibeAdapter
        from wmbench.adapters import legacy

        self.put(self.local)
        assets = self.slot / "assets/videoseal"
        for name in ("resources", "submodules", "src"):
            (assets / name).mkdir()
        modules = {name: types.ModuleType(name) for name in (
            "wibench", "wibench.download", "wibench.watermark_data",
            "test_wibe_wrapper", "videoseal", "videoseal.utils", "videoseal.utils.cfg")}
        modules["wibench"].download = modules["wibench.download"]
        modules["wibench.watermark_data"].TorchBitWatermarkData = object
        wrapper = modules["test_wibe_wrapper"]
        wrapper.VideosealWrapper = object
        wrapper.REQUIRED_FILES_VIDEOSEAL = ["y_256b_img.pth"]
        wrapper.URL_VIDEOSEAL = "https://nextcloud.invalid/not-used"
        seal = MagicMock()
        setup = modules["videoseal.utils.cfg"].setup_model_from_model_card = MagicMock(return_value=seal)
        options = dict(model_id="videoseal", asset_root=str(assets), repo_root=str(self.root / "source"),
            wrapper="test_wibe_wrapper:VideosealWrapper", model_card="original-card.yaml", device="cpu")
        cwd, paths = Path.cwd(), list(sys.path)
        try:
            with patch.dict(sys.modules, modules), \
                    patch.object(legacy, "download", side_effect=AssertionError("Nextcloud used")):
                adapter = WibeAdapter(options)
        finally:
            os.chdir(cwd)
            sys.path[:] = paths
        setup.assert_called_once_with(Path("original-card.yaml"))
        self.assertIs(adapter.seal, seal)
        seal.eval.return_value.to.assert_called_once_with("cpu")


class CombinedOverlayTests(test_identity_fix.OverlayIntegration):
    """Also rerun the old package-resume and worker-dispatch smoke tests."""

    def call(self, action, config):
        with patch.dict(os.environ, {fix.HASH_ENV: fix.file_sha256(fix.__file__),
                identity_fix.HASH_ENV: identity_fix.module_sha256()}), \
                patch.object(sys, "argv", ["repair", action, "--settings", str(config)]):
            fix.main()

    def test_controller_uses_overlay_for_actual_invoke_command(self):
        from wmbench import colab_canary
        config = self.base / "controller-settings.json"
        write_json(config, dict(output=str(self.root)))
        job = {**self.job, "engine": "sdxl", "project": str(self.base / "project")}

        def delegated_run(project, settings):
            runner.invoke(job, "generate", lambda: None)
            return True

        with patch.object(runner, "run", side_effect=delegated_run), \
                patch.object(colab_canary, "monitored_process", return_value=dict(exit_code=0, reason=None)) as monitor:
            self.call("run", config)
        command, env = monitor.call_args.args[:2]
        self.assertIn(fix.ENTRYPOINT, command)
        self.assertNotIn("wmbench_neural.runner", command)
        self.assertEqual(env[fix.HASH_ENV], fix.file_sha256(fix.__file__))
        self.assertEqual(env[identity_fix.HASH_ENV], identity_fix.module_sha256())

    def test_prepare_intercepts_only_videoseal_then_calls_original_stage(self):
        config = self.base / "prepare-job.json"
        for name in ("videoseal", "wam", "chunkyseal"):
            job = {**self.job, "model": {"id": name}}
            write_json(config, job)
            events = []
            with patch.object(fix, "expected_checkpoint", side_effect=lambda job: events.append("videoseal")
                    or (_ for _ in ()).throw(ValueError("probe"))), \
                    patch.object(runner, "prepare", side_effect=lambda job: events.append("original")):
                if name == "videoseal":
                    job["session"] = "123"
                    write_json(config, job)
                    with self.assertRaisesRegex(ValueError, "probe"):
                        self.call("prepare", config)
                    self.assertEqual(events, ["videoseal"])
                else:
                    self.call("prepare", config)
                    self.assertEqual(events, ["original"])

    def test_both_checksum_guards_fail_before_any_work(self):
        for name in (fix.HASH_ENV, identity_fix.HASH_ENV):
            with patch.dict(os.environ, {fix.HASH_ENV: fix.file_sha256(fix.__file__),
                    identity_fix.HASH_ENV: identity_fix.module_sha256(), name: "wrong"}), \
                    patch.object(sys, "argv", ["repair", "prepare", "--settings", "missing.json"]), \
                    self.assertRaisesRegex(RuntimeError, "checksum|hash"):
                fix.main()


class SavedBankContractTests(unittest.TestCase):
    def test_actual_bank_registry_matches_unchanged_distributed_catalogue(self):
        from wmbench_bank.contracts import models_for_run
        project = Path(runner.__file__).resolve().parents[2] / "production"
        saved = read_json(Path(__file__).parent / "fixtures/videoseal_bank_identity.json")
        self.assertEqual(saved["registry"], models_for_run(project)["videoseal"])
        self.assertEqual(saved["weights"][fix.CHECKPOINT_KEY], fix.CHECKPOINT_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
