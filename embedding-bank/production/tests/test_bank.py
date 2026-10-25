"""CPU tests against the extracted deployable release, not an import-only mock."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from PIL import Image

from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError
from wmbench_bank.contracts import EXPECTED, MODES, capacity, message, models_for_run, score
from wmbench_bank.report import report, image_mean
from wmbench_bank.worker import encode_pair, decode_pair, complete

PROJECT = Path(os.environ["WMBENCH_TEST_RELEASE"])


class ToyAdapter:
    """A synthetic storage/retry fixture, never represented as a real method."""
    def __init__(self):
        self.embeds = 0
        self.decodes = 0
        self.payloads = {}
        self.fail = False

    def embed(self, source, output, payload, metadata):
        self.embeds += 1
        key = f"key-{self.embeds}"
        self.payloads[key] = list(payload)
        # Vary the residual to exercise many quantization breakpoints.
        image = load_rgb(source)
        residual = np.arange(image.size).reshape(image.shape) % 21 - 10
        save_rgb(output, np.clip(image.astype(int)+residual, 0, 255).astype(np.uint8))
        write_json(metadata, {"key": key, "expected_bits": payload})

    def decode(self, source, metadata):
        self.decodes += 1
        if self.fail:
            raise RuntimeError("synthetic transient decoder failure")
        meta = read_json(metadata)
        assert not any(meta["expected_bits"]), "Reference leaked into decoder"
        bits = self.payloads[meta["key"]].copy()
        bits[-1] ^= 1  # Nonzero BER must be retained, never retried for improvement.
        return {"bits": bits}


class BankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = models_for_run(PROJECT)
        self.model = self.models["meta_hidden_revisited"]
        source = self.root / "_inputs/images/000000000139.png"
        array = np.random.default_rng(8).integers(20, 220, (512, 512, 3), dtype=np.uint8)
        save_rgb(source, array)
        self.image = {"image_id": 139, "path": source.relative_to(self.root).as_posix(), "sha256": sha256(source)}

    def tearDown(self):
        self.temp.cleanup()

    def test_catalogue_and_selected_profiles(self):
        self.assertEqual(set(self.models), EXPECTED)
        self.assertEqual(len(self.models), 29)
        self.assertEqual(self.models["sepwater"]["native_geometry"]["width"], 224)
        self.assertEqual(self.models["rosteals"]["engine"], "author")
        self.assertEqual(self.models["distseal_posthoc_dcae"]["engine"], "author")
        self.assertEqual(self.model["native_geometry"]["width"], 256)
        self.assertEqual(capacity(self.models["invismark"])["information_bits"], 94)
        self.assertEqual(capacity(self.models["fnns_omniseal"])["information_bits"], 32)
        self.assertEqual(self.models["editguard"]["validated_lock"], "locks/editguard.lock")

    def test_messages_are_length_prefixes_and_full_capacity(self):
        a = message(self.model, 139, 1)
        b = message(self.models["trustmark_q"], 139, 1)
        self.assertEqual(a["information"], b["information"][:48])
        self.assertEqual(message(self.model, 139, 1), a)
        self.assertNotEqual(a, message(self.model, 139, 2))
        bits = a["payload"].copy()
        bits[-1] ^= 1
        result = score({"bits": bits}, a)
        self.assertEqual(result["information_errors"], 1)
        self.assertAlmostEqual(result["ber"], 1/48)

    def test_roundtrip_modes_and_resume_nonzero_ber(self):
        adapter = ToyAdapter()
        for replicate in range(1, 6):
            encode_pair(adapter, self.model, self.root, self.image, replicate, {})
            decode_pair(adapter, self.model, self.root, self.image, replicate, {})
        self.assertTrue(complete(self.root, self.model, [self.image]))
        count = adapter.embeds, adapter.decodes
        native = self.root / self.model["id"] / "native/000000000139_m01.png"
        before = sha256(native)
        for replicate in range(1, 6):
            encode_pair(adapter, self.model, self.root, self.image, replicate, {})
            decode_pair(adapter, self.model, self.root, self.image, replicate, {})
        self.assertEqual(count, (adapter.embeds, adapter.decodes))
        self.assertEqual(sha256(native), before)
        self.assertEqual(load_rgb(native).shape, (256, 256, 3))

    def test_decoder_failure_retry_does_not_embed(self):
        adapter = ToyAdapter()
        encode_pair(adapter, self.model, self.root, self.image, 1, {})
        adapter.fail = True
        self.assertGreater(decode_pair(adapter, self.model, self.root, self.image, 1, {}), 0)
        adapter.fail = False
        self.assertEqual(decode_pair(adapter, self.model, self.root, self.image, 1, {}), 0)
        self.assertEqual(adapter.embeds, 1)
        self.assertTrue(list((self.root/self.model["id"] / "_meta/records/technical_error_history").glob("*.json")))

    def test_corrupt_png_stops_not_overwrites(self):
        adapter = ToyAdapter()
        encode_pair(adapter, self.model, self.root, self.image, 1, {})
        output = self.root / self.model["id"] / "native/000000000139_m01.png"
        output.write_bytes(b"damaged")
        with self.assertRaises(EvidenceError):
            encode_pair(adapter, self.model, self.root, self.image, 1, {})
        self.assertEqual(output.read_bytes(), b"damaged")
        self.assertEqual(adapter.embeds, 1)

    def test_unmatched_is_not_fake_png(self):
        adapter = ToyAdapter()
        def identity(source, output, payload, metadata):
            save_rgb(output, load_rgb(source))
            write_json(metadata, {"expected_bits": payload})
        adapter.embed = identity
        encode_pair(adapter, self.model, self.root, self.image, 1, {})
        for mode in MODES[1:]:
            self.assertFalse((self.root / self.model["id"] / mode / "000000000139_m01.png").exists())
            record = read_json(self.root / self.model["id"] / "_meta/records" / f"000000000139_m01_{mode}.json")
            self.assertEqual(record["status"], "unmatched")

    def test_report_partial_coverage_and_image_weights(self):
        write_json(self.root / "trial.json", {"models": {self.model["id"]: self.model}})
        write_json(self.root / "_inputs/manifest.json", {"images": [self.image]})
        adapter = ToyAdapter()
        encode_pair(adapter, self.model, self.root, self.image, 1, {})
        decode_pair(adapter, self.model, self.root, self.image, 1, {})
        result = report(self.root, {"n_images": 1}, refresh_methods=[self.model["id"]])
        self.assertFalse(result["all_selected_technical_complete"])
        self.assertEqual(result["planned_carriers"], 20)
        self.assertAlmostEqual(result["models"][0]["clean_ber_percent"], 100/48)
        self.assertEqual(image_mean([{"image_id": 1, "x": 0}, {"image_id": 1, "x": 0},
                                    {"image_id": 2, "x": 1}], "x"), .5)
        with patch("wmbench_bank.report.record_path", side_effect=AssertionError("Unnecessary Drive scan")):
            again = report(self.root, {"n_images": 1})
        self.assertEqual(again, result)

    def test_prefix_extension_identity_and_release_tamper(self):
        from wmbench_bank.runner import context
        source = self.root / "source"
        source.mkdir()
        images = []
        for index in range(1, 7):
            target = source / f"_inputs/{index}.png"
            save_rgb(target, np.zeros((512, 512, 3), dtype=np.uint8))
            images.append({"image_id": index, "path": f"_inputs/{index}.png", "sha256": sha256(target)})
        data = {"images": images, "content_hash": "synthetic-test-only"}
        write_json(source / "_inputs/manifest.json", data)
        (source / "_inputs/legacy_manifest.csv").write_text("fixture")
        settings = {"bank": str(source), "output": str(self.root / "out"), "scratch": str(self.root / "scratch"), "n_images": 5}
        with patch("wmbench_bank.runner.freeze_inputs", return_value=data):
            first = context(PROJECT, settings)
            second = context(PROJECT, {**settings, "n_images": 6})
        self.assertEqual(first["identity"], second["identity"])
        self.assertEqual(first["images"], second["images"][:5])
        with self.assertRaises(ValueError):
            context(PROJECT, {**settings, "n_images": 101})

    def test_report_does_not_reuse_completed_smaller_cohort_status(self):
        write_json(self.root / "trial.json", {"models": {self.model["id"]: self.model}})
        write_json(self.root / "_inputs/manifest.json", {"images": [self.image]})
        write_json(self.root / self.model["id"] / "_meta/status.json", {"status": "complete"})
        result = report(self.root, {"n_images": 1})
        self.assertFalse(result["all_selected_technical_complete"])
        self.assertFalse(result["all_targets_available"])
        self.assertEqual(result["models"][0]["status"], "incomplete")

    def test_controller_creates_lease_dir_and_dispatches_prepare_then_work(self):
        import wmbench_bank.runner as runner
        root, scratch = self.root / "run", self.root / "scratch"
        calls = []
        @contextmanager
        def lease(path):
            self.assertTrue(path.is_dir())
            yield lambda: None
        def monitor(command, env, log, slot, seconds, limits, assert_owned=None):
            action = command[command.index("wmbench_bank.runner")+1]
            calls.append(action)
            if action == "prepare":
                write_json(root / self.model["id"] / "_meta/prepared.json", {"status": "ready", "python": "model-python"})
            return {"exit_code": 0, "reason": None}
        ctx = {"output": root, "scratch": scratch, "bank": self.root / "source", "identity": "test",
               "models": {self.model["id"]: self.model}, "selected": [self.model["id"]], "images": [self.image]}
        with patch.object(runner, "os", SimpleNamespace(name="posix")), \
             patch.object(runner, "context", return_value=ctx), \
             patch.object(runner, "complete", side_effect=[False, True]), \
             patch.object(runner, "report"), \
             patch.object(runner, "stage_environment", return_value={}), \
             patch("wmbench.colab_canary.controller_lease", lease), \
             patch("wmbench.colab_canary.monitored_process", monitor), \
             patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=100*1024**3)):
            runner.run(PROJECT, {"output": str(root), "max_hours": 1, "cleanup": False})
        self.assertEqual(calls, ["prepare", "work"])
        self.assertEqual(read_json(root / self.model["id"] / "_meta/status.json")["status"], "complete")

    def test_author_environment_has_actual_import_fixes_and_locks(self):
        text = (PROJECT / "src/authornight/environment.py").read_text()
        self.assertIn('"ImageNet-C" / "imagenet_c"', text)
        self.assertIn('"ImageNet-C/imagenet_c"', text)
        self.assertIn("slackclient==2.9.3", (PROJECT / "authors/locks/rosteals.txt").read_text())
        for method in ("sepwater", "distseal"):
            self.assertTrue((PROJECT / f"authors/locks/{method}.txt").is_file())

    def test_resume_dispatches_only_two_failed_methods_preserving_other_files(self):
        """Synthetic scheduler statuses exercise the 27-complete/2-failed case."""
        import wmbench_bank.runner as runner
        root, scratch = self.root / "run", self.root / "scratch"
        missing = {"rosteals", "sepwater"}
        finished = set(self.models) - missing
        snapshots = {}
        for name in finished:
            target = root / name / "native/retained-evidence.png"
            save_rgb(target, np.zeros((3, 3, 3), dtype=np.uint8))
            snapshots[target] = sha256(target)
            status = root / name / "_meta/status.json"
            write_json(status, {"status": "complete", "session": "historical"})
            snapshots[status] = sha256(status)
        calls = []

        @contextmanager
        def lease(path):
            yield lambda: None

        def monitor(command, env, log, slot, seconds, limits, assert_owned=None):
            job = read_json(Path(command[-1]))
            name = job["model"]["id"]
            action = command[command.index("wmbench_bank.runner") + 1]
            calls.append((name, action))
            self.assertIn(name, missing)
            if action == "prepare":
                write_json(root / name / "_meta/prepared.json", {"status": "ready", "python": "model-python"})
            else:
                finished.add(name)
            return {"exit_code": 0, "reason": None}

        ctx = {"output": root, "scratch": scratch, "bank": self.root / "source", "identity": "test",
               "models": self.models, "selected": list(self.models), "images": [self.image]}
        with patch.object(runner, "os", SimpleNamespace(name="posix")), \
             patch.object(runner, "context", return_value=ctx), \
             patch.object(runner, "complete", side_effect=lambda root, model, images: model["id"] in finished), \
             patch.object(runner, "report"), \
             patch.object(runner, "stage_environment", return_value={}), \
             patch("wmbench.colab_canary.controller_lease", lease), \
             patch("wmbench.colab_canary.monitored_process", monitor), \
             patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
            runner.run(PROJECT, {"output": str(root), "max_hours": 1, "cleanup": False})
        self.assertEqual(calls, [("rosteals", "prepare"), ("rosteals", "work"),
                                 ("sepwater", "prepare"), ("sepwater", "work")])
        for path, expected in snapshots.items():
            self.assertEqual(sha256(path), expected)

    def test_author_bridge_tensor_paths_and_blind_file_decode(self):
        """Synthetic neural cores check adapter wiring, not pretrained accuracy."""
        import torch
        from wmbench_bank.adapters import AuthorAdapter
        from authornight.metrics import pixels
        original_tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original_tensor(*args, **kwargs)
        source = self.root / self.image["path"]
        for name in ("rosteals", "sepwater", "distseal"):
            adapter = AuthorAdapter.__new__(AuthorAdapter)
            adapter.name, adapter.torch = name, torch
            adapter.native_bits = 64 if name == "distseal" else 100
            payload = [i % 2 for i in range(adapter.native_bits)]
            def tensor(image):
                return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].float()/255
            class LatentCore:
                def encode_first_stage(self, image):
                    return image
                def decode_first_stage(self, latent):
                    return latent
                def __call__(self, latent, unused, message):
                    return latent + .01, None
            ref = SimpleNamespace(tensor=tensor,
                decode=lambda x: np.array([1 if b else -1 for b in payload]),
                model=LatentCore(), export=lambda marked, cover, image: pixels(marked, "floor"),
                ae=SimpleNamespace(encode_pre_quant=lambda x: (x, x.shape), decode=lambda z, s: z),
                wm=SimpleNamespace(embedder=lambda z, m: torch.ones_like(z)*.01,
                                   blender=lambda z, residual: z+.5*residual))
            adapter.reference = ref
            output, metadata = self.root / f"{name}.png", self.root / f"{name}.json"
            with patch.object(torch, "tensor", cpu_tensor):
                adapter.embed(str(source), str(output), payload, str(metadata))
            self.assertEqual(load_rgb(output).shape, (512, 512, 3))
            self.assertEqual(adapter.decode(str(output), "nonexistent-metadata.json")["bits"], payload)
            if name == "distseal":
                np.testing.assert_array_equal(adapter.last_reconstruction, load_rgb(source))

    @unittest.skipUnless(os.environ.get("WMBENCH_META_FIXTURES"), "Local pretrained Meta fixtures unavailable")
    def test_real_meta_cpu_production_worker_four_modes(self):
        import torch
        from wmbench.adapters.meta_hidden import ASSET_HASHES, MetaHiddenAdapter
        torch.set_num_threads(4)
        fixture = Path(os.environ["WMBENCH_META_FIXTURES"])
        assets = self.root / "assets"
        for relative in ASSET_HASHES:
            destination = assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture / Path(relative).name, destination)
        with Image.open(fixture / "demo.png") as file:
            save_rgb(self.root / self.image["path"], np.array(file.convert("RGB").resize((512, 512))))
        self.image["sha256"] = sha256(self.root / self.image["path"])
        adapter = MetaHiddenAdapter({"asset_root": str(assets), "device": "cpu"})
        encode_pair(adapter, self.model, self.root, self.image, 1, {"test": "real_cpu"})
        self.assertEqual(decode_pair(adapter, self.model, self.root, self.image, 1, {"test": "real_cpu"}), 0)
        observed = []
        for mode in MODES:
            folder = self.root / self.model["id"] / "_meta/records"
            enc = read_json(folder / f"000000000139_m01_{mode}.json")
            self.assertIn(enc["status"], ("saved", "unmatched"))
            if enc["status"] == "unmatched":
                observed.append({"mode": mode, "status": "unmatched", "matching": enc["matching"]})
                continue
            dec = read_json(folder / f"000000000139_m01_{mode}_decode.json")
            self.assertEqual(len(dec["decoded_bits"]), 48)
            observed.append({"mode": mode, "psnr": enc["psnr"], "ber": dec["ber"]})
        self.assertEqual(observed[0]["mode"], "native")
        print("REAL_META_PRODUCTION_CPU", json.dumps(observed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
