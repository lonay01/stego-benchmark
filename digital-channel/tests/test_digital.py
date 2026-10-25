"""Offline contract, resume, aggregation and saved-pixel tests."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
import contextlib
import types
import csv
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError
from wmbench_bank.contracts import models_for_run, message
from wmbench_digital.spec import ATTACKS, BY_ID, digest, seed
from wmbench_digital.attacks import apply, png_bytes
from wmbench_digital.storage import read_result, result_path, save_result, complete, remove_packages
from wmbench_digital.packages import prepare_packages
from wmbench_digital.worker import decode, measured
from wmbench_digital.report import method_report, comparison, interval

PROJECT = Path(os.environ["WM_DIGITAL_TEST_RELEASE"])


class Attacks(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(1).integers(30, 220, (128, 128, 3), dtype=np.uint8)

    def test_exact_matrix(self):
        from collections import Counter
        self.assertEqual(Counter(a["family"] for a in ATTACKS),
                         dict(clean=1, jpeg=4, noise=20, blur=4, brightness=4, crop=15, resize=3, rotation=8, compound=20))
        self.assertEqual(len({a["id"] for a in ATTACKS}), 79)

    def test_all_attacks_preserve_geometry_and_source(self):
        original = self.image.copy()
        for attack in ATTACKS:
            value, params = apply(self.image, attack, 139, 1)
            self.assertEqual(value.shape, self.image.shape)
            self.assertEqual(value.dtype, np.uint8)
            self.assertEqual(params["seed"], seed(139, 1, attack["id"]))
        np.testing.assert_array_equal(self.image, original)

    def test_deterministic_and_native_sizes(self):
        for size in (128, 224, 256, 400, 512):
            image = np.full((size, size, 3), 127, np.uint8)
            a = BY_ID["compound_2_r3"]
            x, px = apply(image, a, 139, 5)
            y, py = apply(image, a, 139, 5)
            self.assertEqual(png_bytes(x), png_bytes(y))
            self.assertEqual(px, py)
            self.assertEqual(x.shape, image.shape)

    def test_common_noise_and_parameters(self):
        large, _ = apply(np.full((512, 512, 3), 127, np.uint8), BY_ID["noise_0.01_r1"], 139, 1)
        small, _ = apply(np.full((128, 128, 3), 127, np.uint8), BY_ID["noise_0.01_r1"], 139, 1)
        np.testing.assert_array_equal(small, large[192:320, 192:320])
        self.assertAlmostEqual(float(large.std()/255), .01, delta=.0002)

    def test_crop_retains_area_not_side(self):
        _, params = apply(self.image, BY_ID["crop_0.5_r1"], 1, 1)
        self.assertAlmostEqual(params["actual_area"], .5, delta=.01)
        self.assertGreater(params["actual_area"], .4)

    def test_blur_does_not_mix_channels(self):
        image = np.zeros_like(self.image)
        image[:, :, 0] = 255
        value, _ = apply(image, BY_ID["blur_2_r1"], 1, 1)
        np.testing.assert_array_equal(value, image)

    def test_compound_order(self):
        from wmbench_digital import attacks
        original = attacks.resize
        calls = []
        def log_resize(image, scale):
            calls.append(scale)
            return original(image, scale)
        with patch.object(attacks, "resize", log_resize):
            _, p = attacks.apply(self.image, BY_ID["compound_4_r2"], 20, 2)
        self.assertEqual(calls, [.25])
        self.assertTrue(.6 <= p["brightness"] <= 1.4)
        self.assertTrue(-15 <= p["angle"] <= 15)
        self.assertEqual(p["jpeg"], 30)

    def test_contracts_29_unchanged(self):
        models = models_for_run(PROJECT / "production")
        self.assertEqual(len(models), 29)
        self.assertEqual(models["meta_hidden_revisited"]["native_geometry"]["width"], 256)
        self.assertEqual(models["fin"]["native_geometry"]["width"], 128)
        self.assertEqual(models["sepwater"]["native_geometry"]["width"], 224)
        for model in models.values():
            assignment = message(model, 139, 1)
            self.assertEqual(len(assignment["payload"]), model["native_bits"])
            self.assertEqual(assignment, message(model, 139, 1))


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.bank, self.scratch = [self.base / p for p in ("results", "bank", "scratch")]
        for p in (self.root, self.bank, self.scratch):
            p.mkdir()
        self.method = "test_method"
        assignment = dict(payload=[0, 1], capacity=dict(information_indices=[0, 1], reserved_bits={}, information_bits=2, interface="raw"))
        image = np.full((128, 128, 3), 128, np.uint8)
        Image.fromarray(image).save(self.bank/"carrier.png")
        Image.fromarray(image-1).save(self.bank/"crop.png")
        write_json(self.bank/"decoder.json", {"expected_bits": [0, 0]})
        clean = dict(status="measured", decoded_bits=[0, 1], ber=0., exact=True)
        self.case = dict(method=self.method, id="000000000139_m01_native", key="test-key", signature="12345678"*8,
            image_id=139, replicate=1, mode="native", output="carrier.png", source_crop="crop.png", decoder_metadata="decoder.json",
            carrier_sha256=sha256(self.bank/"carrier.png"), assignment=assignment, clean=clean,
            size=[128, 128], psnr=48.1308036086791, matching=None, embedding_seconds=1.)
        write_json(self.root/self.method/"cases.json", [self.case])
        write_json(self.root/self.method/"coverage.json", [dict(method=self.method, mode="native", image_id=139, replicate=1, status="saved")])
        write_json(self.root/"selection.json", {"n_images": 1, "models": [self.method]})
        self.job = dict(results=str(self.root), output=str(self.root/"_runtime"), bank=str(self.bank),
            scratch=str(self.scratch), slot=str(self.scratch/"slots"/self.method), model={"id": self.method},
            identity="identity", deadline=time.time()+3600, session="test", settings={"ssd_reserve_gib": 0})

    def tearDown(self):
        self.temp.cleanup()

    def packages(self):
        prepare_packages(self.job)
        return read_json(self.root/self.method/"packages"/(self.case["id"]+".json"))

    def test_packages_resume_and_regenerate(self):
        before = sha256(self.bank/"carrier.png")
        first = self.packages()
        self.assertEqual(len(first["entries"]), 78)
        with zipfile.ZipFile(first["archive"]) as archive:
            self.assertEqual(len(archive.namelist()), 79)
            self.assertTrue(all(n.startswith(self.method+"/native/") or n == "manifest.json" for n in archive.namelist()))
        second = self.packages()
        self.assertEqual(first, second)
        Path(first["archive"]).unlink()  # Simulate lost temporary runtime disk.
        regenerated = self.packages()
        self.assertEqual(first["sha256"], regenerated["sha256"])
        self.assertEqual(before, sha256(self.bank/"carrier.png"))

    def test_decode_commit_report_cleanup_roundtrip(self):
        manifest = self.packages()
        class Adapter:
            calls = 0
            def decode(self, png, metadata):
                self.calls += 1
                self_outer.assertEqual(read_json(metadata)["expected_bits"], [0, 0])
                with Image.open(png) as image:
                    self_outer.assertEqual(image.size, (128, 128))
                return {"bits": [0, 1]}
        self_outer, adapter = self, Adapter()
        with patch("wmbench_digital.worker.load_adapter", return_value=adapter):
            decode(self.job)
            self.assertEqual(adapter.calls, 79)
            decode(self.job)
            self.assertEqual(adapter.calls, 79)
        result = read_result(result_path(self.root, self.case), self.case)
        self.assertTrue(complete(result))
        counts = method_report(self.root, self.method)
        self.assertEqual(counts["measured"], 79)
        self.assertEqual(counts["complete_carriers"], 1)
        remove_packages(self.root, [self.case], [manifest], [self.scratch, self.root/"_temporary"], "identity", self.method)
        self.assertFalse(Path(manifest["archive"]).exists())
        self.assertTrue((self.bank/"carrier.png").exists())
        self.assertTrue(result_path(self.root, self.case).exists())

    def test_cleanup_requires_committed_results(self):
        manifest = self.packages()
        with self.assertRaises(EvidenceError):
            remove_packages(self.root, [self.case], [manifest], [self.scratch], "identity", self.method)
        self.assertTrue(Path(manifest["archive"]).exists())

    def test_cleanup_rejects_foreign_location(self):
        value = dict(case_key=self.case["key"], rows={a["id"]: {"status": "measured"} for a in ATTACKS})
        save_result(result_path(self.root, self.case), value)
        with self.assertRaises(EvidenceError):
            remove_packages(self.root, [self.case], [{"archive": str(self.bank/"carrier.png")}],
                            [self.scratch], "identity", self.method)
        self.assertTrue((self.bank/"carrier.png").exists())

    def test_corrupted_zip_stops_resume(self):
        manifest = self.packages()
        Path(manifest["archive"]).write_bytes(b"broken")
        with self.assertRaises(EvidenceError):
            self.packages()

    def test_fallback_drive(self):
        self.job["settings"]["ssd_reserve_gib"] = 10**9
        manifest = self.packages()
        self.assertTrue(Path(manifest["archive"]).is_relative_to(self.root/"_temporary"))

    def test_clean_drift_stops_before_attack(self):
        self.packages()
        class Adapter:
            calls = 0
            def decode(self, *_):
                self.calls += 1
                return {"bits": [1, 1]}
        adapter = Adapter()
        with patch("wmbench_digital.worker.load_adapter", return_value=adapter), self.assertRaises(EvidenceError):
            decode(self.job)
        self.assertEqual(adapter.calls, 1)
        self.assertFalse(result_path(self.root, self.case).exists())

    def test_ecc_rejection_without_bits(self):
        class Adapter:
            def decode(self, *_):
                return {"bits": None, "details": {"ecc_valid": False}}
        row = measured(Adapter(), "unused", "unused", self.case, 1)
        self.assertEqual(row["status"], "rejected")
        self.assertIsNone(row["ber"])
        self.assertFalse(row["exact"])

    def test_quality_plumbing_actual_psnr_ssim_stubbed_lpips(self):
        import torch
        import lpips
        from wmbench_digital.quality import run
        class Metric(torch.nn.Module):
            def forward(self, x, y):
                if x.shape != (1, 3, 128, 128) or x.min() < -1 or x.max() > 1:
                    raise ValueError("Wrong metric input domain")
                return (x-y).abs().mean()
        job = {**self.job, "settings": {**self.job["settings"], "device": "cpu"}}
        with patch.object(lpips, "LPIPS", return_value=Metric()):
            run(job)
        quality = read_json(self.root/self.method/"quality"/(self.case["id"]+".json"))
        self.assertAlmostEqual(quality["psnr"], self.case["psnr"], places=6)
        self.assertTrue(.99 < quality["ssim"] < 1.)
        self.assertTrue(0 < quality["lpips"] < .01)

    def test_malformed_bits_are_error_not_ber(self):
        class Adapter:
            def decode(self, *_):
                return {"bits": [1]}
        row = measured(Adapter(), "unused", "unused", self.case, 1)
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["ber"])

    def test_wrong_case_checkpoint_rejected(self):
        save_result(result_path(self.root, self.case), {"case_key": "another", "rows": {}})
        with self.assertRaises(EvidenceError):
            self.packages()

    def test_checkpoint_content_checksum(self):
        path = result_path(self.root, self.case)
        save_result(path, {"case_key": self.case["key"], "rows": {}})
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        envelope["record"]["case_key"] = "modified"
        path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        with self.assertRaises(EvidenceError):
            read_result(path)

    def test_controller_stage_order_and_commit(self):
        from wmbench_digital import runner
        from wmbench import colab_canary
        model = {"id": self.method}
        ctx = dict(identity="identity", bank=self.bank, output=self.root, scratch=self.scratch,
                   models={self.method: model}, selected=[self.method], images=[{"image_id": 139}])
        settings = dict(output=str(self.root), max_hours=1, ssd_reserve_gib=60, pairwise=False)
        stages = []
        @contextlib.contextmanager
        def lease(_):
            yield lambda: None
        class Adapter:
            def decode(self, *_):
                return {"bits": [0, 1]}
        def invoke(job, stage, *_):
            stages.append(stage)
            if stage == "attacks":
                job["settings"]["ssd_reserve_gib"] = 0
                prepare_packages(job)
            elif stage == "quality":
                write_json(self.root/self.method/"quality"/(self.case["id"]+".json"),
                           {"case_key": self.case["key"], "ssim": 1., "lpips": 0.})
            elif stage == "decode":
                with patch("wmbench_digital.worker.load_adapter", return_value=Adapter()):
                    decode(job)
        with patch.object(runner, "os", types.SimpleNamespace(name="posix")), \
             patch.object(runner, "context", return_value=ctx), \
             patch.object(runner, "collect", return_value=[self.case]), \
             patch.object(runner, "invoke", side_effect=invoke), \
             patch.object(colab_canary, "controller_lease", lease):
            runner.run(PROJECT/"production", settings)
        self.assertEqual(stages, ["attacks", "quality", "prepare", "decode"])
        self.assertEqual(read_json(self.root/self.method/"status.json")["status"], "complete")
        manifest = read_json(self.root/self.method/"packages"/(self.case["id"]+".json"))
        self.assertFalse(Path(manifest["archive"]).exists())

    def test_invoke_isolated_decoder_python(self):
        from wmbench_digital import runner
        from wmbench import colab_canary
        job = {**self.job, "project": str(PROJECT/"production")}
        prepared = Path(job["output"])/self.method/"_meta/prepared.json"
        write_json(prepared, {"status": "ready", "python": "/isolated/python"})
        with patch.object(runner, "stage_environment", return_value={}), \
             patch.object(colab_canary, "monitored_process", return_value={"exit_code": 0, "reason": None}) as monitor:
            runner.invoke(job, "decode", self.root/"job.json", lambda: None)
        self.assertIn("/isolated/python", monitor.call_args.args[0])
        self.assertIn("wmbench_digital.runner", monitor.call_args.args[0])
        self.assertIn("production", monitor.call_args.args[1]["PYTHONPATH"])

    def test_image_balanced_aggregation_and_missing_macro(self):
        cases = []
        for image_id, replicate, ber in [(139, 1, 0.), (139, 2, 0.), (141, 1, 1.)]:
            case = {**self.case, "image_id": image_id, "replicate": replicate,
                    "id": f"{image_id}_{replicate}_native", "key": f"{image_id}_{replicate}"}
            rows = {a["id"]: {"status": "measured", "ber": 0. if a["family"] == "clean" else ber,
                              "exact": a["family"] == "clean" or ber == 0., "seconds": 1.}
                    for a in ATTACKS}
            save_result(result_path(self.root, case), {"case_key": case["key"], "rows": rows})
            cases.append(case)
        write_json(self.root/self.method/"cases.json", cases)
        write_json(self.root/"selection.json", {"n_images": 2, "models": [self.method]})
        method_report(self.root, self.method)
        with (self.root/self.method/"tables/attack_summary.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        jpeg = next(r for r in rows if r["family"] == "jpeg")
        self.assertEqual(float(jpeg["ber_percent"]), 50.)  # not 33.3% over messages
        self.assertEqual(float(jpeg["delta_ber_pp"]), 50.)
        path = result_path(self.root, cases[-1])
        partial = read_result(path)
        del partial["rows"]["jpeg_90_r1"]
        save_result(path, partial)
        method_report(self.root, self.method)
        with (self.root/self.method/"tables/macro_summary.csv").open(newline="") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(int(row["ber_percent_images"]), 1)


class BankValidation(unittest.TestCase):
    def test_real_production_record_contract_and_readonly_collection(self):
        from wmbench_digital.bank import context, collect
        from wmbench_bank.contracts import POLICY, MODES
        from wmbench_bank.worker import signature
        from wmbench_full_payload.worker import record_path, task_id
        from wmbench_full_payload.storage import sealed
        from wmbench.io import digest as bank_digest
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            bank, output, scratch = [base/p for p in ("bank", "results", "scratch")]
            bank.mkdir()
            models = models_for_run(PROJECT/"production")
            model = models["fin"]
            image = {"image_id": 139, "path": "_inputs/images/000000000139.png", "sha256": "source-hash"}
            data = {"content_hash": "dataset", "images": [image, {**image, "image_id": 141}]}
            write_json(bank/"_inputs/manifest.json", data)
            write_json(bank/"trial.json", {"models": models, "policy": POLICY, "dataset": "dataset",
                "release": bank_digest(read_json(PROJECT/"production/release.json"))})
            crop = bank/"crop.png"
            Image.new("RGB", (128, 128)).save(crop)
            write_json(bank/"decoder.json", {})
            for rep in range(1, 6):
                stem, sig = task_id(image, rep), signature(model, image, rep)
                assignment = message(model, image["image_id"], rep)
                for mode in MODES:
                    row = dict(status="saved", assignment=assignment, output="crop.png", source_crop="crop.png",
                               decoder_metadata="decoder.json", psnr="Infinity", seconds=0.)
                    sealed(bank, record_path(bank, model, stem, mode), sig, row, [crop, bank/"decoder.json"])
                    sealed(bank, record_path(bank, model, stem, mode+"_decode"), sig,
                           dict(status="measured", ber=0., exact=True, decoded_bits=assignment["payload"]), [crop, bank/"decoder.json"])
            before = {p.relative_to(bank).as_posix(): sha256(p) for p in bank.rglob("*") if p.is_file()}
            settings = dict(bank=str(bank), output=str(output), scratch=str(scratch), n_images=1, models=["fin"])
            ctx = context(PROJECT/"production", settings)
            larger = context(PROJECT/"production", {**settings, "n_images": 2})
            self.assertEqual(ctx["identity"], larger["identity"])
            self.assertEqual(ctx["images"], larger["images"][:1])
            context(PROJECT/"production", settings)
            cases = collect(ctx, model)
            self.assertEqual(len(cases), 20)
            after = {p.relative_to(bank).as_posix(): sha256(p) for p in bank.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            crop.write_bytes(b"corrupt")
            with self.assertRaises(EvidenceError):
                collect(ctx, model)


class Statistics(unittest.TestCase):
    def test_negative_delta_preserved(self):
        row = comparison(dict(status="measured", ber=.1, exact=False), dict(status="measured", ber=.2, exact=False))
        self.assertAlmostEqual(row["delta_ber_pp"], -10)

    def test_rejection_has_exact_loss_but_not_ber(self):
        row = comparison(dict(status="rejected", ber=None, exact=False), dict(status="measured", ber=0, exact=True))
        self.assertIsNone(row["delta_ber_pp"])
        self.assertEqual(row["exact_loss_pp"], 100)

    def test_missing_not_failure_or_success(self):
        row = comparison(dict(status="missing"), dict(status="measured", ber=0, exact=True))
        self.assertIsNone(row["ber_percent"])
        self.assertIsNone(row["exact_percent"])

    def test_bootstrap_fixed_image_units(self):
        self.assertEqual(interval([0, 1, 2, 4]), interval([0, 1, 2, 4]))
        self.assertEqual(interval([10]), (10, 10, 10))
        self.assertEqual(interval([]), (None, None, None))


@unittest.skipUnless(os.environ.get("WM_DIGITAL_META_FIXTURE"), "Real Meta checkpoint fixture not supplied")
class RealMeta(unittest.TestCase):
    def test_checkpoint_saved_attacks_blind_decode(self):
        import torch
        from wmbench.adapters import meta_hidden
        from hidden_fnns_reference import meta_source_classes
        torch.set_num_threads(2)
        fixture = Path(os.environ["WM_DIGITAL_META_FIXTURE"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source, target in [("models.py", "hidden/models.py"), ("attenuations.py", "hidden/attenuations.py"),
                                   ("hidden_replicate.pth", "hidden/ckpts/hidden_replicate.pth")]:
                path = root/target
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fixture/source, path)
            with patch.object(meta_hidden, "load_classes", meta_source_classes):
                adapter = meta_hidden.MetaHiddenAdapter({"asset_root": str(root), "device": "cpu"})
            with Image.open(fixture/"demo.png") as img:
                img.convert("RGB").resize((256, 256)).save(root/"input.png")
            payload = np.random.default_rng(3).integers(0, 2, 48).tolist()
            adapter.embed(str(root/"input.png"), str(root/"wm.png"), payload, str(root/"metadata.json"))
            write_json(root/"decoder.json", {"expected_bits": [0]*48})
            image = np.array(Image.open(root/"wm.png"))
            outcomes = []
            for attack in (ATTACKS[0], BY_ID["jpeg_70_r1"], BY_ID["rotation_5_r1"], BY_ID["compound_1_r1"]):
                out, params = apply(image, attack, 139, 1)
                (root/"attack.png").write_bytes(png_bytes(out))
                bits = adapter.decode(str(root/"attack.png"), str(root/"decoder.json"))["bits"]
                self.assertEqual(len(bits), 48)
                outcomes.append({"attack": attack["id"], "errors": sum(a != b for a, b in zip(payload, bits)), "bits": 48})
            print("REAL_META_SMOKE", json.dumps(outcomes), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
