"""CPU contract tests; synthetic operators are never experiment results."""

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError, freeze
from wmbench_full_payload.contracts import message, score
from wmbench_neural import bank, channel, decode, generate, report, storage
from wmbench_neural.spec import ATTACKS, POLICY, seed, digest, batches


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="neural-fixture-")
        self.base = Path(self.directory.name)
        self.bank, self.root = self.base / "bank", self.base / "output"
        self.bank.mkdir()
        self.root.mkdir()
        self.model = dict(id="synthetic", native_bits=48)
        self.cases = []
        for image_id in (139, 285):
            for rep in (1, 2):
                stem = f"{image_id:012d}_m{rep:02d}_native"
                assignment = message(self.model, image_id, rep)
                png = self.bank / (stem+".png")
                array = np.random.default_rng(image_id+rep).integers(0, 256, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(array).save(png)
                metadata = self.bank / (stem+".json")
                write_json(metadata, dict(payload=[0]*48))
                case = dict(id=stem, method="synthetic", image_id=image_id, replicate=rep, mode="native",
                    key=digest(stem), signature=digest(stem), output=png.name, carrier_sha256=sha256(png),
                    source_crop=png.name, decoder_metadata=metadata.name, assignment=assignment,
                    clean=score(dict(bits=assignment["payload"]), assignment), psnr=42.0, matching=None,
                    embedding_seconds=1.0, size=[128, 128])
                self.cases.append(case)
        self.job = dict(model=self.model, results=str(self.root), bank=str(self.bank), identity="fixture-v1",
            slot=str(self.base / "slot"), scratch=str(self.base / "scratch"), output=str(self.root / "_runtime"),
            session="test", deadline=time.time()+600, settings=dict(gpu_batch_size=3), engine="vae")
        write_json(self.root / "synthetic/cases.json", self.cases)
        write_json(self.root / "synthetic/attack-cases.json", [{k: c[k] for k in
            ("id", "key", "method", "image_id", "replicate", "mode", "output", "carrier_sha256", "size")} for c in self.cases])
        write_json(self.root / "selection.json", dict(models=["synthetic"], n_images=2))
        write_json(self.root / "synthetic/coverage.json", [dict(mode="native", image_id=c["image_id"], status="saved") for c in self.cases])
        write_json(self.root / "_inputs/captions.json", dict(captions={str(i): {"text": "official test caption"} for i in (139, 285)}))

    def tearDown(self):
        self.directory.cleanup()

    def fake_measured(self, adapter, png, metadata, case, rng):
        self.assertTrue(Path(png).is_file())
        self.assertFalse(any(read_json(metadata)["payload"]))
        with Image.open(png) as image:
            self.assertEqual(list(image.size), case["size"])
        return {**score(dict(bits=case["assignment"]["payload"]), case["assignment"]), "seconds": .01}

    def prime_controls(self):
        generate.prepare_geometry(self.job)
        with patch.object(decode, "load", return_value=object()), patch.object(decode.worker, "measured", side_effect=self.fake_measured), patch.object(decode.worker, "runtime_info", return_value={"fixture": True}):
            decode.controls(self.job)

    def pack(self, attack=ATTACKS[1]):
        cases = self.cases[:2]
        inputs = generate.load_images(self.bank, cases)
        outputs = [np.asarray(i) for i in inputs]
        generate.persist(self.job, cases, attack, inputs, outputs,
            [dict(amortized_attack_seconds=.1)]*2, [0.0]*2, {"synthetic": True})
        return cases, read_json(storage.manifest_path(self.root, "synthetic", attack, cases[0]["image_id"]))


class Contracts(unittest.TestCase):
    def test_all_exposed_bits_scored_not_common_thirty(self):
        assignment = message(dict(id="synthetic", native_bits=90), 139, 1)
        bits = list(assignment["payload"])
        bits[89] ^= 1
        row = score(dict(bits=bits), assignment)
        self.assertAlmostEqual(row["ber"], 1/90)
        self.assertFalse(row["exact"])
    def test_matrix(self):
        self.assertEqual(len(ATTACKS), 21)
        self.assertEqual(len({a["id"] for a in ATTACKS}), 21)
        self.assertEqual(Counter(a["family"] for a in ATTACKS), dict(clean=1, vae=2, diffusion=9, instruction=6, inpaint=3))

    def test_seed_excludes_method_mode_and_n(self):
        a = dict(image_id=139, replicate=2, method="a", mode="native", n=5)
        b = dict(image_id=139, replicate=2, method="b", mode="matched_45", n=100)
        self.assertEqual([seed(a, attack) for attack in ATTACKS], [seed(b, attack) for attack in ATTACKS])
        self.assertNotEqual(seed(a, ATTACKS[1]), seed({**a, "replicate": 3}, ATTACKS[1]))
        self.assertEqual(len({seed(a, attack) for attack in ATTACKS}), 21)

    def test_fixed_padding(self):
        self.assertEqual(list(batches([1, 2, 3, 4, 5], 4)), [([1, 2, 3, 4], 4), ([5, 5, 5, 5], 1)])
        self.assertEqual(list(batches([], 4)), [])

    def test_caption_minimum_annotation(self):
        value = dict(annotations=[dict(id=11, image_id=139, caption="later"), dict(id=3, image_id=139, caption="first")])
        result = bank.choose_captions(value, {139})
        self.assertEqual(result["139"]["text"], "first")
        with self.assertRaises(ValueError):
            bank.choose_captions(value, {139, 42})

    def test_policy_roundtrips_json(self):
        self.assertEqual(json.loads(json.dumps(POLICY)), POLICY)

    def test_geometry_and_quantization(self):
        im = Image.new("RGB", (128, 128), (127, 0, 255))
        self.assertEqual(channel.grid(im).size, (512, 512))
        self.assertEqual(channel.grid(channel.grid(im), (128, 128)).getpixel((0, 0)), (127, 0, 255))
        np.testing.assert_array_equal(channel.rgb8(np.array([-.2, .5, 1.2])), np.array([0, 128, 255], dtype=np.uint8))
        with self.assertRaises(ValueError):
            channel.rgb8([np.nan])

    def test_mask_area_and_quality(self):
        a = np.zeros((512, 512, 3), dtype=np.uint8)
        b = a.copy()
        b[128:384, 128:384] = 40
        row = generate.pixel_quality(a, b)
        self.assertAlmostEqual(row["mae_inside_mask"], 40)
        self.assertEqual(row["mae_outside_mask"], 0)
        self.assertAlmostEqual(row["psnr"], 10*np.log10(255**2/400))


class StorageTests(Fixture):
    def test_real_bank_contract_extends_five_to_hundred(self):
        from wmbench.io import digest as bank_digest
        from wmbench_bank.contracts import models_for_run, POLICY as BANK_POLICY
        project = Path(__file__).resolve().parents[1] / "production"
        models, release = models_for_run(project), read_json(project / "release.json")
        self.assertEqual(len(models), 29)
        self.assertIn("meta_hidden_revisited", models)
        manifest = dict(content_hash="synthetic-fixed-corpus", images=[dict(image_id=i) for i in range(100)])
        write_json(self.bank / "_inputs/manifest.json", manifest)
        write_json(self.bank / "trial.json", dict(policy=BANK_POLICY, models=models, release=bank_digest(release), dataset=manifest["content_hash"]))
        settings = {**self.job["settings"], "models": "all", "bank": str(self.bank), "output": str(self.root),
                    "scratch": str(self.base / "scratch"), "n_images": 5}
        five = bank.context(project, settings)
        hundred = bank.context(project, {**settings, "n_images": 100})
        self.assertEqual(five["identity"], hundred["identity"])
        self.assertEqual(five["images"], hundred["images"][:5])
        with self.assertRaises(EvidenceError):
            bank.context(project, {**settings, "gpu_batch_size": 4})

    def test_sealed_record_and_key_mismatch(self):
        case = self.cases[0]
        path = storage.result_path(self.root, case)
        record = dict(case_key=case["key"], rows={})
        storage.save_result(path, record)
        self.assertEqual(storage.read_result(path, case), record)
        with self.assertRaises(EvidenceError):
            storage.read_result(path, {**case, "key": "wrong"})

    def test_corrupted_record_fails_not_recomputed(self):
        case = self.cases[0]
        path = storage.result_path(self.root, case)
        storage.save_result(path, dict(case_key=case["key"], rows={}))
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        envelope["record"]["rows"]["clean"] = {"status": "measured"}
        path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        with self.assertRaises(EvidenceError):
            storage.read_result(path, case)

    def test_cache_resume_and_corruption(self):
        cases, manifest = self.pack()
        self.assertTrue(storage.cached(self.job, cases, ATTACKS[1]))
        (self.root / manifest["archive"]).write_bytes(b"corrupt")
        with self.assertRaises(EvidenceError):
            storage.cached(self.job, cases, ATTACKS[1])

    def test_deletion_requires_all_committed_rows(self):
        cases, manifest = self.pack()
        with self.assertRaises(EvidenceError):
            storage.remove_committed_package(self.job, cases, ATTACKS[1])
        self.assertTrue((self.root / manifest["archive"]).exists())
        for case in cases:
            storage.save_result(storage.result_path(self.root, case), dict(case_key=case["key"],
                rows={ATTACKS[1]["id"]: dict(status="measured", package_key=manifest["key"])}))
        storage.remove_committed_package(self.job, cases, ATTACKS[1])
        self.assertFalse((self.root / manifest["archive"]).exists())
        self.assertTrue((self.bank / cases[0]["output"]).exists())

    def test_foreign_path_cannot_be_deleted(self):
        cases, manifest = self.pack()
        for case in cases:
            storage.save_result(storage.result_path(self.root, case), dict(case_key=case["key"],
                rows={ATTACKS[1]["id"]: dict(status="measured", package_key=manifest["key"])}))
        path = storage.manifest_path(self.root, "synthetic", ATTACKS[1], cases[0]["image_id"])
        write_json(path, {**manifest, "archive": str(self.bank / cases[0]["output"])})
        with self.assertRaises(EvidenceError):
            storage.remove_committed_package(self.job, cases, ATTACKS[1])

    def test_regeneration_must_match_bytes(self):
        cases, manifest = self.pack()
        altered = {"x.png": b"different"}
        with self.assertRaises(EvidenceError):
            storage.save_package(self.job, cases, ATTACKS[1], manifest["entries"], altered)

    def test_local_carrier_cache_reuses_exact_png(self):
        cache = self.base / "inputs-cache"
        expected = generate.load_images(self.bank, self.cases[:1], cache)
        path = cache / (self.cases[0]["id"]+".png")
        self.assertEqual(sha256(path), self.cases[0]["carrier_sha256"])
        # Remove only this test fixture's source; the hash-verified cache remains usable.
        (self.bank / self.cases[0]["output"]).unlink()
        actual = generate.load_images(self.bank, self.cases[:1], cache)
        np.testing.assert_array_equal(np.asarray(expected[0]), np.asarray(actual[0]))
        path.write_bytes(b"bad cache")
        with self.assertRaises(EvidenceError):
            generate.load_images(self.bank, self.cases[:1], cache)

    def test_package_paths_survive_result_root_move(self):
        import shutil
        cases, manifest = self.pack()
        self.assertFalse(Path(manifest["archive"]).is_absolute())
        other = self.base / "copied-output"
        shutil.copytree(self.root, other)
        self.assertTrue(storage.cached({**self.job, "results": str(other)}, cases, ATTACKS[1]))

    def test_freeze_rejects_changes(self):
        path = self.root / "spec.json"
        freeze(path, POLICY)
        freeze(path, POLICY)
        with self.assertRaises(EvidenceError):
            freeze(path, {**POLICY, "version": "other"})

    def test_control_and_resize_roundtrip(self):
        self.prime_controls()
        for case in self.cases:
            record = storage.read_result(storage.result_path(self.root, case), case)
            self.assertEqual(record["rows"]["clean"]["ber"], 0)
            self.assertEqual(record["resize_only"]["ber"], 0)
            self.assertFalse(storage.complete(record))

    def test_clean_drift_stops_before_attacks(self):
        with patch.object(decode, "load", return_value=object()), patch.object(decode.worker, "measured", return_value=dict(status="measured", decoded_bits=[1]*48, ber=.5, exact=False)), patch.object(decode.worker, "runtime_info", return_value={}):
            with self.assertRaises(EvidenceError):
                decode.controls(self.job)

    def test_reference_bits_rejected_in_metadata(self):
        case = self.cases[0]
        write_json(self.bank / case["decoder_metadata"], dict(payload=[1]*48))
        with self.assertRaises(EvidenceError):
            decode.blind_metadata(self.job, case)

    def test_macro_does_not_hide_blocked(self):
        self.prime_controls()
        for case in self.cases:
            path = storage.result_path(self.root, case)
            record = storage.read_result(path, case)
            for a in ATTACKS[1:]:
                record["rows"][a["id"]] = dict(status="measured", ber=.1, exact=False, seconds=.1)
            record["rows"]["inpaint_r1"] = dict(status="blocked_generation", ber=None, exact=None)
            storage.save_result(path, record)
        counts = report.method_report(self.root, "synthetic")
        self.assertEqual(counts["blocked_generation"], 4)
        self.assertEqual(counts["complete_carriers"], 4)
        with (self.root / "synthetic/tables/macro_summary.csv").open(newline="") as f:
            macro = list(csv.DictReader(f))[0]
        self.assertEqual(macro["ber_percent"], "")

    def test_macro_weights_four_families_equally(self):
        self.prime_controls()
        for case in self.cases:
            path = storage.result_path(self.root, case)
            record = storage.read_result(path, case)
            for a in ATTACKS[1:]:
                record["rows"][a["id"]] = dict(status="measured",
                    ber={"vae": .1, "diffusion": .2, "instruction": .3, "inpaint": .4}[a["family"]],
                    exact=False, seconds=.1)
            storage.save_result(path, record)
        report.method_report(self.root, "synthetic")
        with (self.root / "synthetic/tables/macro_summary.csv").open(newline="") as f:
            macro = list(csv.DictReader(f))[0]
        self.assertAlmostEqual(float(macro["ber_percent"]), 25)
        self.assertAlmostEqual(float(macro["delta_ber_pp"]), 25)
        self.assertAlmostEqual(float(macro["bit_accuracy_percent"]), 75)


class FakeChannel:
    calls = 0
    def __init__(self, engine, cache):
        self.identity = dict(synthetic=True, engine=engine)
        self.engine = engine
    def apply(self, images, attack, seeds, captions):
        type(self).calls += 1
        arrays = [(np.asarray(im).astype(np.uint16)+np.random.default_rng(s).integers(0, 5)) % 256 for im, s in zip(images, seeds)]
        arrays = [a.astype(np.uint8) for a in arrays]
        flags = [False]*len(arrays)
        if attack["id"] == "ip2p_style_r3":
            arrays[0], flags[0] = None, True
        return arrays, flags, dict(batch_seconds=.01, amortized_attack_seconds=.01/len(images),
                                   actual_denoising_steps=0 if self.engine == "vae" else 3)


class FakeMetric:
    identity = dict(synthetic=True)
    def compare(self, a, b):
        return [float(np.mean(np.abs(x.astype(float)-y.astype(float)))/255) for x, y in zip(a, b)]


class Integration(Fixture):
    def test_all_conditions_resume_and_cleanup(self):
        self.prime_controls()
        before = {p.name: sha256(p) for p in self.bank.iterdir()}
        with patch.object(generate, "NeuralChannel", FakeChannel), patch.object(generate, "PerceptualMetric", FakeMetric), patch.object(decode, "load", return_value=object()), patch.object(decode.worker, "measured", side_effect=self.fake_measured), patch.object(decode.worker, "runtime_info", return_value={"fixture": True}):
            for engine in ("vae", "sdxl", "ip2p", "inpaint"):
                job = {**self.job, "engine": engine}
                generate.run(job)
                calls = FakeChannel.calls
                generate.run(job)  # Undecoded ZIPs reused, no inference.
                self.assertEqual(FakeChannel.calls, calls)
                decode.run(job)
                generate.run(job)  # Deleted ZIPs with durable rows are not regenerated.
                self.assertEqual(FakeChannel.calls, calls)
        for case in self.cases:
            self.assertTrue(storage.complete(storage.read_result(storage.result_path(self.root, case), case)))
        self.assertFalse(list((self.root / "_temporary").rglob("*.zip")))
        self.assertEqual(before, {p.name: sha256(p) for p in self.bank.iterdir()})
        counts = report.method_report(self.root, "synthetic")
        report.final_report(self.root, pairwise=True)
        self.assertEqual(counts["complete_carriers"], 4)
        self.assertEqual(sum(counts.get(k, 0) for k in storage.TERMINAL), 4*21)
        self.assertGreater(counts["blocked_generation"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
