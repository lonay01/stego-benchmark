"""CPU contracts and mocked replay; no real checkpoint quality claims."""

import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Import the same small frozen release shipped to Colab, not the dirty worktree.
TEST_TEMP = ROOT / ".test-scratch"
TEST_TEMP.mkdir(exist_ok=True)
BASE_TEMP = tempfile.TemporaryDirectory(prefix="neural-io-test-base-", dir=TEST_TEMP)
BASE = Path(BASE_TEMP.name)
with zipfile.ZipFile(ROOT.parent / "dist/COCO_Neural_Channel_RESUMABLE_source.zip") as z:
    for entry in z.infolist():
        path = (BASE / entry.filename).resolve()
        if not path.is_relative_to(BASE.resolve()):
            raise ValueError("Unsafe test fixture ZIP")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(z.read(entry))
sys.path.extend([str(BASE / "src"), str(BASE / "production/src"), str(ROOT.parent / "repairs")])

from neural_io_check import common as c, pipeline, decoder, report, runner
from wmbench_full_payload.contracts import score


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="neural-io-test-", dir=TEST_TEMP)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "diagnostic"
        self.bank = self.root / "bank"
        self.old = self.root / "neural"
        self.pixels = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        c.immutable_bytes(self.bank / "native.png", c.png_bytes(self.pixels))
        c.immutable_bytes(self.bank / "source.png", c.png_bytes(np.roll(self.pixels, 2, axis=0)))
        assignment = dict(payload=[0] * 100, information=[0] * 94, capacity=dict(
            information_indices=[i for i in range(100) if i not in (48, 49, 50, 51, 64, 65)],
            reserved_bits={str(i): 0 for i in (48, 49, 50, 51, 64, 65)},
            information_bits=94, interface="raw_message"))
        clean = score({"bits": [0] * 100}, assignment)
        clean.pop("expected_bits", None)
        self.case = dict(id="test_native", key="case_key", method="invismark", image_id=285,
            replicate=1, mode="native", signature="12345678", size=[512, 512],
            carrier_sha256=c.sha(self.bank / "native.png"), output="native.png",
            source_crop="source.png", decoder_metadata="blind.json", assignment=assignment, clean=clean)
        c.write(self.bank / "blind.json", {"expected_bits": [0] * 100})
        c.write(self.out / "invismark/native-cases.json", [self.case])
        c.write(self.out / "invismark/historical/test_native.json", {"rows": {}})
        self.job = dict(results=str(self.out), bank=str(self.bank), neural=str(self.old),
            model={"id": "invismark"}, engine="controls", deadline=1e20, scratch=str(self.root / "scratch"),
            settings={"gpu_batch_size": 4}, session="test")


class Contracts(Fixture):
    def test_normalization_all_256_values(self):
        values = np.tile(np.arange(256, dtype=np.uint8), (512, 2))
        rgb = np.repeat(values[:, :, None], 3, axis=2)
        np.testing.assert_array_equal(c.normalization_roundtrip(rgb), rgb)

    def test_png_roundtrip_preserves_rgb_order(self):
        a = self.pixels.copy()
        a[:, :, 0] = 255
        a[:, :, 2] = 0
        np.testing.assert_array_equal(c.png_pixels(c.png_bytes(a)), a)

    def test_identity_controls_and_negative(self):
        for name in c.CONTROLS[:4]:
            np.testing.assert_array_equal(c.control_pixels(name, self.pixels, 255 - self.pixels), self.pixels)
        np.testing.assert_array_equal(c.control_pixels("unwatermarked", self.pixels, 255-self.pixels), 255-self.pixels)

    def test_wrong_shape_and_dtype_fail(self):
        for array in (self.pixels.astype(float), self.pixels[:128]):
            with self.assertRaises(ValueError):
                c.png_bytes(array)
        with self.assertRaises(ValueError):
            c.pixel_stats(np.array([np.nan]))

    def test_no_overwrite_of_committed_evidence(self):
        p = self.out / "x.json"
        c.commit(p, {"ber": .5})
        c.commit(p, {"ber": .5})
        with self.assertRaises(RuntimeError):
            c.commit(p, {"ber": 0})
        self.assertEqual(c.read(p), {"ber": .5})

    def test_no_overwrite_of_png(self):
        p = self.out / "x.png"
        c.immutable_bytes(p, b"abc")
        with self.assertRaises(RuntimeError):
            c.immutable_bytes(p, b"def")
        self.assertEqual(p.read_bytes(), b"abc")

    def test_disjoint_roots(self):
        with self.assertRaises(ValueError):
            c.separate_paths(self.bank, self.bank / "nested")
        with self.assertRaises(ValueError):
            c.separate_paths(self.bank, self.bank)
        c.separate_paths(self.bank, self.out, self.old, self.root / "scratch")

    def test_path_escape(self):
        with self.assertRaises(ValueError):
            c.inside(self.out, "../bank/native.png")

    def test_reserved_bits_are_not_information_errors(self):
        bits = [0] * 100
        bits[48] = 1
        row = score({"bits": bits}, self.case["assignment"])
        self.assertEqual(row["ber"], 0)
        self.assertEqual(row["reserved_errors"], [48])
        bits[0] = 1
        self.assertEqual(score({"bits": bits}, self.case["assignment"])["ber"], 1 / 94)

    def test_wrong_payload_length_rejected(self):
        with self.assertRaises(ValueError):
            score({"bits": [0] * 94}, self.case["assignment"])

    def test_same_bytes_requires_same_bits(self):
        self.assertFalse(c.compare_history([1], "a", {"image_sha256": "a", "decoded_bits": [0]})["bits_equal"])
        self.assertIsNone(c.compare_history([1], "b", {"image_sha256": "a", "decoded_bits": [0]})["bits_equal"])

    def test_high_ber_is_not_an_error(self):
        rows = [{"status": "measured", "io_agrees": True, "official": {"ber": .5}}]
        self.assertEqual(c.report_state(rows, 1), "IO_AGREEMENT")
        self.assertEqual(c.report_state(rows, 2), "INCOMPLETE")
        rows[0]["io_agrees"] = False
        self.assertEqual(c.report_state(rows, 1), "REVIEW_IO_OR_REPLAY")

    def test_blocked_is_not_ber_zero(self):
        rows = [{"status": "blocked_generation"}]
        self.assertEqual(c.report_state(rows, 1), "IO_AGREEMENT")
        self.assertIsNone(report.image_mean([dict(image_id=1, ber_percent=None)], "ber_percent"))

    def test_image_not_message_weighted(self):
        rows = [dict(image_id=1, v=0)] * 5 + [dict(image_id=2, v=100)]
        self.assertEqual(report.image_mean(rows, "v"), 50)

    def test_controls_resume_and_corruption(self):
        with patch.object(pipeline, "quality", return_value={"psnr": "Infinity", "ssim": 1}):
            pipeline.controls(self.job)
            before = {p: c.sha(p) for p in self.out.rglob("*.png")}
            pipeline.controls(self.job)
        self.assertEqual(len(before), 6)
        self.assertEqual(before, {p: c.sha(p) for p in self.out.rglob("*.png")})
        path = next(iter(before))
        path.write_bytes(b"corrupt")
        with self.assertRaises(RuntimeError):
            pipeline.controls(self.job)

    def test_float_cache_validation(self):
        value = pipeline.save_png(self.out, self.case, "vae_1", self.pixels,
                                 dict(float_path="float.npy", float_sha256=hashlib.sha256(b"ok").hexdigest()))
        c.immutable_bytes(self.out / "float.npy", b"ok")
        self.assertTrue(pipeline.generated_complete(self.out, self.case, "vae_1"))
        (self.out / "float.npy").write_bytes(b"wrong")
        with self.assertRaises(RuntimeError):
            pipeline.generated_complete(self.out, self.case, "vae_1")


class Tensor:
    """Minimal NumPy tensor double; exercises IO contracts, not neural numerics."""
    def __init__(self, value): self.a = np.asarray(value)
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    @property
    def device(self): return "cpu"
    def detach(self): return self
    def cpu(self): return self
    def to(self, *args): return self
    def float(self): return Tensor(self.a.astype(np.float32))
    def int(self): return Tensor(self.a.astype(int))
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()
    def permute(self, *axes): return Tensor(self.a.transpose(axes))
    def unsqueeze(self, index): return Tensor(np.expand_dims(self.a, index))
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))
    def numel(self): return self.a.size
    def min(self): return self.a.min()
    def max(self): return self.a.max()
    def all(self): return self.a.all()
    def __getitem__(self, key): return Tensor(self.a[key])
    def __truediv__(self, x): return Tensor(self.a / x)
    def __mul__(self, x): return Tensor(self.a * x)
    def __sub__(self, x): return Tensor(self.a - x)
    def __gt__(self, x): return Tensor(self.a > x)


TORCH = types.SimpleNamespace(from_numpy=Tensor, no_grad=nullcontext,
    round=lambda t: Tensor(np.rint(t.a)), isfinite=lambda t: Tensor(np.isfinite(t.a)),
    is_tensor=lambda t: isinstance(t, Tensor))


class Module:
    training = False
    def __init__(self): self.hooks = []
    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(fn))
    def call(self, x):
        for hook in self.hooks: hook(self, (x,))


class FakeAdapter:
    torch, device = TORCH, "cpu"
    def __init__(self, method="invismark"):
        self.model_id = method
        self.module = Module()
        def decode(t):
            self.module.call(t)
            self.last_input = t.a
            return Tensor(np.zeros((1, 100), np.float32))
        self.wrapper = types.SimpleNamespace(invismark=types.SimpleNamespace(model=types.SimpleNamespace(
            decoder=self.module, _decode=decode)))
        def detect(t, is_video):
            assert not is_video
            self.module.call(t)
            return {"preds": Tensor(np.concatenate([[-1.0], np.ones(1024)])[None, :, None, None])}
        self.seal = types.SimpleNamespace(detector=self.module, detect=detect)
    def _image(self, path):
        return Tensor(c.rgb(path).astype(np.float32).transpose(2, 0, 1) / 255)


class DecoderTests(Fixture):
    def test_direct_invismark_range_shape_and_length(self):
        adapter = FakeAdapter()
        result, trace = decoder.traced(adapter, lambda: decoder.direct(adapter, self.pixels))
        self.assertEqual(len(result["bits"]), 100)
        np.testing.assert_array_equal(adapter.last_input,
            (self.pixels.astype(np.float32).transpose(2, 0, 1) / 255 * 2 - 1)[None])
        self.assertEqual(trace[0]["shape"], [1, 3, 512, 512])
        self.assertEqual(adapter.module.hooks, [])

    def test_chunky_detection_coordinate_and_singleton_dimensions(self):
        adapter = FakeAdapter("chunkyseal")
        result, _ = decoder.traced(adapter, lambda: decoder.direct(adapter, self.pixels))
        self.assertEqual(result["bits"], [1] * 1024)

    def test_float_range_not_guessed(self):
        with self.assertRaises(ValueError):
            decoder.direct(FakeAdapter(), self.pixels.astype(np.float32))
        result = decoder.direct(FakeAdapter(), self.pixels.astype(np.float32) / 255)
        self.assertEqual(len(result["bits"]), 100)

    def test_training_mode_rejected(self):
        adapter = FakeAdapter()
        adapter.module.training = True
        with self.assertRaises(RuntimeError):
            decoder.traced(adapter, lambda: {})

    def test_hook_removed_on_exception(self):
        adapter = FakeAdapter()
        with self.assertRaises(ZeroDivisionError):
            decoder.traced(adapter, lambda: 1/0)
        self.assertEqual(adapter.module.hooks, [])

    def test_mock_controls_decode_resume_and_report(self):
        adapter = FakeAdapter()
        calls = []
        def measured(_adapter, path, metadata, case, seed):
            self.assertTrue(not any(c.read(metadata)["expected_bits"]))
            calls.append(str(path))
            direct = decoder.direct(adapter, c.rgb(path))
            result = score(direct, case["assignment"])
            result.pop("expected_bits", None)
            return result
        with patch.object(pipeline, "quality", return_value={}):
            pipeline.controls(self.job)
        with patch("wmbench_digital.worker.load_adapter", return_value=adapter), \
             patch("wmbench_digital.worker.measured", side_effect=measured), \
             patch("wmbench.worker.seed_everything"), patch("wmbench.worker.runtime_info", return_value={"test": True}), \
             patch.object(decoder, "source_audit"):
            decoder.decode(self.job)
            first_calls = len(calls)
            self.assertEqual(first_calls, 13)  # Clean replay + two file decodes per condition.
            decoder.decode(self.job)
            self.assertEqual(len(calls), first_calls + 1)  # Only one necessary clean replay on resume.
        rows = list((self.out / "invismark/checks").glob("*/*.json"))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(c.read(p)["io_agrees"] for p in rows))
        report.report(self.out, {"n_images": 1})
        self.assertIn("INCOMPLETE", (self.out / "REPORT.md").read_text())
        self.assertTrue((self.out / "per_case.csv").is_file())
        self.assertTrue(list((self.out / "gallery").glob("*.jpg")))


class GenerationTests(Fixture):
    def test_vae_float_and_png_reference_are_saved(self):
        import wmbench_neural.channel
        c.write(self.out / "invismark/attack-cases.json", [self.case])
        c.write(self.out / "_inputs/captions.json", {"captions": {"285": {"text": "Frozen caption"}}})
        identity = {"engine": "vae"}
        c.write(self.old / "_execution/vae.json", identity)
        old_hashes = {p: c.sha(p) for folder in (self.bank, self.old) for p in folder.rglob("*") if p.is_file()}
        class Channel:
            def __init__(self, *args): self.identity = identity
            def apply(_self, images, attack, seeds, captions):
                return [np.asarray(im).copy() for im in images], [False] * len(images), {"actual_denoising_steps": 0, "batch_seconds": 1}
        floating = lambda ch, images: np.stack([np.array(im).astype(np.float32)/255 for im in images])
        with patch.object(wmbench_neural.channel, "NeuralChannel", Channel), \
             patch.object(pipeline, "vae_float", side_effect=floating), patch.object(pipeline, "quality", return_value={}):
            pipeline.generate({**self.job, "engine": "vae"})
        receipt = c.read(self.out / "invismark/generated/vae_1/test_native.json")
        np.testing.assert_array_equal(np.rint(np.load(self.out / receipt["float_path"]) * 255).astype(np.uint8), self.pixels)
        self.assertTrue(pipeline.generated_complete(self.out, self.case, "vae_1"))
        self.assertTrue(pipeline.generated_complete(self.out, self.case, "vae_2"))
        self.assertEqual(old_hashes, {p: c.sha(p) for folder in (self.bank, self.old) for p in folder.rglob("*") if p.is_file()})

    def test_vae_inconsistent_float_fails_without_scoring(self):
        import wmbench_neural.channel
        c.write(self.out / "invismark/attack-cases.json", [self.case])
        c.write(self.out / "_inputs/captions.json", {"captions": {"285": {"text": "Frozen caption"}}})
        c.write(self.old / "_execution/vae.json", {"engine": "vae"})
        class Channel:
            def __init__(self, *args): self.identity = {"engine": "vae"}
            def apply(_self, images, attack, seeds, captions):
                return [np.array(im) for im in images], [False]*4, {"actual_denoising_steps": 0, "batch_seconds": 1}
        with patch.object(wmbench_neural.channel, "NeuralChannel", Channel), \
             patch.object(pipeline, "vae_float", return_value=np.zeros((4, 512, 512, 3), np.float32)):
            with self.assertRaisesRegex(RuntimeError, "float reference"):
                pipeline.generate({**self.job, "engine": "vae"})
        self.assertFalse((self.out / "invismark/generated/vae_1/test_native.json").exists())

    def test_historical_batch_order_and_resume(self):
        import wmbench_neural.channel
        from wmbench_neural.spec import BY_ID
        group = []
        for index, mode in enumerate(("native", "matched_39", "matched_42", "matched_45")):
            group.append({**self.case, "id": f"case_{index}", "mode": mode})
        c.write(self.out / "invismark/attack-cases.json", group)
        c.write(self.out / "_inputs/captions.json", {"captions": {"285": {"text": "Frozen caption"}}})
        identity = {"engine": "sdxl", "scheduler_config": {"_use_default_values": ["a", "b"]}}
        c.write(self.old / "_execution/sdxl.json", identity)
        batches_seen = []
        class Channel:
            def __init__(self, *args): self.identity = identity
            def apply(_self, images, attack, seeds, captions):
                self.assertEqual(attack, BY_ID["sdxl_0.15_r1"])
                self.assertEqual(captions, ["Frozen caption"] * 4)
                batches_seen.append(seeds)
                return [np.asarray(im).copy() for im in images], [False]*4, {"actual_denoising_steps": 3, "batch_seconds": 1}
        job = {**self.job, "engine": "sdxl"}
        with patch.object(wmbench_neural.channel, "NeuralChannel", Channel), patch.object(pipeline, "quality", return_value={}):
            pipeline.generate(job)
            pipeline.generate(job)
        self.assertEqual(len(batches_seen), 1)
        receipt = c.read(self.out / "invismark/generated/sdxl_0.15_r1/case_0.json")
        self.assertEqual(receipt["batch_case_ids"], [f"case_{i}" for i in range(4)])
        self.assertEqual(len(list((self.out / "invismark/images").rglob("*.png"))), 1)

    def test_execution_change_fails_before_generation(self):
        import wmbench_neural.channel
        c.write(self.out / "invismark/attack-cases.json", [self.case])
        c.write(self.old / "_execution/sdxl.json", {"repo": "original"})
        channel = types.SimpleNamespace(identity={"repo": "changed"})
        with patch.object(wmbench_neural.channel, "NeuralChannel", return_value=channel):
            with self.assertRaisesRegex(RuntimeError, "execution differs"):
                pipeline.generate({**self.job, "engine": "sdxl"})


class ReleaseTests(unittest.TestCase):
    def test_all_notebook_cells_compile_and_release_hash(self):
        dist = ROOT / "dist"
        name = "InvisMark_ChunkySeal_NEURAL_DIAGNOSTICS"
        nb = json.loads((dist / (name + ".ipynb")).read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")
                self.assertEqual(cell["outputs"], [])
        data = (dist / (name + "_source.zip")).read_bytes()
        self.assertIn(hashlib.sha256(data).hexdigest(), json.dumps(nb))
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            manifest = json.loads(z.read("manifest.json"))
            for path, checksum in manifest["files"].items():
                self.assertEqual(hashlib.sha256(z.read(path)).hexdigest(), checksum)
                if path.endswith(".py"):
                    compile(z.read(path), path, "exec")
            self.assertEqual(hashlib.sha256(z.read("base.zip")).hexdigest(), c.BASE_SHA)


if __name__ == "__main__":
    unittest.main()
