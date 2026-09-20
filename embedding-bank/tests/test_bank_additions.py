"""Offline adapter contracts, not pretrained inference or robustness evidence."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from wmbench.adapters.bank_additions import (
    DistSealPosthocAdapter,
    FINAdapter,
    FNNSAdapter,
    TrustMarkVariantAdapter,
    _freeze_parameters,
    _load_distseal_inference_state,
    _source_namespace,
    _verified_trustmark_asset,
    validate_fnns_payload,
)
from wmbench.adapters.extended import _write_json


ROOT = Path(__file__).resolve().parents[1]


class BankAdditionTests(unittest.TestCase):
    def test_scriptmodule_freeze_retains_input_gradients(self):
        import torch

        decoder = torch.jit.script(torch.nn.Linear(3, 48))
        with self.assertRaisesRegex(RuntimeError, "not supported on ScriptModules"):
            decoder.requires_grad_(False)
        _freeze_parameters(decoder)
        image = torch.ones((1, 3), requires_grad=True)
        decoder(image).square().sum().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in decoder.parameters()))

    def test_distseal_loads_inference_weights_with_training_autoencoder(self):
        import torch

        model = torch.nn.Linear(3, 64)
        state = {name: torch.ones_like(value) for name, value in model.state_dict().items()}
        state["autoencoder.model.encoder.weight"] = torch.zeros(2, 2)
        state["autoencoder.model.decoder.bias"] = torch.zeros(2)
        counts = _load_distseal_inference_state(model, state)
        self.assertEqual(counts, {"loaded_keys": 2, "unused_training_autoencoder_keys": 2})
        self.assertEqual(len(state), 4)  # filtering does not mutate the original checkpoint
        self.assertTrue(all(torch.all(value == 1) for value in model.state_dict().values()))

    def test_distseal_keeps_strict_missing_unexpected_and_shape_checks(self):
        import torch

        model = torch.nn.Linear(3, 64)
        state = model.state_dict()
        for problem, broken in (
            ("Missing key", {"weight": state["weight"]}),
            ("Unexpected key", {**state, "unrecognized.weight": torch.zeros(1)}),
            ("Unexpected key", {**state, "autoencoder.other.weight": torch.zeros(1)}),
            ("size mismatch", {**state, "weight": torch.zeros(1, 1)}),
        ):
            with self.subTest(problem=problem, keys=list(broken)):
                with self.assertRaisesRegex(RuntimeError, problem):
                    _load_distseal_inference_state(model, broken)

    def test_distseal_does_not_ignore_required_autoencoder_weights(self):
        import torch

        model = torch.nn.Module()
        model.autoencoder = torch.nn.Module()
        model.autoencoder.model = torch.nn.Linear(3, 4)
        with self.assertRaisesRegex(ValueError, "unexpectedly owns"):
            _load_distseal_inference_state(model, model.state_dict())

    def test_distseal_accepts_clean_inference_only_state(self):
        import torch

        model = torch.nn.Linear(3, 64)
        counts = _load_distseal_inference_state(model, model.state_dict())
        self.assertEqual(counts["unused_training_autoencoder_keys"], 0)
        with self.assertRaisesRegex(ValueError, "named state dictionary"):
            _load_distseal_inference_state(model, {1: torch.zeros(1)})

    def test_distseal_constructor_loads_separate_pinned_autoencoder(self):
        # Real torch state validation; deliberately tiny substituted networks.
        import torch

        network = torch.nn.Linear(3, 64)
        network.blender = SimpleNamespace(scaling_w=0)
        state = dict(network.state_dict())
        state["autoencoder.model.encoder.weight"] = torch.zeros(2, 2)
        dcae = torch.nn.Identity()
        loader = Mock(return_value=dcae)
        diffusers = ModuleType("diffusers")
        diffusers.AutoencoderDC = SimpleNamespace(from_pretrained=loader)
        cfg = ModuleType("distseal.utils.cfg")
        cfg.get_config_from_checkpoint = Mock(
            return_value=SimpleNamespace(
                args=SimpleNamespace(nbits=64, attenuation="none", get=lambda _key: "none")
            )
        )
        cfg.setup_model = Mock(return_value=network)
        modules = {
            "diffusers": diffusers,
            "distseal": ModuleType("distseal"),
            "distseal.utils": ModuleType("distseal.utils"),
            "distseal.utils.cfg": cfg,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dcae").mkdir()
            (root / "dcae/config.json").write_text("{}")
            torch.save({"model": state}, root / "checkpoint.pt")
            with patch.dict(sys.modules, modules):
                adapter = DistSealPosthocAdapter(
                    {
                        "repo_root": str(root),
                        "asset_root": str(root),
                        "weights": "checkpoint.pt",
                        "device": "cpu",
                        "scaling_w": 0.5,
                        "autoencoder_revision": "test-only-revision",
                    }
                )
            self.assertIs(adapter.autoencoder, dcae)
            self.assertEqual(adapter.checkpoint_loading["unused_training_autoencoder_keys"], 1)
            self.assertEqual(adapter.model.blender.scaling_w, 0.5)
            self.assertTrue(all(not p.requires_grad for p in adapter.model.parameters()))
            loader.assert_called_once_with(
                str(root / "dcae"), local_files_only=True, torch_dtype=torch.float32
            )

    def test_import_does_not_load_heavy_frameworks(self):
        code = (
            "import sys; from wmbench.adapters import bank_additions; "
            "assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_registry_has_exact_five_source_pinned_adapters(self):
        registry = json.loads((ROOT / "registry/bank_additions.json").read_text("utf-8"))
        expected = {"trustmark_b", "trustmark_p", "fin", "fnns_omniseal", "distseal_posthoc_dcae"}
        self.assertEqual({model["id"] for model in registry["models"]}, expected)
        for model in registry["models"]:
            self.assertEqual(model["track"], "posthoc")
            self.assertRegex(model["source"]["revision"], "^[0-9a-f]{40}$")
            self.assertEqual(model["capability"]["status"], "ready_unverified")
            module, name = model["adapter"].split(":")
            adapter = getattr(importlib.import_module(module), name)
            self.assertEqual(adapter.native_bits, model["native_bits"])
            self.assertTrue(callable(adapter.embed))
            self.assertTrue(callable(adapter.decode))
            for requirement in model["requirements"]:
                self.assertIn("==", requirement)
        fnns = next(m for m in registry["models"] if m["id"] == "fnns_omniseal")
        prefix = [fnns["reserved_bits"][str(i)] for i in range(16)]
        self.assertEqual(prefix, fnns["options"]["detection_prefix"])
        self.assertEqual(fnns["payload_layout"]["common_indices"], list(range(18, 48)))

    def test_fnns_mapping_rejects_wrong_prefix_or_payload(self):
        prefix = [1, 0] * 8
        payload = prefix + [0, 1] + [1, 0] * 15
        self.assertEqual(validate_fnns_payload(payload, prefix), payload)
        invalid = payload.copy()
        invalid[0] ^= 1
        for values in (invalid, payload[:-1], [0.0] * 48):
            with self.assertRaises(ValueError):
                validate_fnns_payload(values, prefix)

    def test_trustmark_lightning_inference_dependency_is_declared(self):
        # TrustMark_Arch inherits LightningModule even when no training is used.
        models = json.loads((ROOT / "registry/bank_additions.json").read_text("utf-8"))["models"]
        for model in models:
            if model["id"] in ("trustmark_b", "trustmark_p"):
                self.assertIn("lightning==2.4.0", model["requirements"])
                self.assertIn("pytorch-lightning==2.4.0", model["requirements"])

    def test_trustmark_asset_checks_precede_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = b"model test fixture, not a checkpoint"
            file = root / "encoder_B.ckpt"
            file.write_bytes(value)
            checksum = hashlib.md5(value).hexdigest()
            self.assertEqual(
                _verified_trustmark_asset(root, "irrelevant/encoder_B.ckpt", checksum), file
            )
            with self.assertRaises(ValueError):
                _verified_trustmark_asset(root, file.name, "0" * 32)
            with self.assertRaises(FileNotFoundError):
                _verified_trustmark_asset(root, "absent.ckpt", checksum)

    def test_selected_source_namespace_does_not_execute_root_initializer(self):
        name = "_wmbench_test_source_namespace"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "__init__.py").write_text("raise RuntimeError('not part of image inference')")
            try:
                _source_namespace(name, path)
                self.assertEqual(sys.modules[name].__path__, [str(path)])
                _source_namespace(name, path)
                with self.assertRaises(RuntimeError):
                    _source_namespace(name, path.parent)
            finally:
                sys.modules.pop(name, None)

    def test_trustmark_png_interface_and_blind_decode_with_fake_model(self):
        # Deliberately a fake IO test; not a watermark success measurement.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cover, marked, metadata = root / "cover.png", root / "marked.png", root / "meta.json"
            Image.new("RGB", (512, 512), (90, 100, 110)).save(cover)
            bits = [1, 0] * 50
            adapter = TrustMarkVariantAdapter.__new__(TrustMarkVariantAdapter)
            adapter.variant = "B"
            adapter.model_id = "trustmark_b"
            adapter.weight_hashes = {"fixture": "not-a-real-weight"}
            adapter.model = SimpleNamespace(
                encode=Mock(side_effect=lambda image, message, MODE: image.copy()),
                decode=Mock(return_value=("01" * 50, True, -1)),
            )
            adapter.embed(str(cover), str(marked), bits, str(metadata))
            data = json.loads(metadata.read_text("utf-8"))
            self.assertEqual(data["expected_bits"], bits)
            with Image.open(marked) as saved:
                self.assertEqual(saved.size, (512, 512))
            # Expected answers do not reach the decoder; it returns the fake decoder's output.
            data["expected_bits"] = [0] * 100
            _write_json(str(metadata), data)
            result = adapter.decode(str(marked), str(metadata))
            self.assertEqual(result["bits"], [0, 1] * 50)
            self.assertEqual(adapter.model.decode.call_args.kwargs, {"MODE": "binary"})
            data["variant"] = "P"
            _write_json(str(metadata), data)
            with self.assertRaises(ValueError):
                adapter.decode(str(marked), str(metadata))

    def test_fin_requires_explicit_resolution_policy_before_heavy_import(self):
        with self.assertRaisesRegex(ValueError, "resolution_policy"):
            FINAdapter({})

    def test_fin_decoding_inverts_inl_before_fed(self):
        adapter = FINAdapter.__new__(FINAdapter)
        adapter.device = "cpu"
        adapter._native_image = lambda image: image
        normalized = np.ones((1, 3, 128, 128), dtype=np.float32)
        corrected = np.full_like(normalized, 0.25)
        adapter.inl = Mock(return_value=corrected)
        adapter.fed = Mock(return_value=(None, np.ones((1, 64))))
        adapter.torch = SimpleNamespace(zeros=lambda shape, **kwargs: np.zeros(shape))
        logits = adapter._extract_logits(normalized)
        self.assertEqual(logits.shape, (1, 64))
        self.assertTrue(adapter.inl.call_args.kwargs["rev"])
        self.assertTrue(adapter.fed.call_args.kwargs["rev"])
        self.assertIs(adapter.fed.call_args.args[0][0], corrected)
        self.assertEqual(adapter.fed.call_args.args[0][1].shape, (1, 64))

    def test_distseal_decodes_all_bits_without_expected_message(self):
        adapter = DistSealPosthocAdapter.__new__(DistSealPosthocAdapter)
        logits = np.arange(65, dtype=np.float32).reshape(1, 65)
        adapter.model = SimpleNamespace(detector=Mock(return_value=logits))
        image = np.zeros((1, 3, 512, 512), dtype=np.float32)
        result = adapter._extract_logits(image)
        self.assertEqual(result.shape, (1, 64))
        np.testing.assert_array_equal(result, logits[:, 1:])
        adapter.model.detector.assert_called_once_with(image)

    def test_fnns_decoder_uses_fixed_network_not_expected_bits(self):
        adapter = FNNSAdapter.__new__(FNNSAdapter)
        image = np.zeros((1, 3, 512, 512), dtype=np.float32)
        expected = np.ones((1, 48))
        adapter.model = SimpleNamespace(
            normalize=lambda value: value, model=Mock(return_value=expected)
        )
        self.assertIs(adapter._extract_logits(image), expected)

    def test_source_compiles_and_has_no_eval_or_exec(self):
        path = ROOT / "src/wmbench/adapters/bank_additions.py"
        source = path.read_text("utf-8")
        tree = ast.parse(source)
        compile(tree, str(path), "exec")
        forbidden = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
        ]
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
