"""Offline regression for the reported Colab queue crash and safe source updates."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wmbench import embedding_bank as bank
from wmbench.io import digest, read_json, sha256, write_json


PROJECT = Path(__file__).resolve().parents[1]


class BlockedQueueTests(unittest.TestCase):
    def test_unimplemented_camera_has_no_completion_signature(self):
        model = bank.bank_registry(PROJECT)["camera_shooting_instance"]
        with patch.object(bank, "_signature", side_effect=AssertionError("must not sign")):
            self.assertFalse(bank._complete(Path("unused"), model, [{"image_id": "139"}], True))

    def test_blocked_camera_does_not_prevent_the_last_five_methods(self):
        registry = bank.bank_registry(PROJECT)
        selected = list(bank.MODEL_IDS[-6:])
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, scratch = root / "bank", root / "scratch"
            context = {
                "bank": output,
                "scratch": scratch,
                "models": registry,
                "images": [{"image_id": "139", "sha256": "test-only-source"}],
                "selected": selected,
                "identity": "test-only-identity",
            }

            def monitor(command, env, log, slot, *args, **kwargs):
                job = read_json(command[-1])
                name = job["model"]["id"]
                stage = command[-3]
                seen.append((name, stage))
                if stage == "prepare":
                    write_json(
                        output / name / "_meta/prepared.json", {"python": "test-only-python"}
                    )
                    return {"exit_code": 0, "reason": ""}
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("RuntimeError: synthetic worker failure\n", encoding="utf-8")
                return {"exit_code": 1, "reason": ""}

            stdout = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(patch.object(bank, "prepare_bank", return_value=context))
                stack.enter_context(
                    patch(
                        "wmbench.colab_canary.controller_lease",
                        return_value=nullcontext(lambda: None),
                    )
                )
                stack.enter_context(
                    patch("wmbench.colab_canary.monitored_process", side_effect=monitor)
                )
                stack.enter_context(
                    patch("wmbench.environment.worker_environment", return_value={})
                )
                stack.enter_context(redirect_stdout(stdout))
                bank.run(
                    PROJECT, {"output_root": str(output), "cleanup": False, "min_free_disk_gib": 0}
                )
            self.assertEqual(
                seen, [(name, stage) for name in selected[1:] for stage in ("prepare", "work")]
            )
            status = read_json(output / selected[0] / "_meta/status.json")
            self.assertEqual(status["status"], "blocked")
            self.assertIn("integration_required", status["reason"])
            report = read_json(output / "_reports/report.json")
            self.assertEqual(report["expected_png"], 24)
            self.assertEqual(report["saved_png"], 0)
            self.assertIn("synthetic worker failure", stdout.getvalue())
            self.assertTrue(all(row["status"] == "failed" for row in report["rows"][1:]))

    def test_short_payload_without_capability_is_blocked(self):
        self.assertFalse(
            bank._complete(Path("unused"), {"id": "test", "native_bits": 24}, [{}], True)
        )


class SourceCompatibilityTests(unittest.TestCase):
    def test_published_transition_matches_the_exact_current_runtime(self):
        fingerprint = digest(
            {
                p.relative_to(PROJECT).as_posix(): sha256(p)
                for p in sorted((PROJECT / "src").rglob("*.py"))
            }
        )
        transitions = read_json(PROJECT / "compatibility/embedding_bank_hotfixes.json")[
            "transitions"
        ]
        self.assertIn(fingerprint, [row["to_source_sha256"] for row in transitions])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project, self.output = self.root / "project", self.root / "bank"
        self.original = {
            "policy": bank.POLICY,
            "dataset": "test-only-dataset",
            "models": {"test": {"native_bits": 32}},
            "source_code_sha256": "old-test-only-code",
        }
        self.updated = {**self.original, "source_code_sha256": "new-test-only-code"}
        write_json(self.output / "bank.json", self.original)
        write_json(
            self.project / "compatibility/embedding_bank_hotfixes.json",
            {
                "transitions": [
                    {
                        "from_source_sha256": self.original["source_code_sha256"],
                        "to_source_sha256": self.updated["source_code_sha256"],
                        "reason": "synthetic audited fix",
                    }
                ]
            },
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_audited_code_update_preserves_original_identity_and_results(self):
        frozen = (self.output / "bank.json").read_bytes()
        sentinel = self.output / "test/native/sentinel.png"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"test-only saved artifact; not a real PNG")
        for _ in range(2):
            identity = bank.freeze_bank_identity(self.project, self.output, self.updated)
            self.assertEqual(identity, self.original)
        self.assertEqual((self.output / "bank.json").read_bytes(), frozen)
        self.assertEqual(sentinel.read_bytes(), b"test-only saved artifact; not a real PNG")
        receipt = read_json(self.output / "_reports/source_compatibility/new-test-only-code.json")
        self.assertEqual(receipt["from_source_sha256"], "old-test-only-code")

    def test_scientific_changes_are_rejected_even_with_approved_source(self):
        for key, value in (("dataset", "different"), ("models", {}), ("policy", {})):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.updated)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "No audited compatible update"):
                    bank.freeze_bank_identity(self.project, self.output, changed)

    def test_unknown_code_is_not_approved_implicitly(self):
        changed = {**self.updated, "source_code_sha256": "unaudited-code"}
        with self.assertRaisesRegex(ValueError, "No audited compatible update"):
            bank.freeze_bank_identity(self.project, self.output, changed)

    def test_missing_allowlist_is_not_a_bypass(self):
        with self.assertRaisesRegex(ValueError, "No audited compatible update"):
            bank.freeze_bank_identity(self.root / "other-project", self.output, self.updated)

    def test_unchanged_code_does_not_need_a_transition(self):
        self.assertEqual(
            bank.freeze_bank_identity(self.project, self.output, self.original), self.original
        )


if __name__ == "__main__":
    unittest.main()
