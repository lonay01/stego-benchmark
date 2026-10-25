"""No model/network dependencies: exercise fail-closed repair transactions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sepmark_byte_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = self.root / "bank"
        self.work = self.root / "recovery"
        self.work.mkdir()
        records = self.bank / "sepmark/_meta/records"
        records.mkdir(parents=True)
        (self.bank / "sepmark/native").mkdir()
        self.targets = {}
        for stem in recovery.TARGETS:
            candidate = self.work / f"{stem}.png"
            candidate.write_bytes(b"test-only-png:" + stem.encode())
            self.targets[stem] = recovery.sha(candidate)
            relative = f"sepmark/native/{stem}.png"
            (self.bank / relative).write_bytes(b"")
            recovery.save_new(records / f"{stem}_native.json", {
                "status": "saved", "signature": "original", "output": relative,
                "artifacts": [{"path": relative, "sha256": self.targets[stem]}],
            })
        self.patch = patch.object(recovery, "TARGETS", self.targets)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        evidence = recovery.collect_evidence(self.bank)
        recovery.save_new(self.work / "original_record_hashes.json", evidence)
        recovery.save_new(self.work / "candidate_report.json", {"bank": str(self.bank), "ready": True})

    def test_restore_preserves_all_record_bytes_and_is_idempotent(self):
        before = recovery.collect_evidence(self.bank)
        recovery.commit(self.work)
        recovery.commit(self.work)
        self.assertEqual(recovery.collect_evidence(self.bank), before)
        for stem, expected in self.targets.items():
            self.assertEqual(recovery.sha(self.bank / f"sepmark/native/{stem}.png"), expected)
            self.assertEqual((self.work / "empty_originals" / f"{stem}.png").stat().st_size, 0)

    def test_bad_candidate_does_not_touch_any_target(self):
        stem = list(self.targets)[-1]
        (self.work / f"{stem}.png").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "Candidate changed"):
            recovery.commit(self.work)
        self.assertTrue(all((self.bank / f"sepmark/native/{s}.png").stat().st_size == 0 for s in self.targets))

    def test_nonempty_damaged_target_is_not_overwritten(self):
        stem = next(iter(self.targets))
        target = self.bank / f"sepmark/native/{stem}.png"
        target.write_bytes(b"do not replace")
        with self.assertRaisesRegex(ValueError, "Nonempty changed"):
            recovery.commit(self.work)
        self.assertEqual(target.read_bytes(), b"do not replace")

    def test_changed_record_is_not_accepted(self):
        stem = next(iter(self.targets))
        record = self.bank / f"sepmark/_meta/records/{stem}_native.json"
        row = recovery.read(record)
        row["signature"] = "changed"
        record.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "Original records changed"):
            recovery.commit(self.work)

    def test_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            recovery.inside(self.bank, "../outside")


if __name__ == "__main__":
    unittest.main()
