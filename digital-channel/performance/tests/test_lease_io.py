"""Synthetic storage failures: preserve ownership and immutable evidence."""

import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wmbench import colab_canary
from wmbench_fast import lease_io


class LeaseIO(unittest.TestCase):
    def test_transient_missing_read_recovers_without_writes(self):
        errors = [FileNotFoundError(errno.ENOENT, "FUSE unavailable")] * 5
        with patch.object(Path, "read_text", side_effect=errors + ['{"token":"owner"}']), \
                patch.object(lease_io.time, "sleep") as sleep, \
                patch.object(Path, "write_text", side_effect=AssertionError("No recreation")):
            self.assertEqual(lease_io.read_lease(Path("lease")), {"token": "owner"})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4, 8, 8])

    def test_persistent_missing_read_fails_closed(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError(errno.ENOENT, "gone")), \
                patch.object(lease_io.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Ownership was not bypassed"):
                lease_io.read_lease(Path("lease"))
        self.assertEqual(sum(call.args[0] for call in sleep.call_args_list), 39)

    def test_transport_read_recovers(self):
        for code in lease_io.TRANSIENT:
            with self.subTest(errno=code), \
                    patch.object(Path, "read_text", side_effect=[OSError(code, "transient"), '{}']), \
                    patch.object(lease_io.time, "sleep"):
                self.assertEqual(lease_io.read_lease(Path("lease")), {})

    def test_permission_error_is_immediate(self):
        with patch.object(Path, "read_text", side_effect=PermissionError(errno.EACCES, "denied")), \
                patch.object(lease_io.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                lease_io.read_lease(Path("lease"))
            sleep.assert_not_called()

    def test_partial_json_can_recover(self):
        with patch.object(Path, "read_text", side_effect=['{"token":', '{"token":"same"}']), \
                patch.object(lease_io.time, "sleep"):
            self.assertEqual(lease_io.read_lease(Path("lease"))["token"], "same")

    def test_persistent_corruption_stops(self):
        with patch.object(Path, "read_text", return_value='{"token":'), \
                patch.object(lease_io.time, "sleep"):
            with self.assertRaises(RuntimeError):
                lease_io.read_lease(Path("lease"))

    def test_nonobject_is_not_ownership(self):
        with patch.object(Path, "read_text", return_value='[]'):
            with self.assertRaises(ValueError):
                lease_io.read_lease(Path("lease"))

    def test_install_changes_only_reader(self):
        previous = colab_canary.read_lease
        controller = colab_canary.controller_lease
        try:
            lease_io.install()
            lease_io.install()
            self.assertIs(colab_canary.read_lease, lease_io.read_lease)
            self.assertIs(colab_canary.controller_lease, controller)
        finally:
            colab_canary.read_lease = previous

    def test_actual_lease_rejects_foreign_token_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(colab_canary, "read_lease", lease_io.read_lease):
            root = Path(tmp)
            with colab_canary.controller_lease(root, heartbeat_interval=3600) as owned:
                owned()
                path = root / "active_controller.json"
                other = {"token": "other-controller", "pid": os.getpid()}
                path.write_text(json.dumps(other))
                with self.assertRaisesRegex(RuntimeError, "lease lost"):
                    owned()
            self.assertEqual(json.loads(path.read_text()), other)

    def test_actual_lease_recovers_only_same_owner(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(colab_canary, "read_lease", lease_io.read_lease):
            root = Path(tmp)
            with colab_canary.controller_lease(root, heartbeat_interval=3600) as owned:
                path = root / "active_controller.json"
                before = path.read_bytes()
                raw = Path.read_text
                counter = [0]
                def flaky(file, *args, **kwargs):
                    if file == path and counter[0] < 3:
                        counter[0] += 1
                        raise FileNotFoundError(errno.ENOENT, "transient")
                    return raw(file, *args, **kwargs)
                with patch.object(Path, "read_text", flaky), patch.object(lease_io.time, "sleep"):
                    owned()
                self.assertEqual(path.read_bytes(), before)

    def test_second_controller_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(colab_canary, "read_lease", lease_io.read_lease):
            with colab_canary.controller_lease(Path(tmp), heartbeat_interval=3600):
                with self.assertRaisesRegex(RuntimeError, "Another controller"):
                    with colab_canary.controller_lease(Path(tmp)):
                        self.fail("Concurrent owner accepted")

    def test_preflight_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, output = root / "bank", root / "results"
            bank.mkdir()
            (output / "_reports").mkdir(parents=True)
            marker = bank / "carrier.png"
            marker.write_bytes(b"unchanged evidence")
            lease = output / "_reports/active_controller.json"
            lease.write_text('{"token":"existing"}')
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            lease_io.storage_preflight({"bank": str(bank), "output": str(output)})
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_preflight_missing_folder_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                lease_io.storage_preflight({"bank": str(root / "bank"), "output": str(root / "out")})
            self.assertEqual(list(root.iterdir()), [])

    def test_preflight_disconnected_mount(self):
        with patch.object(lease_io.os.path, "ismount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                lease_io.storage_preflight({"bank": "/content/drive/MyDrive/bank",
                                           "output": "/content/drive/MyDrive/results"})

    def test_failed_preflight_leaves_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bank").mkdir()
            (root / "out/_reports").mkdir(parents=True)
            with patch.object(lease_io, "read_lease", return_value={"token": "not-the-probe"}):
                with self.assertRaisesRegex(RuntimeError, "readback differs"):
                    lease_io.storage_preflight({"bank": str(root / "bank"), "output": str(root / "out")})
            self.assertEqual(len(list((root / "out/_reports").glob(".storage-probe_*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
