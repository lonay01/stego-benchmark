"""Execute only the source-extraction part of the real notebook bootstrap offline."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bank_builder", PROJECT / "scripts/build_embedding_bank.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ExternalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "source.zip"

    def tearDown(self):
        self.temporary.cleanup()

    def extract(self, payload, expected=None):
        self.archive.write_bytes(payload)
        prefix = builder.BOOTSTRAP.split("# The kernel never receives Torch")[0]
        prefix = prefix.replace("__CHECKSUM__", expected or hashlib.sha256(payload).hexdigest())
        namespace = {
            "Path": Path,
            "IN_COLAB": False,
            "OUTPUT_ROOT": self.root / "bank",
            "SCRATCH_ROOT": self.root / "scratch",
            "SOURCE_ARCHIVE": self.archive,
            "DEVICE": "cpu",
            "shutil": __import__("shutil"),
        }
        exec(compile(prefix, "notebook_bootstrap_source", "exec"), namespace)
        return namespace["PROJECT"]

    def test_source_zip_extracts_readable_python_and_resumes(self):
        files = {"src/example.py": b"answer = 42\n"}
        payload = builder.zip_bytes(files)
        project = self.extract(payload)
        self.assertEqual((project / "src/example.py").read_bytes(), files["src/example.py"])
        self.assertEqual(project, self.extract(payload))

    def test_mismatched_zip_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.extract(builder.zip_bytes({"test.py": b""}), expected="0" * 64)

    def test_archive_may_not_escape_source_directory(self):
        with self.assertRaisesRegex(ValueError, "Unsafe source archive path"):
            self.extract(builder.zip_bytes({"../outside.py": b""}))

    def test_locally_modified_source_is_not_overwritten(self):
        payload = builder.zip_bytes({"test.py": b"original\n"})
        project = self.extract(payload)
        target = project / "test.py"
        target.write_bytes(b"modified\n")
        with self.assertRaisesRegex(ValueError, "Modified release source"):
            self.extract(payload)
        self.assertEqual(target.read_bytes(), b"modified\n")

    def colab_upload(self, returned, expected):
        # Run the actual source-extraction block; Drive/GPU checks precede it.
        prefix = (
            "import hashlib, io, zipfile\nSCRATCH_ROOT.mkdir"
            + builder.BOOTSTRAP.split("SCRATCH_ROOT.mkdir", 1)[1].split(
                "# The kernel never receives Torch"
            )[0]
        )
        namespace = {
            "Path": Path,
            "IN_COLAB": True,
            "SCRATCH_ROOT": self.root / "scratch",
            "SOURCE_ARCHIVE": self.archive,
        }
        colab = ModuleType("google.colab")
        colab.files = SimpleNamespace(upload=Mock(return_value=returned))
        with patch.dict("sys.modules", {"google.colab": colab}):
            exec(
                compile(prefix.replace("__CHECKSUM__", expected), "colab_upload", "exec"), namespace
            )
        colab.files.upload.assert_called_once()
        return namespace["PROJECT"]

    def test_colab_asks_for_new_zip_when_old_release_is_present(self):
        old = builder.zip_bytes({"source.py": b"old\n"})
        new = builder.zip_bytes({"source.py": b"fixed\n"})
        self.archive.write_bytes(old)
        project = self.colab_upload({"source (1).zip": new}, hashlib.sha256(new).hexdigest())
        self.assertEqual((project / "source.py").read_bytes(), b"fixed\n")
        self.assertEqual(self.archive.read_bytes(), old)

    def test_colab_rejects_wrong_uploaded_release(self):
        wrong = builder.zip_bytes({"source.py": b"wrong\n"})
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.colab_upload({"wrong.zip": wrong}, "0" * 64)


if __name__ == "__main__":
    unittest.main()
