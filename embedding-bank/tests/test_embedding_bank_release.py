"""Execute only the source-extraction part of the real notebook bootstrap offline."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
