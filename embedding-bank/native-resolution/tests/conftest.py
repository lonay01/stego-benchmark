"""Import the exact source release shipped with the notebook."""
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_builder", ROOT / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
builder.build()
TEMP = tempfile.TemporaryDirectory(prefix="native-canary-tests-")
PROJECT = Path(TEMP.name)
with zipfile.ZipFile(ROOT / "dist" / builder.ARCHIVE) as archive:
    archive.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))


@pytest.fixture
def project():
    return PROJECT
