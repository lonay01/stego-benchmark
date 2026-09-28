"""Validate the distributed source archive rather than development-only imports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src/wmbench_meta_recheck").is_dir():
    PROJECT = ROOT
else:
    spec = importlib.util.spec_from_file_location("recheck_builder", ROOT / "build_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.build()
    TEMP = tempfile.TemporaryDirectory(prefix="meta-recheck-tests-")
    PROJECT = Path(TEMP.name)
    with zipfile.ZipFile(ROOT / "dist" / builder.ARCHIVE) as archive:
        archive.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))


@pytest.fixture
def project():
    return PROJECT
