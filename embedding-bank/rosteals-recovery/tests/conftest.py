"""Tests always import the built, distributed sources, not an unpatched parent."""

from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "authornight").is_dir():
    sys.path.insert(0, str(ROOT))
else:
    _temporary = tempfile.TemporaryDirectory(prefix="rosteals-fix-tests-")
    with zipfile.ZipFile(ROOT / "dist/RoSteALS_AUTHOR_FIX_source.zip") as archive:
        archive.extractall(_temporary.name)
    sys.path.insert(0, _temporary.name)
