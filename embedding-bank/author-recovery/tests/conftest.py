"""Use bundled modules in a release, or the frozen parent package in checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "authornight").exists():
    sys.path.insert(0, str(ROOT.parent / "author-night"))
sys.path.insert(0, str(ROOT))
