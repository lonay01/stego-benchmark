"""Import repair only; all previous inference and measurement code stays frozen."""

import importlib.util
from pathlib import Path
import sys

from wmbench.io import read_json
from wmbench_full_payload.storage import EvidenceError


def rosteals_import_paths(options):
    """WIBE's YAML names RoSteALS.*; both parent and legacy ldm root are needed."""
    repository = Path(options["repo_root"]).resolve()
    rosteals = repository / "submodules/RoSteALS"
    paths = [rosteals.parent, rosteals, repository / "submodules/taming-transformers"]
    for path in reversed(paths):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing pinned source directory: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    found = importlib.util.find_spec("RoSteALS")
    locations = list(found.submodule_search_locations or []) if found else []
    if not locations or any(Path(p).resolve() != rosteals for p in locations):
        raise EvidenceError("RoSteALS import resolves outside the pinned checkout")
    return [str(p) for p in paths]


def work(job, phase):
    if job["model"]["id"] == "rosteals" and phase == "reference":
        root = Path(job["output"])
        prepared = read_json(root / "rosteals/_meta/prepared.json")
        options = read_json(prepared["config"])["options"]
        paths = rosteals_import_paths(options)
        print("Reference import roots:", paths, flush=True)
    from wmbench_author_reference.worker import work as frozen_work
    frozen_work(job, phase)
