"""Validate the delivered bytes, the narrow diff, and all CPU regression suites."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import nbformat

from build_lease_io_fix import BASE, BASE_SHA, DIST, NAME, ROOT


def main():
    started = time.monotonic()
    old_zip = DIST / (BASE + "_source.zip")
    assert hashlib.sha256(old_zip.read_bytes()).hexdigest() == BASE_SHA
    notebook = nbformat.read(DIST / (NAME + ".ipynb"), as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in cells:
        ast.parse(cell.source)
    outputs = []
    sandbox = ROOT / "validation"
    sandbox.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lease-io-", dir=sandbox) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(old_zip) as old, zipfile.ZipFile(DIST / (NAME + "_source.zip")) as new:
            changed = {n for n in old.namelist() if old.read(n) != new.read(n)}
            added = set(new.namelist()) - set(old.namelist())
            assert changed == {"src/wmbench_fast/runner.py", "src/wmbench_fast/launch.py",
                               "performance-release.json"}, changed
            assert added == {"src/wmbench_fast/lease_io.py", "performance-tests/test_lease_io.py",
                             "LEASE_IO_FIX.md"}, added
            # Narrowness of changes in existing entrypoints is executable evidence.
            expected_runner = old.read("src/wmbench_fast/runner.py").decode().replace(
                "def run(project, settings):\n", "def run(project, settings):\n"
                "    from .lease_io import install\n    install()\n")
            assert new.read("src/wmbench_fast/runner.py").decode().replace("\r\n", "\n") == expected_runner.replace("\r\n", "\n")
            expected_launcher = old.read("src/wmbench_fast/launch.py").decode().replace(
                "    settings = read_json(args.settings)\n",
                "    settings = read_json(args.settings)\n    if args.action == \"start\":\n"
                "        from .lease_io import storage_preflight\n        storage_preflight(settings)\n")
            assert new.read("src/wmbench_fast/launch.py").decode().replace("\r\n", "\n") == expected_launcher.replace("\r\n", "\n")
            for name in new.namelist():
                path = (root / name).resolve()
                assert path.is_relative_to(root.resolve())
                data = new.read(name)
                if name.endswith(".py"):
                    ast.parse(data, filename=name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        for folder, inventory in ((root, "digital-release.json"), (root / "production", "release.json"),
                                   (root, "performance-release.json")):
            for name, expected in json.loads((folder / inventory).read_text())["files"].items():
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "production/src"), str(root / "tests")]),
                   WM_DIGITAL_TEST_RELEASE=str(root), PYTHONIOENCODING="utf-8", TEMP=str(sandbox), TMP=str(sandbox))
        (root / "src/dist").mkdir()
        for suffix in ("RESUMABLE", "STEGASTAMP_FIX", "RESUME_FIX", "STEGASTAMP_CUDA_FIX", "STEGASTAMP_PAIRED_FIX"):
            shutil.copy2(DIST / f"COCO_Digital_Channel_{suffix}.ipynb", root / "src/dist")
        for suite in (root / "tests", root / "performance-tests", ROOT.parent / "repair-tests",
                      ROOT.parent / "cuda-tests", ROOT.parent / "paired-tests"):
            run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(suite), "-v"],
                                 env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            count = re.search(r"Ran (\d+) tests?", run.stderr)
            result = dict(suite=suite.name, exit_code=run.returncode,
                          tests=int(count.group(1)) if count else None, stdout=run.stdout, stderr=run.stderr)
            outputs.append(result)
            print(suite.name, "exit", run.returncode, "tests", result["tests"], flush=True)
            if run.returncode:
                print(run.stdout, run.stderr, flush=True)
    result = dict(code_cells=len(cells), frozen_scientific_sources_unchanged=True,
                  narrow_entrypoint_diff_verified=True, suites=outputs,
                  seconds=time.monotonic() - started,
                  limits="Synthetic CPU storage failures and pipeline smoke tests, not a real Colab FUSE or GPU-model test.")
    (DIST / "lease-io-fix-validation.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    if any(r["exit_code"] for r in outputs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
