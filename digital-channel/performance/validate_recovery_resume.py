"""Validate delivered notebook, frozen sources and CPU regression/smoke suites."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import nbformat

from build_recovery_resume import ROOT, DIGITAL, DIST, NAME, BASE, BASE_SHA


def main():
    started = time.monotonic()
    nb = nbformat.read(DIST / (NAME + ".ipynb"), as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    for c in code:
        compile(c.source, c.id, "exec")
        assert not c.outputs and c.execution_count is None
    originals = {}
    with zipfile.ZipFile(DIST / (BASE + "_source.zip")) as z:
        originals = {n: z.read(n) for n in z.namelist()}
    outputs = []
    sandbox = ROOT / "validation"
    sandbox.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recovery-resume-", dir=sandbox) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(DIST / (NAME + "_source.zip")) as z:
            for n in z.namelist():
                p = (root / n).resolve()
                assert p.is_relative_to(root.resolve())
                data = z.read(n)
                if n.endswith(".py"):
                    ast.parse(data, filename=n)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        # All original scientific/controller-hotfix code is still byte identical.
        differences = [n for n, data in originals.items() if (root / n).read_bytes() != data]
        assert set(differences) == {"src/wmbench_fast/cache.py", "src/wmbench_fast/runner.py",
                                   "src/wmbench_fast/scheduler.py", "src/wmbench_fast/worker.py",
                                   "performance-release.json", "PERFORMANCE.md"}, differences
        for folder, inventory in ((root, "digital-release.json"), (root / "production", "release.json"),
                                  (root, "performance-release.json")):
            for n, expected in json.loads((folder / inventory).read_text())["files"].items():
                assert hashlib.sha256((folder / n).read_bytes()).hexdigest() == expected, n
        (root / "src/dist").mkdir()
        for n in ("RESUMABLE", "STEGASTAMP_FIX", "RESUME_FIX", "STEGASTAMP_CUDA_FIX", "STEGASTAMP_PAIRED_FIX"):
            shutil.copy2(DIST / f"COCO_Digital_Channel_{n}.ipynb", root / "src/dist")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "production/src"),
                       str(root / "tests"), str(root / "performance-tests")]),
                   WM_DIGITAL_TEST_RELEASE=str(root), PYTHONIOENCODING="utf-8", TEMP=str(sandbox), TMP=str(sandbox))
        for suite in (root / "tests", root / "performance-tests", DIGITAL / "repair-tests",
                      DIGITAL / "cuda-tests", DIGITAL / "paired-tests"):
            run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(suite), "-v"],
                                 env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(suite.name, run.returncode, flush=True)
            if run.returncode:
                print(run.stderr, flush=True)
            outputs.append(dict(suite=suite.name, exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr))
    result = dict(code_cells=len(code), frozen_scientific_sources_unchanged=True,
                  suites=outputs, seconds=time.monotonic()-started,
                  limits="CPU tests only. Real StegaStamp recovery and Drive timings must be checked separately in Colab.")
    target = DIST / "recovery-fast-resume-validation.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(target)
    if any(s["exit_code"] for s in outputs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
