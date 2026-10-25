"""Check the delivered repair cell, frozen-source diff and CPU regressions."""

import argparse
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

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "dist"
NAME = "COCO_Digital_Channel_PARALLEL_PSUTIL_FIX"


def validate(source_path):
    started = time.monotonic()
    before = json.loads(source_path.read_text("utf-8"))
    after = nbformat.read(DIST / (NAME + ".ipynb"), as_version=4)
    nbformat.validate(after)
    original_cells = [c for c in after.cells if c.id not in {"psutil-fix-note", "psutil-fix-setup"}]
    assert len(original_cells) == len(before["cells"])
    for old, new in zip(before["cells"], original_cells):
        assert old["cell_type"] == new.cell_type
        assert "".join(old["source"]) == new.source, old.get("id")
    code = [c for c in after.cells if c.cell_type == "code"]
    for cell in code:
        ast.parse(cell.source)
        assert cell.outputs == [] and cell.execution_count is None
    setup = next(c.source for c in after.cells if c.id == "psutil-fix-setup")
    archive_path = DIST / "COCO_Digital_Channel_PARALLEL_FAST_RESUME_source.zip"
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert archive_sha == "3115ddb4920c64f9e30c5e4ace91ffb521b5f449173bcbaeb863ec442b3fad94"
    sandbox = ROOT / "validation"
    sandbox.mkdir(exist_ok=True)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="psutil-repair-", dir=sandbox) as tmp:
        scratch = Path(tmp)
        base = scratch / "performance_release_v1"
        with zipfile.ZipFile(archive_path) as archive:
            originals = {name: archive.read(name) for name in archive.namelist()}
        for name, data in originals.items():
            target = (base / name).resolve()
            assert target.is_relative_to(base.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(base / "src"), str(base / "production/src")]),
                   PYTHONIOENCODING="utf-8", TEMP=str(sandbox), TMP=str(sandbox))
        scope = dict(Path=Path, os=os, json=json, hashlib=hashlib, zipfile=zipfile,
                     SCRATCH_ROOT=scratch, PROJECT=base, ENV=env,
                     SOURCE_ARCHIVE=archive_path, EXPECTED_SHA=archive_sha)
        exec(compile(setup, "delivered-psutil-repair-cell", "exec"), scope)
        fixed = scope["PROJECT"]
        differences = []
        for name, data in originals.items():
            assert (base / name).read_bytes() == data, "Original release modified"
            if (fixed / name).read_bytes() != data:
                differences.append(name)
            if name.endswith(".py"):
                ast.parse((fixed / name).read_bytes(), filename=name)
        assert set(differences) == {"src/wmbench_fast/worker.py", "src/wmbench_fast/scheduler.py",
                                    "performance-release.json"}, differences
        for name in ("worker.py", "scheduler.py"):
            assert (fixed / "src/wmbench_fast" / name).read_bytes() == (ROOT / "src/wmbench_fast" / name).read_bytes()
        for folder, name in ((fixed, "digital-release.json"), (fixed / "production", "release.json"),
                              (fixed, "performance-release.json")):
            manifest = json.loads((folder / name).read_text("utf-8"))
            for member, expected in manifest["files"].items():
                assert hashlib.sha256((folder / member).read_bytes()).hexdigest() == expected, member
        times = {name: (fixed / name).stat().st_mtime_ns for name in originals}
        env_before = dict(env)
        exec(compile(setup, "delivered-psutil-repair-cell-repeat", "exec"), scope)
        assert env == env_before
        assert times == {name: (fixed / name).stat().st_mtime_ns for name in originals}
        # Supply only temporary test fixtures; no changes to shipped sources.
        (fixed / "src/dist").mkdir()
        for name in ("RESUMABLE", "STEGASTAMP_FIX", "RESUME_FIX", "STEGASTAMP_CUDA_FIX", "STEGASTAMP_PAIRED_FIX"):
            shutil.copy2(DIST / f"COCO_Digital_Channel_{name}.ipynb", fixed / "src/dist")
        shutil.copy2(ROOT / "tests/test_psutil_fix.py", fixed / "performance-tests/test_psutil_fix.py")
        env.update(PYTHONPATH=env["PYTHONPATH"] + os.pathsep + str(fixed / "tests"),
                   WM_DIGITAL_TEST_RELEASE=str(fixed))
        for suite in (fixed / "tests", fixed / "performance-tests", ROOT.parent / "repair-tests",
                      ROOT.parent / "cuda-tests", ROOT.parent / "paired-tests"):
            run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(suite), "-v"],
                                 env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(suite.name, run.returncode, flush=True)
            print(run.stderr, flush=True)
            outputs.append(dict(suite=suite.name, exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr))
        # A changed existing repair file must not be silently replaced.
        changed = fixed / "src/wmbench_fast/worker.py"
        changed.write_bytes(b"# temporary deliberate corruption for fail-closed test\n")
        try:
            exec(compile(setup, "delivered-psutil-repair-conflict", "exec"), scope)
        except ValueError as error:
            assert "Different repair already exists" in str(error)
        else:
            raise AssertionError("Corrupt existing repair was silently overwritten")
        assert changed.read_bytes().startswith(b"# temporary deliberate corruption")
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == archive_sha
    result = dict(code_cells=len(code), original_cells_unchanged=True, source_zip_unchanged=True,
                  changed_release_files=sorted(differences), all_inventories_verified=True,
                  repair_cell_idempotent=True, corrupt_repair_rejected=True,
                  seconds=time.monotonic() - started, suites=outputs,
                  limits="CPU regression/smoke only. No real InvisMark CUDA execution locally.")
    target = DIST / "parallel-psutil-fix-validation.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(target)
    if any(r["exit_code"] for r in outputs):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    validate(parser.parse_args().source)
