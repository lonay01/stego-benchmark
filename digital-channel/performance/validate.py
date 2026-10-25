"""Validate the exact source ZIP, notebook, regression tests and CPU smoke."""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent/"dist"
NAME = "COCO_Digital_Channel_PARALLEL_FAST_RESUME"


def main():
    start = time.monotonic()
    notebook = nbformat.read(DIST/(NAME+".ipynb"), as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in cells:
        ast.parse(cell.source)
    outputs = []
    sandbox = ROOT/"validation"
    sandbox.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="parallel-digital-", dir=sandbox) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(DIST/(NAME+"_source.zip")) as archive:
            for name in archive.namelist():
                path = (root/name).resolve()
                assert path.is_relative_to(root.resolve())
                data = archive.read(name)
                if name.endswith(".py"):
                    ast.parse(data, filename=name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        with zipfile.ZipFile(DIST/"COCO_Digital_Channel_RESUMABLE_source.zip") as old:
            for name in old.namelist():
                assert old.read(name) == (root/name).read_bytes(), name
        for folder, inventory in ((root, "digital-release.json"), (root/"production", "release.json"),
                                  (root, "performance-release.json")):
            for name, expected in json.loads((folder/inventory).read_text())["files"].items():
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, name
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/"src"), str(root/"production/src"), str(root/"tests")]),
                   WM_DIGITAL_TEST_RELEASE=str(root), PYTHONIOENCODING="utf-8", TEMP=str(sandbox), TMP=str(sandbox))
        # Legacy notebook-diff tests locate fixtures beside the imported helper.
        # Supply those unmodified notebooks in the temporary validation tree only.
        (root/"src/dist").mkdir()
        for name in ("RESUMABLE", "STEGASTAMP_FIX", "RESUME_FIX", "STEGASTAMP_CUDA_FIX", "STEGASTAMP_PAIRED_FIX"):
            shutil.copy2(DIST/f"COCO_Digital_Channel_{name}.ipynb", root/"src/dist")
        for suite in (root/"tests", root/"performance-tests", ROOT.parent/"repair-tests", ROOT.parent/"cuda-tests", ROOT.parent/"paired-tests"):
            command = [sys.executable, "-m", "unittest", "discover", "-s", str(suite), "-v"]
            run = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(run.stdout)
            print(run.stderr)
            outputs.append(dict(suite=str(suite.name), exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr))
    result = dict(code_cells=len(cells), base_files_byte_identical=True, inventories_verified=True,
                  seconds=time.monotonic()-start, suites=outputs,
                  limits="CPU fixtures only; no real 29-method CUDA run or measured GPU speedup")
    (DIST/"parallel-validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    if any(r["exit_code"] for r in outputs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
