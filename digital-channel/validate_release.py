"""Static validation and offline smoke tests of the exact distributed ZIP."""

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parent
NAME = "COCO_Digital_Channel_RESUMABLE"


def main():
    start = time.monotonic()
    notebook = nbformat.read(ROOT/"dist"/(NAME+".ipynb"), as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in code_cells:
        ast.parse(cell.source)
    modules = 0
    with tempfile.TemporaryDirectory(prefix="wm-digital-validation-") as directory:
        release = Path(directory)
        with zipfile.ZipFile(ROOT/"dist"/(NAME+"_source.zip")) as archive:
            for member in archive.infolist():
                target = (release/member.filename).resolve()
                if not target.is_relative_to(release.resolve()):
                    raise ValueError("Unsafe ZIP member")
                data = archive.read(member)
                if member.filename.endswith(".py"):
                    ast.parse(data.decode("utf-8"), member.filename)
                    modules += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        inventory = json.loads((release/"digital-release.json").read_text())
        for name, expected in inventory["files"].items():
            assert hashlib.sha256((release/name).read_bytes()).hexdigest() == expected
        production = json.loads((release/"production/release.json").read_text())
        for name, expected in production["files"].items():
            assert hashlib.sha256((release/"production"/name).read_bytes()).hexdigest() == expected
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join([str(release/"src"), str(release/"production/src")]),
                   WM_DIGITAL_TEST_RELEASE=str(release), PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(release/"tests"), "-v"],
                                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(result.stdout)
        print(result.stderr)
    report = dict(notebook_code_cells=len(code_cells), parsed_python_modules=modules,
                  test_exit_code=result.returncode, seconds=round(time.monotonic()-start, 2),
                  runtime=sys.version, versions={n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "scipy", "torch", "torchvision", "nbformat")},
                  meta_fixture_supplied=bool(os.environ.get("WM_DIGITAL_META_FIXTURE")),
                  stdout=result.stdout, stderr=result.stderr,
                  full_29_method_cuda_run="not executed; requires user Colab/GPU run")
    (ROOT/"dist/validation-results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
