"""Static validation and CPU tests of the actual distributed source ZIP."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable, str(ROOT / "build_notebook.py")], check=True)
    archive_path = ROOT / "dist/COCO_Neural_Channel_RESUMABLE_source.zip"
    notebook_path = ROOT / "dist/COCO_Neural_Channel_RESUMABLE.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    cells = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            cells += 1
    with zipfile.ZipFile(archive_path) as archive:
        parsed = 0
        for name in archive.namelist():
            if name.endswith(".py"):
                ast.parse(archive.read(name).decode("utf-8"), filename=name)
                parsed += 1
        with tempfile.TemporaryDirectory(prefix="neural-validation-") as directory:
            target = Path(directory)
            for name in archive.namelist():
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise ValueError("Unsafe release member")
            archive.extractall(target)
            env = os.environ.copy()
            env.update(PYTHONPATH=os.pathsep.join([str(target/"src"), str(target/"production/src")]), PYTHONIOENCODING="utf-8",
                       HF_HOME=str(target/"hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(target/"tests"), "-v"],
                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    value = dict(notebook_code_cells=cells, python_files=parsed, notebook_valid=True,
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(), cpu_tests_exit_code=result.returncode,
        cpu_test_output=result.stdout+result.stderr, cuda_tested=False, python=sys.version)
    (ROOT / "dist/validation-results.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(result.stdout)
    print(result.stderr)
    print(json.dumps({k: v for k, v in value.items() if k != "cpu_test_output"}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
