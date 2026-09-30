"""Static checks plus CPU tests executed from the delivered source ZIP."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-report", help="Optionally verify the actual reviewed parent archive")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "dist/RoSteALS_AUTHOR_FIX.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{i}", "exec")
            code_cells += 1
    source = ROOT / "dist/RoSteALS_AUTHOR_FIX_source.zip"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    assert checksum in "\n".join(c.source for c in notebook.cells)
    assert checksum == json.loads((ROOT / "dist/build.json").read_text())["sha256"]
    with tempfile.TemporaryDirectory(prefix="rosteals-shipped-tests-") as directory:
        project = Path(directory)
        with zipfile.ZipFile(source) as archive:
            modules = 0
            for name in archive.namelist():
                if name.endswith(".py"):
                    ast.parse(archive.read(name).decode("utf-8-sig"), filename=name)
                    modules += 1
            archive.extractall(project)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project)
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                                cwd=project, env=env, text=True, capture_output=True)
        print(result.stdout, result.stderr)
        result.check_returncode()
        if args.parent_report:
            parent = project / "actual-parent"
            parent.mkdir()
            with zipfile.ZipFile(args.parent_report) as archive:
                (parent / "plan.json").write_bytes(archive.read("plan.json"))
            check = subprocess.run([sys.executable, "-c",
                "from rostealsfix.migration import make_plan; import sys; "
                "p=make_plan(dict(parent=sys.argv[1],output=sys.argv[2],scratch=sys.argv[3],max_hours=8),sys.argv[4]); "
                "assert len(p['images'])==100; assert list(p['profiles'])==['rosteals']; "
                "print('Actual reviewed parent: checksum, 100 images, profiles verified')",
                str(parent), str(project / "new-output"), str(project / "new-scratch"), checksum],
                cwd=project, env=env, text=True, capture_output=True)
            print(check.stdout, check.stderr)
            check.check_returncode()
    validation = {"source_sha256": checksum, "code_cells_compiled": code_cells,
                  "python_modules_parsed": modules, "distributed_zip_tests": result.stdout.strip(),
                  "gpu_inference_tested_locally": False,
                  "actual_parent_plan_checked": bool(args.parent_report)}
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation/release_checks.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
