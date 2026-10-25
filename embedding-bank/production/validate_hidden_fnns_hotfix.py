"""Validate deployable notebook and scoped recovery against the frozen ZIP."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    import nbformat

    build = json.loads((ROOT / "dist/hidden_fnns_build.json").read_text())
    notebook = nbformat.read(ROOT / "dist" / build["notebook"], as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs
    helper = ROOT / "dist/hotfix_hidden_fnns.py"
    assert helper.read_bytes() == (ROOT / "hotfix_hidden_fnns.py").read_bytes()
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == build["helper_sha256"]
    source = ROOT / "dist/COCO_Embedding_Bank_PRODUCTION_source.zip"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == build["source_archive_sha256"]
    results = {}
    with tempfile.TemporaryDirectory(prefix="hidden-fnns-release-") as directory:
        project = Path(directory)
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                assert (project / name).resolve().is_relative_to(project.resolve())
            archive.extractall(project)
        env = {**os.environ, "PYTHONPATH": str(project / "src"),
               "WMBENCH_TEST_RELEASE": str(project), "PYTHONIOENCODING": "utf-8"}
        for name in ("test_hidden_fnns_hotfix.py",):
            result = subprocess.run([sys.executable, str(ROOT / "tests" / name)], env=env,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(result.stdout)
            print(result.stderr)
            results[name] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    receipt = {**build, "notebook_schema_valid": True, "code_cells_parsed": 4,
               "python": sys.version, "tests": results}
    (ROOT / "validation/hidden_fnns_checks.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if any(v["exit_code"] for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
