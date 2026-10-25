"""Validate the new notebook against the unchanged distributed release, offline."""

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
from build_identity_fix import ROOT, BASE_NAME, BASE_SHA256, NAME, build


def main():
    notebook_path = build()
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    original = nbformat.read(ROOT / "dist" / (BASE_NAME + ".ipynb"), as_version=4)
    original_cells = {c.id: c for c in original.cells}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            count += 1
            ast.parse(cell.source, filename=cell.id)
            if cell.id in original_cells:
                expected = original_cells[cell.id].source.replace(
                    '"wmbench_neural.runner"', '"neural_sdxl_identity_fix"')
                assert cell.source == expected, f"Unexpected existing-cell edit: {cell.id}"
    setup_cell = next(c.source for c in notebook.cells if c.id == "identity-repair-code")
    archive_path = ROOT / "dist" / (BASE_NAME + "_source.zip")
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == BASE_SHA256
    with tempfile.TemporaryDirectory(prefix="neural-identity-validation-") as directory:
        target = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            files = {}
            parsed = 0
            for name in archive.namelist():
                assert (target / name).resolve().is_relative_to(target.resolve())
                content = archive.read(name)
                files[name] = hashlib.sha256(content).hexdigest()
                if name.endswith(".py"):
                    ast.parse(content.decode("utf-8"), filename=name)
                    parsed += 1
            archive.extractall(target)
        scope = dict(PROJECT=target, hashlib=hashlib, ENV={})
        exec(compile(setup_cell, "notebook-repair-cell", "exec"), scope)
        repair_path = target / "src/neural_sdxl_identity_fix.py"
        expected = (ROOT / "repairs/neural_sdxl_identity_fix.py").read_text(encoding="utf-8").encode("utf-8")
        assert repair_path.read_bytes() == expected
        exec(compile(setup_cell, "notebook-repair-cell", "exec"), scope)
        repair_path.write_bytes(b"conflicting local modification")
        try:
            exec(compile(setup_cell, "notebook-repair-cell", "exec"), scope)
        except ValueError as error:
            assert "Different recovery overlay" in str(error)
        else:
            raise AssertionError("Existing different repair must not be overwritten")
        repair_path.write_bytes(expected)
        tests = (ROOT / "repairs/test_identity_fix.py").read_bytes()
        (target / "tests/test_identity_fix.py").write_bytes(tests)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(target / "src"), str(target / "production/src")]),
            "PYTHONIOENCODING": "utf-8", "HF_HOME": str(target / "hf"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(target / "tests"), "-v"],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        assert all(hashlib.sha256((target / name).read_bytes()).hexdigest() == checksum
                   for name, checksum in files.items()), "Original release file modified"
    value = dict(notebook_valid=True, notebook_code_cells=count, original_python_files_checked=parsed,
        original_release_files_unchanged=True, original_notebook_code_changes="3 entrypoint names only",
        base_archive_sha256=BASE_SHA256, repair_sha256=hashlib.sha256(expected).hexdigest(),
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        offline_tests_exit_code=result.returncode, test_output=result.stdout + result.stderr,
        cuda_tested=False, official_diffusion_weights_tested=False, python=sys.version)
    (ROOT / "dist" / (NAME + "_validation.json")).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(result.stdout)
    print(result.stderr)
    print(json.dumps({k: v for k, v in value.items() if k != "test_output"}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
