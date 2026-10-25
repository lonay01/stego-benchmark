"""Static validation and CPU smoke of the exact user-downloadable source ZIP."""

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
NAME = "COCO_Embedding_Bank_PRODUCTION"


def main():
    notebook = json.loads((ROOT / "dist" / f"{NAME}.ipynb").read_text("utf-8"))
    build = json.loads((ROOT / "dist/build.json").read_text())
    data = (ROOT / "dist" / f"{NAME}_source.zip").read_bytes()
    assert hashlib.sha256(data).hexdigest() == build["sha256"]
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for i, source in enumerate(cells):
        ast.parse(source, filename=f"cell_{i}")
        assert "PAYLOAD_BASE64" not in source
    assert build["sha256"] in cells[1]
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    with tempfile.TemporaryDirectory(prefix="wmbench-production-test-") as directory:
        project = Path(directory)
        with zipfile.ZipFile(ROOT / "dist" / f"{NAME}_source.zip") as archive:
            for name in archive.namelist():
                assert (project / name).resolve().is_relative_to(project.resolve())
                if name.endswith(".py"):
                    ast.parse(archive.read(name).decode("utf-8-sig"), filename=name)
            archive.extractall(project)
        manifest = json.loads((project / "release.json").read_text())
        for name, expected in manifest["files"].items():
            assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected
        env = os.environ.copy()
        env.update(PYTHONPATH=str(project / "src"), WMBENCH_TEST_RELEASE=str(project), PYTHONIOENCODING="utf-8")
        fixture = Path(r"D:\GIT\tmp\meta-hidden-reference")
        if fixture.is_dir():
            env.setdefault("WMBENCH_META_FIXTURES", str(fixture))
        result = subprocess.run([sys.executable, str(ROOT / "tests/test_bank.py")], env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        baseline = Path(r"C:\Users\sirezhik\Downloads\COCO_Embedding_Bank_PRODUCTION.ipynb")
        if baseline.is_file():
            env.setdefault("WMBENCH_BASELINE_NOTEBOOK", str(baseline))
        hotfix = subprocess.run([sys.executable, str(ROOT / "tests/test_system_dependency_hotfix.py")], env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout)
    print(result.stderr)
    print(hotfix.stdout)
    print(hotfix.stderr)
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    (folder / "local_checks.json").write_text(json.dumps({
        "archive_sha256": build["sha256"], "python": sys.version, "code_cells_parsed": len(cells),
        "modules_parsed": build["python_modules"], "notebook_schema_valid": True,
        "cpu_tests_exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        "notebook_hotfix_tests_exit_code": hotfix.returncode,
        "notebook_hotfix_stdout": hotfix.stdout, "notebook_hotfix_stderr": hotfix.stderr,
        "full_29_method_cuda_run": "not_executed"}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)
    if hotfix.returncode:
        raise SystemExit(hotfix.returncode)


if __name__ == "__main__":
    main()
