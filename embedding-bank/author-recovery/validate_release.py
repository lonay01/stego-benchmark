"""Compile all cells/modules and validate exact packaged bytes; no GPU claim."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "RoSteALS_SepWater_RECOVERY"


def main():
    import nbformat
    notebook = nbformat.read(ROOT / "dist" / (NAME + ".ipynb"), as_version=4)
    nbformat.validate(notebook)
    payload = (ROOT / "dist" / (NAME + "_source.zip")).read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((ROOT / "dist/build.json").read_text())
    assert sha == manifest["sha256"]
    cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert len(cells) == 4 and sha in cells[1]
    for index, source in enumerate(cells):
        compile(source, f"cell-{index}", "exec")
        assert "BASE64" not in source
    count = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            assert not name.startswith("/") and ".." not in Path(name).parts
            if name.endswith(".py"):
                ast.parse(archive.read(name).decode("utf-8-sig"), filename=name)
                count += 1
        assert archive.read("authorrecovery/ecc.py") == (ROOT / "authorrecovery/ecc.py").read_bytes()
        assert archive.read("locks/rosteals.txt") == (ROOT / "locks/rosteals.txt").read_bytes()
        assert "slackclient==2.9.3" in archive.read("locks/rosteals.txt").decode()
        assert "DISTSEAL_RESUME.md" in archive.namelist()
    assert count == manifest["python_files"]
    result = {"code_cells_compiled": len(cells), "python_modules_parsed": count,
              "nbformat_valid": True, "zip_sha256": sha, "gpu_run": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
