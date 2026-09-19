"""Check shipped notebook cells, separate source ZIP and relocation offline."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = ROOT / "dist/COCO100_Embedding_Bank_RESUMABLE.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    constants = {}
    checked = []
    for index, cell in enumerate(data["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell["outputs"] and cell["execution_count"] is None
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(tree, f"cell_{index}", "exec")
        checked.append(index)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
    assert "PAYLOAD_BASE64" not in constants
    assert notebook.read_bytes() == (ROOT / notebook.name).read_bytes()
    payload = (ROOT / "dist/COCO100_Embedding_Bank_source.zip").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == constants["SOURCE_SHA256"]
    python_files = 0
    with tempfile.TemporaryDirectory(prefix="wmbench_release_") as directory:
        project = Path(directory)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            for name in bundle.namelist():
                assert (project / name).resolve().is_relative_to(project.resolve())
                if name.endswith(".py"):
                    ast.parse(bundle.read(name).decode("utf-8-sig"), filename=name)
                    python_files += 1
                local = ROOT / name
                if local.exists() and name != "README.md":
                    if name == "registry/bank_models.json":
                        assert json.loads(bundle.read(name)) == json.loads(local.read_bytes())
                    else:
                        assert bundle.read(name) == local.read_bytes(), name
            bundle.extractall(project)
        env = {**os.environ, "PYTHONPATH": str(project / "src")}
        command = [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from wmbench.embedding_bank import bank_registry, message_for_model; "
            "r=bank_registry(Path(sys.argv[1])); assert len(r)==32; "
            "assert len([m for m in r.values() if int(m['native_bits'])>=30])==31; "
            "assert message_for_model(r['trustmark_q'],'139')[0] == message_for_model(r['trustmark_b'],'139')[0]; "
            "assert message_for_model(r['trustmark_q'],'139')[0] == message_for_model(r['trustmark_p'],'139')[0]; "
            "assert message_for_model(r['fnns_omniseal'],'139')[0][:16]==[1,0]*8; "
            "assert 'torch' not in sys.modules; print('PACKAGED_REGISTRY_OK: 32, no torch in controller')",
            str(project),
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    report = {
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "code_cells": checked,
        "source_zip_python_files": python_files,
        "source_zip_sha256": constants["SOURCE_SHA256"],
        "packaged_registry_test": result.stdout.strip(),
        "limitations": "Static and offline package checks, not Colab/GPU execution.",
    }
    target = ROOT / "dist/embedding_bank_release_validation.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
