"""Build a self-contained source ZIP and readable four-cell notebook."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ARCHIVE = "Embedding_Bank_NATIVE_RESOLUTION_CANARY_source.zip"
NOTEBOOK = "Embedding_Bank_NATIVE_RESOLUTION_CANARY.ipynb"


def build():
    # Reuse the already reviewed upload/bootstrap implementation. All installed
    # sources are copied into this release; the deployed ZIP needs no sibling tree.
    spec = importlib.util.spec_from_file_location("previous_canary_builder", BASE / "full-payload/build_notebook.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    contents = {}
    for directory in ("src", "registry", "locks"):
        for path in sorted((BASE / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".lock") and "__pycache__" not in path.parts:
                contents[path.relative_to(BASE).as_posix()] = path.read_bytes()
    for source, package in ((BASE / "full-payload/wmbench_full_payload", "wmbench_full_payload"),
                            (ROOT / "wmbench_native_canary", "wmbench_native_canary")):
        for path in sorted(source.glob("*.py")):
            contents[f"src/{package}/{path.name}"] = path.read_bytes()
    for name in ("__init__.py", "core.py", "worker.py", "assets.py"):
        contents[f"src/bank_diagnostics/{name}"] = (BASE / "full-payload/diagnostic_helpers" / name).read_bytes()
    for filename in ("README.md", "VALIDATION.md"):
        contents[filename] = (ROOT / filename).read_bytes()
    contents["THIRD_PARTY.md"] = (BASE / "THIRD_PARTY.md").read_bytes()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, data)
    payload = archive.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()

    def migrate(text):
        return (text.replace("Embedding_Bank_FULL_PAYLOAD_CANARY", "Embedding_Bank_NATIVE_RESOLUTION_CANARY")
            .replace("watermarking_full_payload_canary_v1", "watermarking_native_resolution_canary_v1")
            .replace("watermarking_full_payload_scratch_v1", "watermarking_native_resolution_scratch_v1")
            .replace("wmbench_full_payload.runner", "wmbench_native_canary.runner")
            .replace("full_payload_settings.json", "native_resolution_settings.json")
            .replace("full_payload_canary_report.zip", "native_resolution_canary_report.zip")
            .replace("All 32 entries; unavailable prerequisites stay visible.", "29 runnable entries; 3 explicit exclusions stay visible.")
            .replace("5 full-length messages per image; Native + PSNR39/42/45. No attacks.",
                     "5 full-length messages per image; native carrier/strength, center crops. No attacks or matched PSNR."))

    sections = [
        ("markdown", "# Native-resolution embedding canary\n\n"
         "Five historical COCO images × five messages × 29 runnable configurations = **725 Native PNGs**. "
         "Three unavailable entries remain explicit exclusions.\n\n"
         "Use each selected checkpoint's full useful message and declared carrier size. Smaller carriers are literal central crops "
         "of the frozen RGB512 input. No interpolation, external residual upscaling, PSNR matching or channel attacks. "
         "Flexible models keep the 512 carrier and their official internal preprocessing.\n\n"
         "**Payload note:** DWT uses 32 bits and SSL uses 30, their selected baseline settings—not the previous structural limits. "
         "FNNS: 32 useful + 16 detection bits. InvisMark: 94 useful + 6 reserved bits.\n\n"
         "Run in a Linux Colab GPU runtime. A100 with high system RAM is the previously used environment; "
         "this notebook does not purchase resources. Each model gets a separate environment, released after success.\n\n"
         f"Upload **{ARCHIVE}** when prompted. It contains ordinary, readable Python files. "
         "Old results are read-only. The new report is a canary, not a full robustness benchmark."),
        ("markdown", "## 1. Settings\n\nKeep N_IMAGES=5. BANK_ROOT must point to the bank with the exact historical _inputs folder."),
        ("code", migrate(previous.SETTINGS)),
        ("markdown", "## 2. Drive, source archive and controller\n\nMount Drive and select the matching source ZIP. Model libraries stay out of the notebook kernel."),
        ("code", migrate(previous.BOOTSTRAP).replace("__CHECKSUM__", checksum)),
        ("markdown", "## 3. Run\n\nRun again after a disconnect with the same paths and ZIP. Completed PNG/decoder pairs are verified and reused. "
         "Measured bit errors are not retried. Technical failures remain visible; other models continue. Progress is printed during preparation and after each message."),
        ("code", migrate(previous.RUN)),
        ("markdown", "## 4. Table and review archive\n\nPNG exact/BER, pre-PNG observations, geometry, payload size and technical failures. "
         "Unavailable pre-PNG observations are not successes. Report construction does not load a model or rehash weights."),
        ("code", migrate(previous.REPORT)),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"native-resolution-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            ast.parse(source, filename=f"cell_{index}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(payload)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"archive": ARCHIVE, "sha256": checksum,
        "notebook": NOTEBOOK, "source_files": len(contents), "code_cells": 4,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; ZIP {len(payload):,} bytes; SHA256 {checksum}")


if __name__ == "__main__":
    build()
