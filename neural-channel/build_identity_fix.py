"""Build a small readable recovery overlay; never rebuild the frozen base ZIP."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_NAME = "COCO_Neural_Channel_RESUMABLE"
NAME = "COCO_Neural_Channel_SDXL_IDENTITY_FIX"
BASE_SHA256 = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"


def repair_cell(source):
    checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # A readable Python string, not base64/compressed code. The exact same .py
    # file lives under repairs/ in the repository for review and unit testing.
    content = f'''# Narrow recovery overlay; all files in the original release remain untouched.
REPAIR_SOURCE = {source!r}
REPAIR_SHA256 = {checksum!r}
REPAIR_PATH = PROJECT / "src/neural_sdxl_identity_fix.py"
repair_bytes = REPAIR_SOURCE.encode("utf-8")
if hashlib.sha256(repair_bytes).hexdigest() != REPAIR_SHA256:
    raise ValueError("Repair source checksum mismatch")
if REPAIR_PATH.exists() and REPAIR_PATH.read_bytes() != repair_bytes:
    raise ValueError("Different recovery overlay exists. Use a new scratch folder; keep OUTPUT_ROOT.")
REPAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
REPAIR_PATH.write_bytes(repair_bytes)
if hashlib.sha256(REPAIR_PATH.read_bytes()).hexdigest() != REPAIR_SHA256:
    raise ValueError("Repair readback failed")
ENV["WMBENCH_NEURAL_IDENTITY_FIX_SHA256"] = REPAIR_SHA256
print("Recovery ready: only Diffusers bookkeeping-key order can differ.")
print("Existing results, frozen execution identities and original source ZIP are preserved.")
'''
    # Keep source visible as ordinary multi-line Python, including its docstrings.
    literal = "r'''" + source + "'''"
    if "'''" in source:
        raise ValueError("Choose a different delimiter for the readable embedded module")
    return content.replace(repr(source), literal, 1)


def build():
    archive = ROOT / "dist" / (BASE_NAME + "_source.zip")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Original ZIP changed; do not rebuild or substitute the saved release")
    original = ROOT / "dist" / (BASE_NAME + ".ipynb")
    notebook = json.loads(original.read_text(encoding="utf-8"))
    source = (ROOT / "repairs/neural_sdxl_identity_fix.py").read_text(encoding="utf-8")
    ast.parse(source)
    cells = notebook["cells"]
    updated = 0
    for cell in cells:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code":
            if '"wmbench_neural.runner"' in text:
                text = text.replace('"wmbench_neural.runner"', '"neural_sdxl_identity_fix"')
                updated += 1
            cell.update(source=text.splitlines(True), outputs=[], execution_count=None)
        if cell["id"] == "neural-00":
            cell["source"] = ("# COCO neural channel — SDXL identity recovery\n\n"
                "Continue the existing run with the same N_IMAGES, paths, batch size and GPU/environment. "
                "The original ZIP is reused from Drive. **Do not delete results or start a new output folder.**\n\n"
                "This fix accepts only a permutation of Diffusers' internal default-key list. "
                "All weights, attack settings, seeds, decoders and scientific measurements are unchanged. "
                "Meaningful identity differences still stop the run and now name the fields.\n\n"
                "Run all cells in order in a **separate runtime from the active digital experiment**. "
                "Static and offline CPU tests do not replace a real-GPU acceptance run.\n").splitlines(True)
    if updated != 3 or cells[4]["id"] != "neural-04":
        raise ValueError("Original notebook layout changed; inspect before building")
    cells[5:5] = [dict(cell_type="markdown", id="identity-repair-notes", metadata={}, source=[
        "## 1a. Restore the narrow compatibility fix\n\n",
        "The readable module below is added beside the original sources; none are overwritten. "
        "Its checksum is checked in every worker. Repair provenance is written to "
        "`OUTPUT_ROOT/_repairs/scheduler-default-order-v1/`. Old execution manifests are never rewritten.\n"]),
        dict(cell_type="code", id="identity-repair-code", metadata={}, execution_count=None,
             outputs=[], source=repair_cell(source).splitlines(True))]
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
    notebook["metadata"].setdefault("colab", {})["name"] = NAME + ".ipynb"
    target = ROOT / "dist" / (NAME + ".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)
    return target


if __name__ == "__main__":
    build()
