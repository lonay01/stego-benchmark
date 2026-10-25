"""Add only StegaStamp's paired-control repair to the submitted CUDA notebook."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    notebook = json.loads((ROOT / "dist/COCO_Digital_Channel_STEGASTAMP_CUDA_FIX.ipynb").read_text("utf-8"))
    helper = (ROOT / "stegastamp_paired_hotfix.py").read_text("utf-8")
    checksum = hashlib.sha256(helper.encode()).hexdigest()
    setup = ("# StegaStamp only; all earlier source files and other methods stay unchanged.\n"
             "PAIRED_FIX_SOURCE = r'''\n" + helper + "'''\n"
             'PAIRED_FIX_SOURCE = PAIRED_FIX_SOURCE.lstrip("\\n")\n'
             f'PAIRED_FIX_SHA = "{checksum}"\n'
             'data = PAIRED_FIX_SOURCE.encode("utf-8")\n'
             'if hashlib.sha256(data).hexdigest() != PAIRED_FIX_SHA:\n'
             '    raise ValueError("Paired-control repair source changed")\n'
             'PAIRED_FIX_DIR = SCRATCH_ROOT / "repairs/stegastamp-paired-cuda-v1"\n'
             'for folder in (PAIRED_FIX_DIR, OUTPUT_ROOT / "_release/repairs/stegastamp-paired-cuda-v1"):\n'
             '    folder.mkdir(parents=True, exist_ok=True)\n'
             '    target = folder / "stegastamp_paired_hotfix.py"\n'
             '    if target.exists() and target.read_bytes() != data:\n'
             '        raise ValueError(f"Different paired repair already exists: {target}")\n'
             '    if not target.exists():\n'
             '        target.write_bytes(data)\n'
             '    if hashlib.sha256(target.read_bytes()).hexdigest() != PAIRED_FIX_SHA:\n'
             '        raise ValueError("Paired repair readback failed")\n'
             'ENV["PYTHONPATH"] = str(PAIRED_FIX_DIR) + os.pathsep + ENV["PYTHONPATH"]\n'
             'print("StegaStamp: fixed CUDA numerical profile, repeated Clean checks, separate historical reference.")\n')
    note = """## StegaStamp only: paired Clean under a fixed numerical profile

The prior CUDA repair loaded the GPU correctly, but historical replay still
differed by one bit on `000000000285_m02_matched_45` (51% vs 50% BER).
An old extraction is not a guaranteed bitwise oracle for a new execution.

**This explicitly changes StegaStamp's numerical execution/control policy.**
The graph, weights, PNGs, messages, preprocessing, threshold and attacks do not
change. This is not a claim to reproduce the old execution bit-for-bit.

- CUDA stays mandatory. TF32 is disabled; convolution search is fixed to
  `HEURISTIC`; ORT deterministic compute is enabled with sequential execution.
  These settings are fixed in advance, not selected for a better BER.
- All selected Clean PNGs are decoded three times in one session, then once
  after recreating the decoder. Bits must agree across these four extractions.
  The **first** is the paired control; no voting or best-of-N selection.
- The historical bank stays read-only. Differences from its Clean go to
  `stegastamp/tables/stegastamp_bank_vs_paired_clean.csv`.
- After checks pass, only StegaStamp's old digital records are copied and
  verified under `stegastamp/_preserved_previous_records/stegastamp-paired-cuda-v1/`
  before being withdrawn from active tables and recomputed.
- Attacks and their paired Clean use the same runtime profile. On resume,
  existing paired controls and runtime fingerprints must match exactly.
  Technical failures or non-repeatable results still stop; they are not hidden.

The same controls feed the existing BER/exact/delta calculations. Historical
Clean differences are not counted as damage caused by attacks.

Keep the existing bank/output paths and `N_IMAGES = 5`; select A100 and run all
cells. Completed other methods are reused. No new ZIP or re-embedding needed.
Earlier helper cells remain for provenance; this entry point supersedes both
the one-bit tolerance and the historical-bit-equality gate for StegaStamp only.

Local tests cover the code and available CPU inference, not a Colab GPU run.
The mandatory GPU repeatability checks execute before replacing any record.
"""
    changed, cells = 0, []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and '"stegastamp_cuda_hotfix", "run"' in source:
            cells += [dict(cell_type="markdown", id="stegastamp-paired-note", metadata={}, source=note.splitlines(True)),
                      dict(cell_type="code", id="stegastamp-paired-setup", metadata={}, source=setup.splitlines(True), outputs=[], execution_count=None)]
            cell["source"] = source.replace('"stegastamp_cuda_hotfix", "run"', '"stegastamp_paired_hotfix", "run"').splitlines(True)
            changed += 1
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    if changed != 1:
        raise RuntimeError("Expected one run cell")
    notebook["cells"] = cells
    name = "COCO_Digital_Channel_STEGASTAMP_PAIRED_FIX.ipynb"
    notebook["metadata"]["colab"]["name"] = name
    target = ROOT / "dist" / name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
