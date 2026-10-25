"""Add a StegaStamp-only CUDA overlay to the current resume notebook."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    notebook = json.loads((ROOT / "dist/COCO_Digital_Channel_RESUME_FIX.ipynb").read_text("utf-8"))
    helper = (ROOT / "stegastamp_cuda_hotfix.py").read_text("utf-8")
    checksum = hashlib.sha256(helper.encode()).hexdigest()
    setup = ("# StegaStamp only. No source archive, bank, attack or other decoder changes.\n"
             "CUDA_FIX_SOURCE = r'''\n" + helper + "'''\n"
             'CUDA_FIX_SOURCE = CUDA_FIX_SOURCE.lstrip("\\n")\n'
             f'CUDA_FIX_SHA = "{checksum}"\n'
             'if hashlib.sha256(CUDA_FIX_SOURCE.encode()).hexdigest() != CUDA_FIX_SHA:\n'
             '    raise ValueError("CUDA repair source changed")\n'
             'CUDA_FIX_DIR = SCRATCH_ROOT / "repairs/stegastamp-cuda-v1"\n'
             'for folder in (CUDA_FIX_DIR, OUTPUT_ROOT / "_release/repairs/stegastamp-cuda-v1"):\n'
             '    folder.mkdir(parents=True, exist_ok=True)\n'
             '    target = folder / "stegastamp_cuda_hotfix.py"\n'
             '    data = CUDA_FIX_SOURCE.encode("utf-8")\n'
             '    if target.exists() and target.read_bytes() != data:\n'
             '        raise ValueError(f"Different CUDA repair already exists: {target}")\n'
             '    if not target.exists():\n'
             '        target.write_bytes(data)\n'
             '    if hashlib.sha256(target.read_bytes()).hexdigest() != CUDA_FIX_SHA:\n'
             '        raise ValueError("CUDA repair readback failed")\n'
             'ENV["PYTHONPATH"] = str(CUDA_FIX_DIR) + os.pathsep + ENV["PYTHONPATH"]\n'
             'print("StegaStamp: require actual CUDA, strict Clean replay, preserve legacy CPU results before recomputation.")\n')
    note = """## StegaStamp CUDA runtime repair (supersedes the one-bit exception)

The full previous logs show `libcublasLt.so.12` missing. ONNX silently used CPU,
whereas the embedding bank records CUDA. The cleanup repair above is still used;
the earlier **one-bit Clean exception is NOT used** by this run entry point.

This overlay changes StegaStamp only:

1. Link the existing controller CUDA 12 / cuDNN 9 libraries into its subprocesses.
2. Require actual CUDA session providers; disable automatic CPU fallback.
3. Replay every selected Clean carrier, requiring exact agreement with the bank.
4. Only after all pass: preserve old, unverified-backend StegaStamp result files
   byte-for-byte under `stegastamp/_preserved_cpu_records/stegastamp-cuda-v1/` on
   Drive, then withdraw them from active tables and recompute their attacks.

The bank and the other methods' results are unchanged. Failed CUDA preflight
leaves previous active records in place and stops; no tolerance is increased.
Interrupted CUDA runs resume their verified partial records normally.

Keep the same bank/output paths and `N_IMAGES = 5`. Select the same NVIDIA GPU
as the bank (A100), then run all cells. The original source ZIP is reused from
Drive. No manual result deletion, payload re-embedding or new archive is needed.

Preflight evidence: `stegastamp/cuda-preflight/`. GPU-backed execution still
requires verification in Colab; local CPU smoke tests cannot establish bitwise
CUDA replay. Completed methods are reused, not decoded again.
"""
    cells, changed = [], 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and '"digital_resume_hotfix", "run"' in source:
            cells.extend([
                dict(cell_type="markdown", id="stegastamp-cuda-note", metadata={}, source=note.splitlines(True)),
                dict(cell_type="code", id="stegastamp-cuda-setup", metadata={}, source=setup.splitlines(True),
                     execution_count=None, outputs=[]),
            ])
            cell["source"] = source.replace('"digital_resume_hotfix", "run"',
                                             '"stegastamp_cuda_hotfix", "run"').splitlines(True)
            changed += 1
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    if changed != 1:
        raise RuntimeError("Expected one run entry point")
    notebook["cells"] = cells
    name = "COCO_Digital_Channel_STEGASTAMP_CUDA_FIX.ipynb"
    notebook["metadata"]["colab"]["name"] = name
    target = ROOT / "dist" / name
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
