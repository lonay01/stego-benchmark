"""Add only the missing-cache resume repair to the previous notebook."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    notebook = json.loads((ROOT/"dist/COCO_Digital_Channel_STEGASTAMP_FIX.ipynb").read_text("utf-8"))
    helper = (ROOT/"digital_resume_hotfix.py").read_text("utf-8")
    checksum = hashlib.sha256(helper.encode()).hexdigest()
    setup = ('# Resume repair only: no changes to decoders, attacks, or saved results.\n'
             "RESUME_FIX_SOURCE = r'''\n" + helper + "'''\n"
             'RESUME_FIX_SOURCE = RESUME_FIX_SOURCE.lstrip("\\n")\n'
             f'RESUME_FIX_SHA = "{checksum}"\n'
             'if hashlib.sha256(RESUME_FIX_SOURCE.encode("utf-8")).hexdigest() != RESUME_FIX_SHA:\n'
             '    raise ValueError("Resume repair source changed")\n'
             'RESUME_FIX_DIR = SCRATCH_ROOT / "repairs/digital-missing-cache-v1"\n'
             'for folder in (RESUME_FIX_DIR, OUTPUT_ROOT / "_release/repairs/digital-missing-cache-v1"):\n'
             '    folder.mkdir(parents=True, exist_ok=True)\n'
             '    target = folder / "digital_resume_hotfix.py"\n'
             '    if target.exists() and target.read_bytes() != RESUME_FIX_SOURCE.encode("utf-8"):\n'
             '        raise ValueError(f"Different resume repair already exists: {target}")\n'
             '    if not target.exists():\n'
             '        target.write_bytes(RESUME_FIX_SOURCE.encode("utf-8"))\n'
             '    if hashlib.sha256(target.read_bytes()).hexdigest() != RESUME_FIX_SHA:\n'
             '        raise ValueError("Resume repair readback failed")\n'
             'ENV["PYTHONPATH"] = str(RESUME_FIX_DIR) + os.pathsep + ENV["PYTHONPATH"]\n'
             'print("Resume repair enabled: already absent temporary ZIPs need no cleanup.")\n')
    note = """## Resume after temporary cache loss

This fixes `FileNotFoundError: .../attack-cache/.../owner.json` during cleanup
of a completed method after Colab loses its temporary disk. Saved results are
still checked before cleanup. An absent ZIP is skipped; an existing ZIP still
requires the original ownership, path and checksum checks.

Decoders, attacks, metrics and the previous StegaStamp repair are unchanged.
Keep the same bank/output paths and `N_IMAGES = 5`; run all cells in order.
The original source ZIP is reused from Drive. Do not delete results or cache
metadata manually. This repair does not regenerate finished measurements.
"""
    cells, changes = [], 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and '"stegastamp_clean_hotfix", "run"' in source:
            cells.extend([
                dict(cell_type="markdown", id="resume-cache-note", metadata={}, source=note.splitlines(True)),
                dict(cell_type="code", id="resume-cache-setup", metadata={}, source=setup.splitlines(True),
                     outputs=[], execution_count=None),
            ])
            cell["source"] = source.replace('"stegastamp_clean_hotfix", "run"',
                                             '"digital_resume_hotfix", "run"').splitlines(True)
            changes += 1
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    if changes != 1:
        raise RuntimeError("Expected exactly one run entry point")
    notebook["cells"] = cells
    name = "COCO_Digital_Channel_RESUME_FIX.ipynb"
    notebook["metadata"]["colab"]["name"] = name
    target = ROOT/"dist"/name
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
