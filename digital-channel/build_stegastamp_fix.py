"""Generate a minimal repair notebook without rebuilding the frozen source ZIP."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_SHA = "fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff"


def build():
    archive = ROOT / "dist/COCO_Digital_Channel_RESUMABLE_source.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ZIP_SHA:
        raise RuntimeError("Original digital source archive changed")
    original = json.loads((ROOT / "dist/COCO_Digital_Channel_RESUMABLE.ipynb").read_text("utf-8"))
    helper = (ROOT / "stegastamp_clean_hotfix.py").read_text("utf-8")
    ast.parse(helper)
    checksum = hashlib.sha256(helper.encode("utf-8")).hexdigest()
    # This is ordinary readable Python, not an encoded source archive.
    setup = ('# Narrow StegaStamp repair; the verified original release is NOT edited.\n'
             'HOTFIX_SOURCE = r\'\'\'\n' + helper + "'''\n"
             'HOTFIX_SOURCE = HOTFIX_SOURCE.lstrip("\\n")\n'
             f'HOTFIX_SHA = "{checksum}"\n'
             'if hashlib.sha256(HOTFIX_SOURCE.encode("utf-8")).hexdigest() != HOTFIX_SHA:\n'
             '    raise ValueError("StegaStamp repair source changed")\n'
             'HOTFIX_DIR = SCRATCH_ROOT / "repairs/stegastamp-clean-replay-v1"\n'
             'HOTFIX_DIR.mkdir(parents=True, exist_ok=True)\n'
             'for folder in (HOTFIX_DIR, OUTPUT_ROOT / "_release/repairs/stegastamp-clean-replay-v1"):\n'
             '    folder.mkdir(parents=True, exist_ok=True)\n'
             '    target = folder / "stegastamp_clean_hotfix.py"\n'
             '    if target.exists() and target.read_bytes() != HOTFIX_SOURCE.encode("utf-8"):\n'
             '        raise ValueError(f"Different repair already exists: {target}")\n'
             '    if not target.exists():\n'
             '        target.write_bytes(HOTFIX_SOURCE.encode("utf-8"))\n'
             '    if hashlib.sha256(target.read_bytes()).hexdigest() != HOTFIX_SHA:\n'
             '        raise ValueError("Repair readback failed")\n'
             'ENV["PYTHONPATH"] = str(HOTFIX_DIR) + os.pathsep + ENV["PYTHONPATH"]\n'
             'print("StegaStamp-only Clean replay repair enabled; bank and completed results retained.")\n')
    markdown = """## StegaStamp Clean replay repair

Only the StegaStamp Clean comparison is changed. No decoder, weights, payloads,
PSNR matching, attacks or other methods are changed. The original source ZIP
and run identity stay the same; completed cases are reused.

For StegaStamp Matched Quality only, a **single changed bit out of 100** may be
accepted when both extractions are non-exact and two additional decodes match
the **first fresh decode exactly**. No voting, best-BER selection or bit repair.
All other Clean mismatches still stop execution. A partial case cannot switch
its Clean reference. Model, dependency and PNG checksum checks remain enabled.

The old bank result is retained. Attack deltas use the actual fresh Clean result.
The discrepancy and all verification decodes are saved under
`watermarking_digital_channel_v1/stegastamp/clean-replay/` and referenced in
the case record. This is an explicit replay-policy exception, not a claim that
the historical and current outputs are identical.

**Run all cells with the same bank/output paths and N_IMAGES=5.** No new ZIP is
needed if the original source ZIP was saved on Drive. If requested, upload the
same `COCO_Digital_Channel_RESUMABLE_source.zip` as before. Do not delete results.
"""
    cells = []
    changes = 0
    for cell in original["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and 'command = [str(CONTROLLER), "-m", "wmbench_digital.runner", "run"' in source:
            cells.extend([
                {"cell_type": "markdown", "id": "stegastamp-repair-note", "metadata": {}, "source": markdown.splitlines(True)},
                {"cell_type": "code", "id": "stegastamp-repair-setup", "metadata": {}, "source": setup.splitlines(True),
                 "execution_count": None, "outputs": []},
            ])
            source = source.replace('"wmbench_digital.runner", "run"', '"stegastamp_clean_hotfix", "run"')
            cell["source"] = source.splitlines(True)
            changes += 1
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    if changes != 1:
        raise RuntimeError("Expected one run entry point")
    original["cells"] = cells
    name = "COCO_Digital_Channel_STEGASTAMP_FIX.ipynb"
    original["metadata"]["colab"]["name"] = name
    target = ROOT / "dist" / name
    target.write_text(json.dumps(original, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)
    return target


if __name__ == "__main__":
    build()
