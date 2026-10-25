"""Build VideoSeal recovery on the existing notebook without rebuilding its ZIP."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREVIOUS = "COCO_Neural_Channel_SDXL_IDENTITY_FIX"
NAME = "COCO_Neural_Channel_VIDEOSEAL_DOWNLOAD_FIX"
BASE_NAME = "COCO_Neural_Channel_RESUMABLE"
BASE_SHA256 = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"
PREVIOUS_SHA256 = "e9a8ee25443982045a6c69077794fa6ed0401cf00fd5a8d9cbe37b07b3b3d08f"
OLD_N_IMAGES = "N_IMAGES = 5                  # After the canary: change ONLY this to 100."
NEW_N_IMAGES = "N_IMAGES = 100                # Resume the current full run; use 5 only for the earlier canary."


def repair_cell(source):
    if "'''" in source:
        raise ValueError("Embedded module contains the notebook string delimiter")
    checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f'''# Transport-only VideoSeal recovery; the frozen release is not edited.
VIDEOSEAL_REPAIR_SOURCE = r\x27\x27\x27{source}\x27\x27\x27
VIDEOSEAL_REPAIR_SHA256 = {checksum!r}
VIDEOSEAL_REPAIR_PATH = PROJECT / "src/neural_videoseal_download_fix.py"
videoseal_repair_bytes = VIDEOSEAL_REPAIR_SOURCE.encode("utf-8")
if hashlib.sha256(videoseal_repair_bytes).hexdigest() != VIDEOSEAL_REPAIR_SHA256:
    raise ValueError("VideoSeal repair source checksum mismatch")
if VIDEOSEAL_REPAIR_PATH.exists() and VIDEOSEAL_REPAIR_PATH.read_bytes() != videoseal_repair_bytes:
    raise ValueError("Different VideoSeal overlay exists; use new scratch, keep OUTPUT_ROOT.")
VIDEOSEAL_REPAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
VIDEOSEAL_REPAIR_PATH.write_bytes(videoseal_repair_bytes)
if hashlib.sha256(VIDEOSEAL_REPAIR_PATH.read_bytes()).hexdigest() != VIDEOSEAL_REPAIR_SHA256:
    raise ValueError("VideoSeal repair readback failed")
ENV["WMBENCH_VIDEOSEAL_DOWNLOAD_FIX_SHA256"] = VIDEOSEAL_REPAIR_SHA256
print("VideoSeal transport recovery ready. Exact saved checkpoint SHA-256 required.")
print("Existing results and pending attack packages are preserved. No GPU job started yet.")
'''


def build():
    archive = ROOT / "dist" / (BASE_NAME + "_source.zip")
    previous = ROOT / "dist" / (PREVIOUS + ".ipynb")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Original ZIP changed; do not rebuild the saved release")
    if hashlib.sha256(previous.read_bytes()).hexdigest() != PREVIOUS_SHA256:
        raise ValueError("Previous recovery notebook changed; inspect before rebuilding")
    notebook = json.loads(previous.read_text(encoding="utf-8"))
    source = (ROOT / "repairs/neural_videoseal_download_fix.py").read_text(encoding="utf-8")
    ast.parse(source)
    updated = 0
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell["id"] in {"neural-08", "neural-10", "neural-12"}:
            assert '"neural_sdxl_identity_fix"' in text
            text = text.replace('"neural_sdxl_identity_fix"', '"neural_videoseal_download_fix"')
            updated += 1
        elif cell["id"] == "neural-02":
            assert OLD_N_IMAGES in text
            text = text.replace(OLD_N_IMAGES, NEW_N_IMAGES)
        elif cell["id"] == "neural-00":
            text = ("# COCO neural channel — VideoSeal download recovery\n\n"
                "Resume the current **100-image run**, with the same bank, output folder, batch size "
                "and A100-SXM4-80GB execution environment. Do not remove saved results or attack ZIPs.\n\n"
                "Only VideoSeal checkpoint transport changes: Meta's endpoint has a pinned mirror fallback. "
                "Bytes are accepted only if their SHA-256 matches the saved bank, then cached on Drive. The previous SDXL "
                "identity-order repair is retained unchanged. All scientific checks remain enabled.\n\n"
                "Run all cells in order in a **separate runtime from any active digital experiment**. "
                "The unchanged source ZIP is reused from Drive. CPU/static tests do not replace a "
                "real-GPU resume acceptance run.\n")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    assert updated == 3
    position = next(i for i, c in enumerate(notebook["cells"]) if c["id"] == "identity-repair-code") + 1
    notebook["cells"][position:position] = [
        dict(cell_type="markdown", id="videoseal-repair-notes", metadata={}, source=[
            "## 1b. Restore the exact VideoSeal checkpoint\n\n",
            "Adds one readable transport overlay. The saved bank supplies the expected hash; "
            "the decoder, weights, attack grid and clean controls are not changed. "
            "The verified 218 MiB checkpoint and receipts live under "
            "`OUTPUT_ROOT/_repairs/videoseal-download-v1/`, outside temporary model slots.\n"]),
        dict(cell_type="code", id="videoseal-repair-code", metadata={}, outputs=[], execution_count=None,
             source=repair_cell(source).splitlines(True)),
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
    notebook["metadata"].setdefault("colab", {})["name"] = NAME + ".ipynb"
    target = ROOT / "dist" / (NAME + ".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
