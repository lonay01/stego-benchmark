"""Build the notebook and readable source ZIP from a frozen dependency release."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = "Embedding_Bank_AUTHOR_REFERENCE_source.zip"
NOTEBOOK = "Embedding_Bank_AUTHOR_REFERENCE.ipynb"
BASE_SHA256 = "b250d3bfcac3c0c1c9b27a5053a97dcb66b5a71ce1f6b34c8205be71d3d439ed"


def build():
    base_path = ROOT.parent / "native-diagnostics/dist/Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY_source.zip"
    payload = base_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen recovery dependency differs; do not silently rebuild its adapters")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        contents = {n: z.read(n) for n in z.namelist()}
    for path in (ROOT / "wmbench_author_reference").glob("*.py"):
        contents[f"src/wmbench_author_reference/{path.name}"] = path.read_bytes()
    for path in (ROOT / "tests").glob("*.py"):
        contents[f"tests/{path.name}"] = path.read_bytes()
    for filename in ("README.md", "VALIDATION.md", "source_contracts.json"):
        contents[filename] = (ROOT / filename).read_bytes()
    contents["REFERENCE_BASE_RELEASE.json"] = json.dumps({"file": base_path.name, "sha256": BASE_SHA256}).encode()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    payload = stream.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    spec = importlib.util.spec_from_file_location("base_notebook_builder", ROOT.parent / "full-payload/build_notebook.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    def transform(source):
        return (source.replace("Embedding_Bank_FULL_PAYLOAD_CANARY", "Embedding_Bank_AUTHOR_REFERENCE")
                .replace("watermarking_embedding_bank_v1", "watermarking_native_resolution_canary_v1")
                .replace("watermarking_full_payload_canary_v1", "watermarking_author_reference_v1")
                .replace("watermarking_full_payload_scratch_v1", "watermarking_author_reference_scratch_v1")
                .replace("wmbench_full_payload.runner", "wmbench_author_reference.runner")
                .replace("full_payload_settings.json", "author_reference_settings.json")
                .replace("full_payload_canary_report.zip", "author_reference_report.zip"))

    settings = transform(base.SETTINGS).replace('MODELS = "all"',
        'MODELS = ["hidden_benchmark", "rosteals", "sepwater", "sepmark", "distseal_posthoc_dcae"]')
    settings = settings.replace("All 32 entries; unavailable prerequisites stay visible.", "Only these five diagnostic methods.")
    settings = settings.replace("MAX_HOURS = 16", "MAX_HOURS = 6")
    settings = settings.replace("Leave 5 for this canary. Do not launch 100 yet.", "Fixed scope; do not change to 100 in this notebook.")
    settings = settings.replace("manual_assets_root=MANUAL_ASSETS_ROOT)",
        'manual_assets_root=MANUAL_ASSETS_ROOT or str(BANK_ROOT / "_assets"))')
    settings = settings.replace("5 full-length messages per image; Native + PSNR39/42/45. No attacks.",
        "125 historical cases + 50 new serializer-validation cases. No attacks or PSNR matching.")
    bootstrap = transform(base.BOOTSTRAP).replace("__CHECKSUM__", checksum)
    bootstrap = bootstrap.replace('payload = SOURCE_ARCHIVE.read_bytes() if SOURCE_ARCHIVE.is_file() else None',
        'CACHED_ARCHIVE = OUTPUT_ROOT / "_release" / (SOURCE_SHA256 + ".zip")\n'
        'archive_path = SOURCE_ARCHIVE if SOURCE_ARCHIVE.is_file() else CACHED_ARCHIVE\n'
        'payload = archive_path.read_bytes() if archive_path.is_file() else None')
    bootstrap = bootstrap.replace('PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]',
        'if N_IMAGES != 5:\n'
        '    raise ValueError("This diagnostic uses a fixed five-image replay and next-five holdout")\n'
        'CACHED_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)\n'
        'if not CACHED_ARCHIVE.exists():\n'
        '    temporary = CACHED_ARCHIVE.with_suffix(".part")\n'
        '    temporary.write_bytes(payload)\n'
        '    temporary.replace(CACHED_ARCHIVE)\n'
        'PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]')
    result = transform(base.REPORT).replace("Do not switch to 100 images until Native errors and blockers are reviewed.",
        "Review core agreement and holdout serialization separately. Production and old results are unchanged.")
    sections = [
        ("markdown", "# Five-method upstream reference checks\n\n"
         "**HiDDeN, RoSteALS, SepWater:** freshly reload the selected upstream neural core and compare it with the frozen adapter. "
         "Same five historical crops, five full-length messages each, weights and locked dependencies. "
         "Both decoders also receive the other branch's exact pixels.\n\n"
         "**SepMark, DistSeal:** test source-justified export rules on the historical five images and the next five from the frozen manifest. "
         "This part is a serializer check, not an independently reimplemented encoder.\n\n"
         "No attacks, matched PSNR, training, payload shortening or production edits. Results go into a new Drive folder. "
         "A finished test may still have nonzero BER.\n\n"
         "Use **A100 + high RAM** as in the previous diagnostic; model environments and phases run sequentially. "
         "Real GPU execution of this release still needs this Colab test. "
         f"Upload the matching **{ARCHIVE}** when asked; it contains readable source, no Base64 or model weights."),
        ("markdown", "## 1. Settings\n\nKeep N_IMAGES=5. BANK_ROOT is the existing native-resolution canary with actual PNGs and metadata, not a report ZIP. "
         "Do not point OUTPUT_ROOT at previous results. The next five original images must still be available in the bank."),
        ("code", settings),
        ("markdown", "## 2. Drive and readable source package\n\nThe source ZIP is cached on Drive after its first upload. "
         "Each method uses its old dependency lock. Only the lightweight controller is installed in the notebook kernel."),
        ("code", bootstrap),
        ("markdown", "## 3. Run checks\n\nCompleted case/phase records survive reconnects. Re-run cells 1–3 with the same settings. "
         "Source/checkpoint changes or damaged evidence stop the comparison instead of overwriting it. "
         "Keep the same GPU/runtime during a paired run; a software change is reported explicitly."),
        ("code", transform(base.RUN)),
        ("markdown", "## 4. Report and review ZIP\n\nYou can run this after a partial run. It reads small JSON files and does not load models. "
         "Send author_reference_report.zip for review; it excludes weights, source-image and float-tensor files."),
        ("code", result),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"author-reference-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            compile(source, f"cell_{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(payload)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = {"archive": ARCHIVE, "sha256": checksum, "notebook": NOTEBOOK, "base_sha256": BASE_SHA256,
                "python_modules": sum(n.endswith(".py") for n in contents), "code_cells": 4,
                "files": {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()}}
    (dist / "build.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; {len(payload):,} source bytes; SHA256 {checksum}")


if __name__ == "__main__":
    build()
