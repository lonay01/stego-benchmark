"""Package frozen native-canary sources plus the separate diagnostic overlay."""

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
ARCHIVE = "Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY_source.zip"
NOTEBOOK = "Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY.ipynb"


def build():
    release = BASE / "native-resolution/dist"
    manifest = json.loads((release / "build.json").read_text(encoding="utf-8"))
    data = (release / manifest["archive"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Native canary dependency ZIP changed")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        contents = {name: z.read(name) for name in z.namelist()}
    for path in (ROOT / "wmbench_native_diagnostics").glob("*.py"):
        contents[f"src/wmbench_native_diagnostics/{path.name}"] = path.read_bytes()
    for filename in ("README.md", "VALIDATION.md"):
        contents[filename] = (ROOT / filename).read_bytes()
    contents["BASE_RELEASE.json"] = json.dumps({"archive": manifest["archive"],
                                               "sha256": manifest["sha256"]}, indent=2).encode()
    for name, payload in contents.items():
        if name.endswith(".py"):
            ast.parse(payload.decode("utf-8-sig"), filename=name)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, payload in sorted(contents.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 9, 22, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(item, payload)
    data = archive.getvalue()
    checksum = hashlib.sha256(data).hexdigest()
    spec = importlib.util.spec_from_file_location("base_builder", BASE / "full-payload/build_notebook.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    def migrate(text):
        return (text.replace("Embedding_Bank_FULL_PAYLOAD_CANARY", "Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY")
            .replace("watermarking_embedding_bank_v1", "watermarking_native_resolution_canary_v1")
            .replace("watermarking_full_payload_canary_v1", "watermarking_native_diagnostics_v1")
            .replace("watermarking_full_payload_scratch_v1", "watermarking_native_diagnostics_scratch_v1")
            .replace("wmbench_full_payload.runner", "wmbench_native_diagnostics.runner")
            .replace("full_payload_settings.json", "native_diagnostics_settings.json")
            .replace("full_payload_canary_report.zip", "native_diagnostics_report.zip"))
    settings = migrate(base.SETTINGS).replace('MODELS = "all"',
        'MODELS = ["hidden_benchmark", "rosteals", "sepwater", "sepmark", "distseal_posthoc_dcae"]')
    settings = settings.replace("All 32 entries; unavailable prerequisites stay visible.", "Only these five diagnostic methods.")
    settings = settings.replace("MAX_HOURS = 16", "MAX_HOURS = 6")
    settings = settings.replace("manual_assets_root=MANUAL_ASSETS_ROOT)",
        'manual_assets_root=MANUAL_ASSETS_ROOT or str(BANK_ROOT / "_assets"))')
    settings = settings.replace("5 full-length messages per image; Native + PSNR39/42/45. No attacks.",
        "Same historical 5 images x 5 messages x 5 methods. Native diagnostics only.")
    settings = settings.replace("N_IMAGES = 5                 # Leave 5 for this canary. Do not launch 100 yet.",
        "N_IMAGES = 5                 # Fixed diagnostic scope; not a production benchmark.")
    bootstrap = migrate(base.BOOTSTRAP).replace("__CHECKSUM__", checksum)
    bootstrap = bootstrap.replace('payload = SOURCE_ARCHIVE.read_bytes() if SOURCE_ARCHIVE.is_file() else None',
        'CACHED_ARCHIVE = OUTPUT_ROOT / "_release" / (SOURCE_SHA256 + ".zip")\n'
        'archive_path = SOURCE_ARCHIVE if SOURCE_ARCHIVE.is_file() else CACHED_ARCHIVE\n'
        'payload = archive_path.read_bytes() if archive_path.is_file() else None')
    bootstrap = bootstrap.replace('PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]',
        'CACHED_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)\n'
        'if not CACHED_ARCHIVE.exists():\n'
        '    temporary = CACHED_ARCHIVE.with_suffix(".part")\n'
        '    temporary.write_bytes(payload)\n'
        '    temporary.replace(CACHED_ARCHIVE)\n\n'
        'if N_IMAGES != 5:\n'
        '    raise ValueError("This diagnostic replays exactly the original five-image canary")\n'
        'PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]')
    report = migrate(base.REPORT).replace(
        'Do not switch to 100 images until Native errors and blockers are reviewed.',
        'Review the differences before changing production code. Old results have not been replaced.')
    sections = [
        ("markdown", "# Native diagnostics — recovery\n\n"
         "Fixes the source-inspection crash before HiDDeN/RoSteALS measurements. "
         "Keep the original output folder and all five method IDs: verified SepWater/SepMark/DistSeal results "
         "are reused without preparing their environments or running their models again. "
         "No saved measurements or original run identities are overwritten. "
         "The recognized audit-only code update is recorded separately.\n\n"
         "**HiDDeN, RoSteALS, SepWater, SepMark, DistSeal post-hoc.** Same five historical COCO crops and five full-length messages. "
         "No attacks, PSNR matching, retraining or production changes.\n\n"
         "Use **A100 + high RAM**, matching the prior run. Models use isolated environments, one at a time. "
         "Read-only source: `MyDrive/watermarking_native_resolution_canary_v1`. "
         "New results: `MyDrive/watermarking_native_diagnostics_v1`.\n\n"
         f"Upload **{ARCHIVE}** when prompted. It contains readable Python files, not Base64 or weights. "
         "The source archive is cached on Drive for reconnection. Finished measurements are reused, including bit errors."),
        ("markdown", "## 1. Settings\n\nKeep the five-image scope, all five method IDs and the previous OUTPUT_ROOT. "
         "The completed three methods will be verified and skipped. Change BANK_ROOT only if the completed canary is in a different folder."),
        ("code", settings),
        ("markdown", "## 2. Drive and isolated controller\n\nThe historical source needs actual PNGs and metadata, not only its review ZIP."),
        ("code", bootstrap),
        ("markdown", "## 3. Diagnose\n\nReplay saved PNGs; re-embed with the same seed; inspect raw, clipped, RGB8 and PNG boundaries. "
         "Compare fixed serializers and same-latent AE controls. Completed stages survive a runtime reset. "
         "A finished cell is not a claim that all bits decoded correctly."),
        ("code", migrate(base.RUN)),
        ("markdown", "## 4. Results\n\nRead the boundary table and per-case records. Serializer alternatives are diagnostic, "
         "not automatically adopted fixes. This archive excludes image/tensor/weight files."),
        ("code", report),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"native-diag-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            ast.parse(source, filename=f"cell_{index}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(data)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"archive": ARCHIVE, "sha256": checksum, "notebook": NOTEBOOK,
        "source_files": len(contents), "python_modules": sum(n.endswith('.py') for n in contents),
        "code_cells": 4, "files": {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; source ZIP {len(data):,} bytes; SHA256 {checksum}")


if __name__ == "__main__":
    build()
