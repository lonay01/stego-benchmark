"""Deterministic readable source archive; the previous release stays immutable."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "four-method-review/dist"
BASE_SHA = "74d37a9c372beaaa10c45f3b44cb05d0a2eca08bfd302b97d620e0697e335a78"
ARCHIVE = "Four_Method_NIGHT_CHECK_source.zip"
NOTEBOOK = "Four_Method_NIGHT_CHECK.ipynb"


def build():
    payload = (BASE / "Four_Method_QUALITY_CHECK_source.zip").read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA:
        raise ValueError("Previous audited release changed")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        contents = {n: archive.read(n) for n in archive.namelist() if not n.startswith("tests/")}
    # Mirrors wmbench.io.digest, so only the exact reviewed v1 can be imported.
    code_hashes = {n: hashlib.sha256(data).hexdigest() for n, data in contents.items()
                   if n.startswith("src/") and n.endswith(".py")}
    prior_code = hashlib.sha256(json.dumps(code_hashes, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")).encode()).hexdigest()
    for path in (ROOT / "wmbench_quality_night").glob("*.py"):
        contents["src/wmbench_quality_night/" + path.name] = path.read_bytes()
    for path in (ROOT / "tests").glob("*.py"):
        contents["tests/" + path.name] = path.read_bytes()
    for path in (ROOT / "tests/fixtures").glob("*.txt"):
        contents["tests/fixtures/" + path.name] = path.read_bytes()
    for file in ("README.md", "VALIDATION.md"):
        contents[file] = (ROOT / file).read_bytes()
    contents["local_source_smoke.py"] = (ROOT / "local_source_smoke.py").read_bytes()
    smoke = ROOT / "validation/local_source_smoke.json"
    if smoke.is_file():
        contents["validation/local_source_smoke.json"] = smoke.read_bytes()
    contents["NIGHT_BASE.json"] = json.dumps({"sha256": BASE_SHA,
        "prior_code_sha256": prior_code, "note": "Unchanged inference modules/registry/locks; new diagnostic controller only"}, indent=2).encode()
    contracts = json.loads(contents["source_contracts.json"])
    contracts["sepwater"]["files"].update({
        "pre_trained_models/inference.yaml": "7a984a4c8a77e69b1892afb4ad150c59655586ed",
        "control_Latent_Diffusion_Model/autoencoder3.py": "797b7b24b4813e16db353b029409978da5f8fd5d",
    })
    contents["source_contracts.json"] = json.dumps(contracts, indent=2).encode()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    source_zip = stream.getvalue()
    checksum = hashlib.sha256(source_zip).hexdigest()
    old = json.loads((BASE / "Four_Method_QUALITY_CHECK.ipynb").read_text("utf-8"))
    code = ["".join(c["source"]) for c in old["cells"] if c["cell_type"] == "code"]
    settings = '''from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile, io

N_IMAGES = 20              # 5 historical + 15 additional; choose 5..100 BEFORE first run.
MAX_HOURS = 12             # Safety budget, not an ETA or guaranteed Colab session duration.
MODELS = ["sepwater", "editguard", "rosteals", "distseal_posthoc_dcae"]
DEVICE = "cuda"
SOURCE_ARCHIVE = Path("Four_Method_NIGHT_CHECK_source.zip")
DOWNLOAD_AUTHOR_ASSETS = True
MANUAL_ASSETS_ROOT = ""
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name == "nt":
    raise RuntimeError("Run this frozen GPU diagnostic in Colab or Linux, not native Windows")
DRIVE_ROOT = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
BANK_ROOT = DRIVE_ROOT / "watermarking_native_resolution_canary_v1"     # READ ONLY
PREVIOUS_ROOT = DRIVE_ROOT / "watermarking_four_method_quality_v1"     # READ ONLY
OUTPUT_ROOT = DRIVE_ROOT / "watermarking_four_method_night_v2"         # NEW results
SCRATCH_ROOT = Path("/content/watermarking_four_method_night_scratch_v2") if IN_COLAB else Path("/workspace/night-scratch")
SETTINGS = dict(bank=str(BANK_ROOT), previous=str(PREVIOUS_ROOT), output=str(OUTPUT_ROOT),
    scratch=str(SCRATCH_ROOT), n_images=N_IMAGES, models=MODELS, device=DEVICE,
    max_hours=MAX_HOURS, cleanup=True, download_author_assets=DOWNLOAD_AUTHOR_ASSETS,
    manual_assets_root=MANUAL_ASSETS_ROOT or str(BANK_ROOT / "_assets"))
print(f"Planned: {N_IMAGES * 5 * 7} image/message/configuration cases; same five messages per image.")
print("Previous results will be copied only after compatibility and artifact checks.")
print("New results:", OUTPUT_ROOT)
'''
    bootstrap = (code[1].replace(BASE_SHA, checksum)
        .replace("Four_Method_QUALITY_CHECK", "Four_Method_NIGHT_CHECK")
        .replace("quality_settings.json", "night_settings.json")
        .replace('if N_IMAGES != 5:\n    raise ValueError("This diagnostic uses exactly the same five source images")',
                 'if type(N_IMAGES) is not int or not 5 <= N_IMAGES <= 100:\n    raise ValueError("Choose 5..100 images")'))
    # Reject overlapping previous/output paths before caching any files.
    bootstrap = bootstrap.replace('if not shutil.which("git"):',
        'for other in (BANK_ROOT, OUTPUT_ROOT, SCRATCH_ROOT):\n'
        '    if PREVIOUS_ROOT.resolve().is_relative_to(other.resolve()) or other.resolve().is_relative_to(PREVIOUS_ROOT.resolve()):\n'
        '        raise ValueError("Previous results must remain separate and read-only")\n'
        'if not shutil.which("git"):')
    run = code[2].replace("wmbench_quality_review.runner", "wmbench_quality_night.runner")
    run = run.replace('print("Queue finished or time budget reached. Inspect the report, not only this message.")', '''summary = json.loads((OUTPUT_ROOT / "_reports/summary.json").read_text())
print(f"Measured {summary['measured']}/{summary['expected']}; complete={summary['complete']}")
print("Saved report:", OUTPUT_ROOT / "Four_Method_NIGHT_report.zip")''')
    report = code[3].replace("wmbench_quality_review.runner", "wmbench_quality_night.runner").replace(
        "Four_Method_QUALITY_report.zip", "Four_Method_NIGHT_report.zip")
    sections = [
        ("markdown", "# Four-method overnight quality check\n\n"
         "RoSteALS · SepWater · DistSeal · EditGuard. **20 images, five full messages, seven configurations: 700 cases.** "
         "Up to 100 compatible completed v1 cases are reused; the default run measures 600 remaining cases. "
         "Historical and additional images are reported separately. Native only, no attacks or matched PSNR.\n\n"
         "**Use A100 + high RAM**, as in the prior quality run. One isolated model environment at a time. "
         "Results and logs are saved to Google Drive after every case. This does not guarantee that Colab will stay alive overnight. "
         "Do not run the old and new queues simultaneously. This notebook has local CPU/static tests, not a completed four-model GPU validation."),
        ("markdown", "## 1. Settings\n\nDefault N_IMAGES=20 is the expanded diagnostic. "
         "Changing N after starting requires a new OUTPUT_ROOT. MAX_HOURS can be changed when resuming."),
        ("code", settings),
        ("markdown", "## 2. Mount Drive and load readable Python sources\n\nUpload `Four_Method_NIGHT_CHECK_source.zip` when prompted. "
         "Wait until Drive is authorized and the ZIP is accepted before leaving. It is cached for subsequent sessions; no Base64 code is embedded."),
        ("code", bootstrap),
        ("markdown", "## 3. Run / resume\n\nThe broken SepWater 256px paths are replaced by declared 224px profiles. "
         "EditGuard is restored from the full historical installed-package snapshot. Normal profile errors do not block sibling profiles; "
         "evidence mismatches block that method. Nonzero BER is preserved, not retried. "
         "A disconnect may interrupt a case; run cells 1–3 again with the same files, paths and GPU to resume."),
        ("code", run),
        ("markdown", "## 4. Report\n\nThis cell can also report a partial run. "
         "Read `complete`, coverage, consistency checks and error reasons separately. Send the small report ZIP, not model weights."),
        ("code", report),
    ]
    cells = []
    for index, (kind, text) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"night-{index:02d}", "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            compile(text, f"cell_{index}", "exec")
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(source_zip)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"sha256": checksum, "base_sha256": BASE_SHA,
        "prior_code_sha256": prior_code, "python_modules": sum(n.endswith('.py') for n in contents),
        "notebook": NOTEBOOK, "archive": ARCHIVE, "code_cells": 4}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}, source {len(source_zip):,} bytes, SHA256 {checksum}")


if __name__ == "__main__":
    build()
