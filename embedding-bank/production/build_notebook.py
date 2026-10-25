"""Build the reproducible notebook and its ordinary, readable source archive."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
NAME = "COCO_Embedding_Bank_PRODUCTION"
# This is a notebook/bootstrap-only repair. Do not invalidate existing banks.
RESUME_SOURCE_SHA = "6d7ddfcd05abdaf46f97ab7b6b174bff098b3043edcf069db64041554263f7b5"
HOTFIX_NAME = NAME + "_MAGICKWAND_FIX"
AUTHOR_FIX_SHA = "d0323ddfca7589efc29ce9b5a6279619c95f75591de5074cf42239b733e13cc7"
RECOVERY_SHA = "42ccd248d1764e78cc4a662626b07dab63778d83c56037bf47f60dbcca11c8dd"
AUTHOR_NIGHT_SHA = "9f186b48fe81c1443b6db48d91a7c0b828b2ebe2b9246b732b59bb08ea0e8f61"

SETTINGS = '''from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile, io

N_IMAGES = 5               # First run: 5. Reviewed large run: 100. Same output directory.
MODELS = "all"             # All 29 reviewed entries; a subset is diagnostic only.
MAX_HOURS = 16             # Per-session ceiling, not a Colab lifetime guarantee.
DEVICE = "cuda"
CLEANUP = True             # Reclaim owned local model environments between methods.
SOURCE_ARCHIVE = Path("COCO_Embedding_Bank_PRODUCTION_source.zip")
MANUAL_ASSETS_ROOT = ""     # Optional previously downloaded author weights.

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if os.name != "posix":
    raise RuntimeError("Use Colab or Linux/NVIDIA for the full 29-method run. Native Windows is not supported.")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
BANK_ROOT = BASE / "watermarking_embedding_bank_v1"  # Frozen old _inputs only; read-only.
OUTPUT_ROOT = BASE / "watermarking_production_bank_v1"  # New bank; keep for N=100.
SCRATCH_ROOT = Path("/content/wmbench_production_scratch") if IN_COLAB else Path("/workspace/wmbench_scratch")
SETTINGS = dict(bank=str(BANK_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
    n_images=N_IMAGES, models=MODELS, device=DEVICE, max_hours=MAX_HOURS, cleanup=CLEANUP,
    download_author_assets=True, manual_assets_root=MANUAL_ASSETS_ROOT)
print("Read-only originals:", BANK_ROOT)
print("Persistent results:", OUTPUT_ROOT)
print(f"{N_IMAGES} images x 5 messages x 29 methods x 4 modes: at most {N_IMAGES*580:,} PNGs.")
print("Each method: save carriers, then blind-decode PNGs. No channel attacks.")
'''

AUTHOR_SYSTEM_DEPENDENCIES = '''# Only RoSteALS/SepWater need these author-import system libraries.
# Restore the setup used by RoSteALS_AUTHOR_FIX and RoSteALS_SepWater_RECOVERY.
# No model source, checkpoint, Python lock, payload or matching rule is changed.
import ctypes
import ctypes.util

def ensure_author_system_dependencies(models, in_colab):
    affected = {"rosteals", "sepwater"}
    selected = affected if models == "all" else affected.intersection(models)
    if not selected:
        return {}

    requirements = {
        "libmagickwand-dev": (
            ("MagickWand-6.Q16", "MagickWand-6.Q16HDRI", "MagickWand-7.Q16HDRI",
             "MagickWand-7.Q16", "MagickWand"), "MagickWandGenesis"),
        "libgl1": (("GL",), None),
    }

    def inspect_libraries():
        found = {}
        for package, (names, symbol) in requirements.items():
            for name in names:
                path = ctypes.util.find_library(name)
                if not path:
                    continue
                try:
                    library = ctypes.CDLL(path)
                    if symbol:
                        getattr(library, symbol)
                except (OSError, AttributeError):
                    continue
                found[package] = path
                break
        return found

    found = inspect_libraries()
    missing = [package for package in requirements if package not in found]
    if missing:
        install = ["apt-get", "-qq", "install", "-y", "--no-install-recommends", *missing]
        if not in_colab or not shutil.which("apt-get") or os.geteuid() != 0:
            raise RuntimeError(
                "RoSteALS/SepWater system libraries are missing. On Debian/Ubuntu run: "
                "sudo apt-get update && sudo " + " ".join(install)
                + ". Then rerun this setup cell; keep the existing output directory.")
        print("Installing missing system libraries for " + ", ".join(sorted(selected))
              + ": " + ", ".join(missing), flush=True)
        subprocess.run(["apt-get", "-qq", "update"], check=True, timeout=600)
        subprocess.run(install, check=True, timeout=600)
        found = inspect_libraries()
        missing = [package for package in requirements if package not in found]
        if missing:
            raise RuntimeError("System-library preflight failed after installation: "
                               + ", ".join(missing) + ". No model has been started.")
    print("RoSteALS/SepWater system-library preflight passed:", found, flush=True)
    return found

AUTHOR_SYSTEM_LIBRARIES = ensure_author_system_dependencies(MODELS, IN_COLAB)
'''

REPORT = '''subprocess.run([str(CONTROLLER), "-m", "wmbench_bank.runner", "report",
                "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
summary = json.loads((OUTPUT_ROOT / "_reports/summary.json").read_text())
print("Technically complete:", summary["all_selected_technical_complete"])
print("All matched targets available:", summary["all_targets_available"])
print("Saved PNGs:", summary["saved_carriers"], "/ planned", summary["planned_carriers"])
# Small review archive: messages, records, audit files and logs, NOT model weights or PNG bank.
review = OUTPUT_ROOT / "production_bank_report.zip"
with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as archive:
    candidates = [OUTPUT_ROOT / "trial.json"]
    candidates += list((OUTPUT_ROOT / "_reports").rglob("*"))
    for method in json.loads((OUTPUT_ROOT / "trial.json").read_text())["models"]:
        candidates += list((OUTPUT_ROOT / method / "_meta").rglob("*"))
    candidates += list((OUTPUT_ROOT / "_author/_meta").rglob("*"))
    for path in candidates:
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".txt", ".log", ".lock"}:
            archive.write(path, path.relative_to(OUTPUT_ROOT))
print("Review archive (contains expected/decoded messages):", review)
print("Full bank: copy the ENTIRE output folder to the external SSD, e.g. E:/watermarking_production_bank_v1.")
print("Only after reviewing the canary, set N_IMAGES=100 and rerun the same notebook with the same paths/ZIP.")
'''


def checked_zip(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Historical evidence release changed: {path}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}


def build():
    spec = importlib.util.spec_from_file_location("bootstrap_builder", BASE / "full-payload/build_notebook.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    contents = {}
    for directory in ("src", "registry", "locks"):
        for path in sorted((BASE / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".lock") and "__pycache__" not in path.parts:
                contents[path.relative_to(BASE).as_posix()] = path.read_bytes()
    for source, package in ((BASE / "full-payload/wmbench_full_payload", "wmbench_full_payload"),
                            (BASE / "native-resolution/wmbench_native_canary", "wmbench_native_canary"),
                            (ROOT / "wmbench_bank", "wmbench_bank"),
                            (BASE / "full-payload/diagnostic_helpers", "bank_diagnostics")):
        for path in sorted(source.glob("*.py")):
            contents[f"src/{package}/{path.name}"] = path.read_bytes()
    # Reuse the ACTUAL fixed release, not the older on-disk author-night code.
    fixed = checked_zip(BASE / "rosteals-recovery/dist/RoSteALS_AUTHOR_FIX_source.zip", AUTHOR_FIX_SHA)
    recovery = checked_zip(BASE / "author-recovery/dist/RoSteALS_SepWater_RECOVERY_source.zip", RECOVERY_SHA)
    night = checked_zip(BASE / "author-night/dist/Three_Method_AUTHOR_NIGHT_source.zip", AUTHOR_NIGHT_SHA)
    for name, data in fixed.items():
        if name.startswith("authornight/"):
            contents["src/"+name] = data
    for method in ("rosteals", "sepwater", "distseal"):
        source = fixed if method == "rosteals" else recovery if method == "sepwater" else night
        contents[f"authors/locks/{method}.txt"] = source[f"locks/{method}.txt"]
    for path in (ROOT / "locks").iterdir():
        contents["locks/"+path.name] = path.read_bytes()
    contents["src/wmbench_bank/build_constraints.py"] = (BASE / "four-method-night/wmbench_quality_night/environment.py").read_bytes()
    for filename in ("README.md", "VALIDATION.md"):
        contents[filename] = (ROOT / filename).read_bytes()
    contents["THIRD_PARTY.md"] = (BASE / "THIRD_PARTY.md").read_bytes()
    contents["reference/evaluation-baselines.md"] = (BASE.parent / "docs/evaluation-baselines.md").read_bytes()
    contents["reference/author_release_provenance.json"] = json.dumps({
        "rosteals_author_fix_sha256": AUTHOR_FIX_SHA, "recovery_sha256": RECOVERY_SHA,
        "distseal_author_night_sha256": AUTHOR_NIGHT_SHA,
        "editguard_native_report_sha256": "219d74cecf34b87608a5eeba7dff80fa4794f644e9105cf1d1c360164ac4abc8",
        "note": "Author inference modules and locked dependencies reused; no old measurements imported."}, indent=2).encode()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    contents["release.json"] = json.dumps({"schema": "production-release-v1", "files": {
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != RESUME_SOURCE_SHA:
        raise ValueError("Notebook-only hotfix must preserve the original source ZIP byte for byte")
    bootstrap = previous.BOOTSTRAP.replace("Embedding_Bank_FULL_PAYLOAD_CANARY", NAME).replace("__CHECKSUM__", checksum)
    bootstrap = bootstrap.replace('payload = SOURCE_ARCHIVE.read_bytes() if SOURCE_ARCHIVE.is_file() else None', '''cached_source = OUTPUT_ROOT / "_release" / SOURCE_ARCHIVE.name
source_candidate = SOURCE_ARCHIVE if SOURCE_ARCHIVE.is_file() else cached_source
payload = source_candidate.read_bytes() if source_candidate.is_file() else None''')
    bootstrap = bootstrap.replace('PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]', '''cached_source.parent.mkdir(parents=True, exist_ok=True)
if not cached_source.exists():
    cached_source.write_bytes(payload)
# Old author dependencies contain a small native BCH extension.
if IN_COLAB and not shutil.which("g++"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential"], check=True)
if not shutil.which("g++"):
    raise RuntimeError("Install build-essential on the Linux server before running author environments.")
PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]''')
    marker = 'PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]'
    if bootstrap.count(marker) != 1:
        raise ValueError("Expected exactly one source-extraction boundary")
    bootstrap = bootstrap.replace(marker, AUTHOR_SYSTEM_DEPENDENCIES + "\n" + marker)
    bootstrap = bootstrap.replace('"full_payload_settings.json"', '"production_settings.json"')
    bootstrap += '''\nsubprocess.run([str(CONTROLLER), "-m", "wmbench_bank.runner", "validate",
                "--settings", str(CONFIG)], env=ENV, check=True)
'''
    run = previous.RUN.replace("wmbench_full_payload.runner", "wmbench_bank.runner")
    sections = [
        ("markdown", "# COCO embedding bank: reviewed configurations\n\n"
         "29 methods, five full messages per image, Native + Matched PSNR39/42/45. "
         "Set **N_IMAGES=5** for the first end-to-end check; after review use **100** with the same result root. "
         "This creates reusable PNGs and clean decoding results, **not channel attacks**.\n\n"
         "Use Linux/NVIDIA (Colab A100 + high RAM matches the author diagnostics). "
         "Full new 29-method CUDA execution has not been performed by this release's local tests. "
         "Upload the matching source ZIP; sources are readable Python, not Base64. Old results are read-only."),
        ("markdown", "## 1. Settings\n\nN_IMAGES is the only change needed for the larger cohort. "
         "BANK_ROOT must contain the frozen historical _inputs; OUTPUT_ROOT is a separate new bank."),
        ("code", SETTINGS),
        ("markdown", "## 2. Drive, release and input validation\n\nThe source ZIP is cached on Drive for reconnects. "
         "Heavy model dependencies never enter the notebook kernel.\n\n"
         "**RoSteALS/SepWater system-library fix:** restore the missing MagickWand/OpenGL preflight "
         "from the earlier author notebooks. The source ZIP and experiment identity are unchanged. "
         "Keep the same OUTPUT_ROOT and N_IMAGES=5 to finish the failed methods; completed methods "
         "are verified and reused, not recomputed. No results are deleted."),
        ("code", bootstrap),
        ("markdown", "## 3. Create carriers and decode them without attacks\n\nOne isolated method at a time. "
         "Saved PNGs, keys, messages and clean scores persist on Drive. Nonzero BER is a measurement, not a retry request."),
        ("code", run),
        ("markdown", "## 4. Report\n\nSafe after a partial run. No model is loaded by the reporting step. "
         "Complete, error-free decoding and available matched targets are different properties."),
        ("code", REPORT),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"id": f"production-{index:02d}", "cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            ast.parse(source, filename=f"cell_{index}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / f"{NAME}_source.zip").write_bytes(payload)
    notebook_text = json.dumps(notebook, ensure_ascii=False, indent=1)+"\n"
    (dist / f"{NAME}.ipynb").write_text(notebook_text, encoding="utf-8")
    (dist / f"{HOTFIX_NAME}.ipynb").write_text(notebook_text, encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"sha256": checksum, "files": len(contents),
        "python_modules": sum(n.endswith('.py') for n in contents), "code_cells": 4,
        "gpu_validation": "not_executed", "notebook_hotfix": "rosteals-sepwater-system-libraries-v1",
        "source_archive_unchanged": True}, indent=2)+"\n", encoding="utf-8")
    print(f"Built {NAME}: {len(contents)} files, SHA256 {checksum}")
    return dist


if __name__ == "__main__":
    build()
