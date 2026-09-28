"""Build a readable-source canary notebook and its checksum-pinned ZIP."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ARCHIVE = "Embedding_Bank_FULL_PAYLOAD_CANARY_source.zip"
NOTEBOOK = "Embedding_Bank_FULL_PAYLOAD_CANARY.ipynb"

SETTINGS = '''from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile, io

N_IMAGES = 5                 # Leave 5 for this canary. Do not launch 100 yet.
MODELS = "all"              # All 32 entries; unavailable prerequisites stay visible.
DEVICE = "cuda"
MAX_HOURS = 16              # Safety ceiling, not a duration estimate or Colab guarantee.
SOURCE_ARCHIVE = Path("Embedding_Bank_FULL_PAYLOAD_CANARY_source.zip")
DOWNLOAD_AUTHOR_ASSETS = True
MANUAL_ASSETS_ROOT = ""     # Optional directory with <method>/<checkpoint>.

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BANK_ROOT = Path("/content/drive/MyDrive/watermarking_embedding_bank_v1")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/watermarking_full_payload_canary_v1")
    SCRATCH_ROOT = Path("/content/watermarking_full_payload_scratch_v1")
else:
    BANK_ROOT = Path("E:/watermarking_embedding_bank_v1")
    OUTPUT_ROOT = Path("E:/watermarking_full_payload_canary_v1")
    SCRATCH_ROOT = Path.home() / "watermarking_full_payload_scratch_v1"

SETTINGS = dict(bank=str(BANK_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
                n_images=N_IMAGES, models=MODELS, device=DEVICE, max_hours=MAX_HOURS,
                cleanup=True, download_author_assets=DOWNLOAD_AUTHOR_ASSETS,
                manual_assets_root=MANUAL_ASSETS_ROOT)
print("Read-only originals:", BANK_ROOT)
print("NEW durable results:", OUTPUT_ROOT)
print("5 full-length messages per image; Native + PSNR39/42/45. No attacks.")
'''

BOOTSTRAP = '''if IN_COLAB:
    drive.mount("/content/drive")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Google Drive is not mounted")
if not (BANK_ROOT / "_inputs/manifest.json").is_file():
    raise FileNotFoundError(f"Set BANK_ROOT to the old bank containing _inputs/manifest.json: {BANK_ROOT}")
for left, right in ((BANK_ROOT, OUTPUT_ROOT), (BANK_ROOT, SCRATCH_ROOT), (OUTPUT_ROOT, SCRATCH_ROOT)):
    if left.resolve().is_relative_to(right.resolve()) or right.resolve().is_relative_to(left.resolve()):
        raise ValueError("Old bank, output and scratch must be separate, non-nested directories")
if not shutil.which("git"):
    raise RuntimeError("Git is required for the pinned author sources")
if DEVICE == "cuda" and not shutil.which("nvidia-smi"):
    raise RuntimeError("Select an NVIDIA GPU runtime first")
if (OUTPUT_ROOT / "bank.json").exists():
    raise ValueError("OUTPUT_ROOT points to a legacy bank. Use a NEW folder.")

SOURCE_SHA256 = "__CHECKSUM__"
payload = SOURCE_ARCHIVE.read_bytes() if SOURCE_ARCHIVE.is_file() else None
if IN_COLAB and (payload is None or hashlib.sha256(payload).hexdigest() != SOURCE_SHA256):
    from google.colab import files
    print("Upload Embedding_Bank_FULL_PAYLOAD_CANARY_source.zip supplied with this notebook.")
    uploads = files.upload()
    candidates = [data for data in uploads.values() if hashlib.sha256(data).hexdigest() == SOURCE_SHA256]
    if not candidates:
        raise ValueError("Wrong source ZIP: checksum does not match this notebook")
    payload = candidates[0]
if payload is None or hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("Place the matching ZIP beside the notebook or set SOURCE_ARCHIVE")

PROJECT = SCRATCH_ROOT / "release" / SOURCE_SHA256[:16]
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        target = (PROJECT / item.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()) or item.is_dir():
            raise ValueError("Unexpected archive member")
        content = archive.read(item)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Release source modified: {target}; use fresh scratch")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

# Only small controller tools enter this kernel. Each model gets a separate venv.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
import uv
UV = str(uv.find_uv_bin())
controller_dir = SCRATCH_ROOT / "controller"
CONTROLLER = controller_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not CONTROLLER.exists():
    subprocess.run([UV, "venv", "--python", "3.10", str(controller_dir)], check=True)
subprocess.run([UV, "pip", "install", "--python", str(CONTROLLER), "numpy==1.26.4",
                "Pillow==10.4.0", "psutil==6.1.1", "gdown==5.2.0"], check=True)
ENV = os.environ.copy()
ENV.update(PYTHONPATH=str(PROJECT / "src"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="1", PATH=str(Path(UV).parent) + os.pathsep + ENV.get("PATH", ""))
CONFIG = SCRATCH_ROOT / "full_payload_settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Free local SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
print("Ready. Stop any previous runner for this output directory before starting.")
'''

RUN = '''# After reconnecting: run cells 1-3 again with the SAME paths and source ZIP.
# Finished measurements (including nonzero BER) are reused, not improved by retrying.
command = [str(CONTROLLER), "-m", "wmbench_full_payload.runner", "run", "--settings", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", start_new_session=(os.name != "nt"))
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"Controller exit={code}; completed results remain in {OUTPUT_ROOT}")
except BaseException:
    if process.poll() is None:
        if os.name != "nt":
            import signal
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    raise
print("Queue finished or time budget reached. Inspect the report, not only this message.")
'''

REPORT = '''subprocess.run([str(CONTROLLER), "-m", "wmbench_full_payload.runner", "report",
                "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))

# Review archive includes diagnostic records/logs, not weights or the large image bank.
review = OUTPUT_ROOT / "full_payload_canary_report.zip"
with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_ROOT.rglob("*"):
        if not path.is_file() or "_assets" in path.relative_to(OUTPUT_ROOT).parts:
            continue
        if path.suffix.lower() in {".json", ".csv", ".md", ".log", ".txt", ".lock"}:
            archive.write(path, path.relative_to(OUTPUT_ROOT).as_posix())
print("Send this archive for review:", review)
print("Do not switch to 100 images until Native errors and blockers are reviewed.")
'''


def build():
    contents = {}
    for directory in ("src", "registry", "locks"):
        for path in sorted((BASE / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".lock") and "__pycache__" not in path.parts:
                contents[path.relative_to(BASE).as_posix()] = path.read_bytes()
    for path in sorted((ROOT / "wmbench_full_payload").glob("*.py")):
        contents["src/wmbench_full_payload/" + path.name] = path.read_bytes()
    # Reuse reviewed stage operations and checkpoint provenance, without modifying
    # the untracked historical diagnostic project or its common30 scorer.
    for name in ("__init__.py", "core.py", "worker.py", "assets.py"):
        contents["src/bank_diagnostics/" + name] = (ROOT / "diagnostic_helpers" / name).read_bytes()
    contents["README.md"] = (ROOT / "README.md").read_bytes()
    contents["VALIDATION.md"] = (ROOT / "VALIDATION.md").read_bytes()
    contents["THIRD_PARTY.md"] = (BASE / "THIRD_PARTY.md").read_bytes()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, data)
    payload = archive.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    sections = [
        ("markdown", "# Full-payload embedding canary\n\n5 COCO images × 5 random full-length messages per method. Native + Matched PSNR39/42/45. "
         "No channel attacks and no changes to the old image bank.\n\nUse an **A100 GPU with high system RAM** for the first run, as in the previous diagnostics. "
         "All methods run sequentially in isolated environments. This notebook does not purchase resources.\n\n"
         "Upload the matching **" + ARCHIVE + "** when asked. It contains readable Python files; no Base64, model weights or COCO images.\n\n"
         "**Limitations:** Camera-Shooting Instance has no verified adapter; Fang requires licensed MATLAB; OmniGuard requires a matching copyright checkpoint. "
         "These remain explicit blockers, not fake passes. FNNS carries 32 variable bits in 48 positions; InvisMark carries 94 in 100. "
         "DWT4096 and SSL2048 probe fixed-configuration structural limits, not guaranteed error-free capacity."),
        ("markdown", "## 1. Settings\n\nOnly edit paths if necessary. The old bank supplies the exact original PNGs, not its old 30-bit messages."),
        ("code", SETTINGS),
        ("markdown", "## 2. Drive, source ZIP and lightweight controller"),
        ("code", BOOTSTRAP.replace("__CHECKSUM__", checksum)),
        ("markdown", "## 3. Run the canary\n\nEach saved image and decoding measurement is checkpointed to Drive. "
         "Progress prints every 45 seconds during preparation/work. A time-budget stop is resumable. "
         "The bank can contain up to 3,200 marked PNGs at N=5, but completion and successful decoding are separate."),
        ("code", RUN),
        ("markdown", "## 4. Results and small review archive\n\nRun this cell even after a partial run. The report does not load models or recompute measurements."),
        ("code", REPORT),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"full-payload-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            ast.parse(source, filename=f"cell_{index}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(payload)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"archive": ARCHIVE, "sha256": checksum,
        "notebook": NOTEBOOK, "source_files": len(contents), "code_cells": 4,
        "gpu_validation": "not_executed", "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; ZIP {len(payload):,} bytes; SHA256 {checksum}")


if __name__ == "__main__":
    build()
