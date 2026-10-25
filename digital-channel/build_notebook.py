"""Build a readable notebook and deterministic source ZIP; no embedded Base64."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
NAME = "COCO_Digital_Channel_RESUMABLE"
PRODUCTION_SHA = "6d7ddfcd05abdaf46f97ab7b6b174bff098b3043edcf069db64041554263f7b5"
HOTFIX_SHA = "0d7406e6193f5496fbc60f46157f86130d338eb2bdee8d9d76a7b7a65a5bc85b"

SETTINGS = '''from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile

N_IMAGES = 5                  # After review: change ONLY this to 100; keep the same output.
MODELS = "all"                # All 29; or a list of exact method IDs for diagnostics.
MAX_HOURS = 16                # A session budget, NOT a guarantee of Colab uptime.
SSD_RESERVE_GIB = 60          # Leave SSD room for one isolated decoder environment.
PAIRWISE_TABLES = True
SOURCE_ARCHIVE = Path("COCO_Digital_Channel_RESUMABLE_source.zip")

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name != "posix":
    raise RuntimeError("Full 29-method execution requires Linux/NVIDIA: use Colab or a GPU server.")
if IN_COLAB:
    drive.mount("/content/drive")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
BANK_ROOT = BASE / "watermarking_production_bank_v1"       # READ ONLY: completed embedding bank.
OUTPUT_ROOT = BASE / "watermarking_digital_channel_v1"    # Durable measurements and reports.
SCRATCH_ROOT = Path("/content/wmbench_digital_scratch") if IN_COLAB else Path("/workspace/wmbench_digital_scratch")
SETTINGS = dict(n_images=N_IMAGES, models=MODELS, max_hours=MAX_HOURS, device="cuda",
    bank=str(BANK_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
    ssd_reserve_gib=SSD_RESERVE_GIB, pairwise=PAIRWISE_TABLES, download_author_assets=True,
    manual_assets_root=str(BANK_ROOT / "_manual_assets"))
print("Bank (read-only):", BANK_ROOT)
print("Results:", OUTPUT_ROOT)
print("Temporary fallback:", OUTPUT_ROOT / "_temporary")
print("Per method: save 78 attacked PNGs per available carrier in ZIPs → decode → commit → remove ZIPs.")
'''

BOOTSTRAP = '''# Upload the matching small source ZIP once; a copy is kept on Drive.
EXPECTED_SHA = "__ZIP_SHA__"
SAVED_ZIP = OUTPUT_ROOT / "_release" / SOURCE_ARCHIVE.name
if not SOURCE_ARCHIVE.is_file() and SAVED_ZIP.is_file():
    SOURCE_ARCHIVE = SAVED_ZIP
if not SOURCE_ARCHIVE.is_file():
    if not IN_COLAB:
        raise FileNotFoundError("Place the matching source ZIP beside this notebook or set SOURCE_ARCHIVE.")
    from google.colab import files
    uploaded = files.upload()
    matches = [Path(name) for name in uploaded if hashlib.sha256(uploaded[name]).hexdigest() == EXPECTED_SHA]
    if len(matches) != 1:
        raise ValueError("Upload exactly the source ZIP supplied with this notebook.")
    SOURCE_ARCHIVE = matches[0]
data = SOURCE_ARCHIVE.read_bytes()
if hashlib.sha256(data).hexdigest() != EXPECTED_SHA:
    raise ValueError("Source ZIP checksum mismatch. Do not use a different notebook's ZIP.")
SAVED_ZIP.parent.mkdir(parents=True, exist_ok=True)
if not SAVED_ZIP.exists():
    SAVED_ZIP.write_bytes(data)
if hashlib.sha256(SAVED_ZIP.read_bytes()).hexdigest() != EXPECTED_SHA:
    raise ValueError("Saved release readback failed")
PROJECT = SCRATCH_ROOT / "release"
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
    for member in archive.infolist():
        target = (PROJECT / member.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()) or member.is_dir():
            raise ValueError("Unsafe or unexpected source ZIP entry")
        content = archive.read(member)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Source changed: {target}. Use a different empty SCRATCH_ROOT; retain OUTPUT_ROOT.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
import uv
UV = str(uv.find_uv_bin())
controller_dir = SCRATCH_ROOT / "controller"
CONTROLLER = controller_dir / "bin/python"
if not CONTROLLER.exists():
    subprocess.run([UV, "venv", "--python", "3.10.12", str(controller_dir)], check=True)
# This is the CPU attack/report + quality-metric environment, NOT a model environment.
subprocess.run([UV, "pip", "install", "--python", str(CONTROLLER),
    "numpy==1.26.4", "Pillow==10.4.0", "scipy==1.11.4", "scikit-image==0.22.0",
    "psutil==6.1.1", "gdown==5.2.0", "torch==2.5.1", "torchvision==0.20.1", "lpips==0.1.4"], check=True)
ENV = os.environ.copy()
ENV.update(PYTHONPATH=os.pathsep.join([str(PROJECT / "src"), str(PROJECT / "production/src")]),
    PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
    PATH=str(Path(UV).parent)+os.pathsep+ENV.get("PATH", ""),
    UV_PYTHON_INSTALL_DIR=str(SCRATCH_ROOT / "python"))
CONFIG = SCRATCH_ROOT / "digital-settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
print("Free SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
'''

RUN = '''# Stop any previous runner for this output folder before starting.
# After a disconnect rerun the same cells with the same bank/output and source ZIP.
command = [str(CONTROLLER), "-m", "wmbench_digital.runner", "run", "--settings", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", start_new_session=True)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Runner stopped. See persistent per-method logs; completed records remain safe.")
except BaseException:
    if process.poll() is None:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
    raise
'''

REPORT = '''subprocess.run([str(CONTROLLER), "-m", "wmbench_digital.runner", "report", "--settings", str(CONFIG)],
    env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
print("Aggregate tables:", OUTPUT_ROOT / "_reports")
print("Each method's tables:", OUTPUT_ROOT / "<method>/tables")
print("Detailed bits, errors and attack parameters:", OUTPUT_ROOT / "<method>/records/*.json.gz")
print("Copy OUTPUT_ROOT to your SSD after completion. Temporary attack ZIPs are intentionally not permanent.")
'''


def build():
    contents = {}
    data = (REPO / "embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION_source.zip").read_bytes()
    if hashlib.sha256(data).hexdigest() != PRODUCTION_SHA:
        raise ValueError("Production source changed; do not silently replace reviewed adapters")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        contents.update({"production/"+n: archive.read(n) for n in archive.namelist()})
    helper = (REPO / "embedding-bank/production/hotfix_hidden_fnns.py").read_bytes()
    if hashlib.sha256(helper).hexdigest() != HOTFIX_SHA:
        raise ValueError("Reviewed Meta helper changed")
    contents["src/hidden_fnns_reference.py"] = helper
    for path in sorted((ROOT / "src").rglob("*.py")):
        contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted((ROOT / "tests").glob("*.py")):
        contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("README.md", "VALIDATION.md"):
        contents[name] = (ROOT/name).read_bytes()
    for name in ("digital-channel-protocol.md", "evaluation-baselines.md", "model-catalogue.md"):
        contents["reference/"+name] = (REPO/"docs"/name).read_bytes()
    release = {"version": "1.0.0", "production_zip_sha256": PRODUCTION_SHA,
               "files": {p: hashlib.sha256(v).hexdigest() for p, v in contents.items() if not p.startswith("production/")},
               "protocol_source": "local repository docs/digital-channel-protocol.md; remote unavailable at build time"}
    contents["digital-release.json"] = (json.dumps(release, indent=2)+"\n").encode()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zip_path = dist / (NAME+"_source.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            if name.endswith(".py"):
                ast.parse(data.decode("utf-8"), filename=name)
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("production_builder", REPO/"embedding-bank/production/build_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    sections = [
        ("markdown", "# COCO digital channel — resumable\n\n29 reviewed decoders. Use the completed production embedding bank; no new embedding.\n\nEach method: save attacks in temporary ZIPs → decode saved PNGs → save results on Drive → remove attack ZIPs. Start with five images. No Base64 code.\n\nUse **Linux + NVIDIA**, preferably the same A100 runtime as the reviewed bank; high RAM. Download the matching source ZIP too. No GPU runtime is guaranteed to last overnight.\n"),
        ("code", SETTINGS), ("markdown", "## 1. Install the orchestrator and verify its source\n"),
        ("code", BOOTSTRAP.replace("__ZIP_SHA__", checksum)),
        ("markdown", "## 2. Restore the reviewed system dependencies\n\nOnly missing MagickWand/GL libraries are installed; no method is changed.\n"),
        ("code", builder.AUTHOR_SYSTEM_DEPENDENCIES),
        ("code", 'subprocess.run([str(CONTROLLER), "-m", "wmbench_digital.runner", "validate", "--settings", str(CONFIG)], env=ENV, check=True)\n'),
        ("markdown", "## 3. Run\n\nKeep the same output after disconnects. Valid results are never retried to improve BER. If one method fails technically, the run stops and retains its temporary ZIPs. Logs and completed rows are on Drive.\n"),
        ("code", RUN), ("markdown", "## 4. Reports\n\nReport rebuilding does not rerun attacks or load a decoder. Partial coverage remains visible.\n"),
        ("code", REPORT)]
    cells = []
    for index, (kind, text) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"digital-{index:02d}", "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                             "language_info": {"name": "python"}, "colab": {"name": NAME+".ipynb"}}}
    (dist/(NAME+".ipynb")).write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    (dist/(NAME+"_SHA256.txt")).write_text(checksum+"  "+zip_path.name+"\n", encoding="ascii")
    print("Built", dist/(NAME+".ipynb"), "ZIP SHA256", checksum)


if __name__ == "__main__":
    build()
