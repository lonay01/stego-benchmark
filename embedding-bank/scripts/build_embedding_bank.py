"""Build a readable bank notebook and its separate, checksum-pinned source ZIP."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmbench.embedding_bank import bank_registry  # noqa: E402

SETTINGS = """# Change paths here. Start at 5; after inspecting the report, set 100.
from pathlib import Path
import sys, os, json, shutil, subprocess, importlib.util
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

N_IMAGES = 5                         # 100 for the complete image bank
MODELS = "all"                       # All 32 requested catalogue entries
DEVICE = "cuda"                      # "cpu" only for a CPU-capable local test
IDENTITY_CHECK = True                # Decode saved PNGs once; no attacks
SOURCE_ARCHIVE = Path("COCO100_Embedding_Bank_source.zip")  # Upload this ZIP alongside the notebook

if IN_COLAB:
    OUTPUT_ROOT = Path("/content/drive/MyDrive/watermarking_embedding_bank_v1")
    SCRATCH_ROOT = Path("/content/watermarking_embedding_scratch")
    COCO_ROOT = Path("/content/drive/MyDrive/watermarking_coco100_visible/coco_fixed_512")
else:
    OUTPUT_ROOT = Path("E:/watermarking_embedding_bank_v1")
    SCRATCH_ROOT = Path.home() / "wmbench_embedding_scratch"
    COCO_ROOT = Path("E:/watermarking_inputs/coco_fixed_512")

MANUAL_ASSETS_ROOT = OUTPUT_ROOT / "_manual_assets"
SETTINGS = {
    "output_root": str(OUTPUT_ROOT), "scratch_root": str(SCRATCH_ROOT),
    "coco_root": str(COCO_ROOT), "manual_assets_root": str(MANUAL_ASSETS_ROOT),
    "n_images": N_IMAGES, "models": MODELS, "device": DEVICE,
    "identity_check": IDENTITY_CHECK, "cleanup": True,
    "max_hours": 12, "prepare_minutes": 60, "model_minutes": 180,
    "min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 150,
}
print("Output:", OUTPUT_ROOT)
print("Originals:", COCO_ROOT / "test", "and", COCO_ROOT / "manifest.csv")
print("Images per model:", N_IMAGES, "| One 30-bit message per image | Four modes")
"""

BOOTSTRAP = """# Source bundle only: model weights are downloaded one method at a time.
import hashlib, io, zipfile
if IN_COLAB:
    drive.mount("/content/drive")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Google Drive is not mounted; do not save results to ephemeral SSD")
elif not OUTPUT_ROOT.anchor or not Path(OUTPUT_ROOT.anchor).exists():
    raise FileNotFoundError("Connect the Samsung SSD or change OUTPUT_ROOT")
if DEVICE.startswith("cuda") and not shutil.which("nvidia-smi"):
    raise RuntimeError("NVIDIA GPU/driver not found. Select a GPU runtime in Colab, or use DEVICE='cpu' for a CPU-capable local test.")
if not shutil.which("git"):
    raise RuntimeError("Git is required for pinned model sources. Install Git and restart Jupyter.")

SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_SHA256 = "__CHECKSUM__"
if not SOURCE_ARCHIVE.is_file() and IN_COLAB:
    from google.colab import files
    print("Upload COCO100_Embedding_Bank_source.zip from the SAME release as this notebook.")
    files.upload()
if not SOURCE_ARCHIVE.is_file():
    raise FileNotFoundError(f"Source ZIP not found: {SOURCE_ARCHIVE.resolve()}. Upload the ZIP or set SOURCE_ARCHIVE to its path on Drive.")
payload = SOURCE_ARCHIVE.read_bytes()
if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("Source ZIP checksum mismatch. Use the ZIP supplied with THIS notebook; do not bypass this check.")
PROJECT = SCRATCH_ROOT / "source" / SOURCE_SHA256[:16]
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        target = (PROJECT / member.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError("Unsafe source archive path")
        content = archive.read(member)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Modified release source: {target}; use a fresh source directory")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

# The kernel never receives Torch or incompatible model dependencies.
try:
    import pip
except ImportError:
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
import uv
UV = str(uv.find_uv_bin())
controller_dir = SCRATCH_ROOT / "controller"
CONTROLLER = controller_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not CONTROLLER.exists():
    subprocess.run([UV, "venv", "--python", "3.10", str(controller_dir)], check=True)
subprocess.run([UV, "pip", "install", "--python", str(CONTROLLER),
    "numpy==1.26.4", "Pillow==10.4.0", "psutil==6.1.1"], check=True)
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(PROJECT / "src")
ENV["PATH"] = str(Path(UV).parent) + os.pathsep + ENV.get("PATH", "")
ENV["PYTHONUNBUFFERED"] = "1"
CONFIG = SCRATCH_ROOT / "embedding_bank_settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
if not (OUTPUT_ROOT / "_inputs/manifest.json").is_file():
    if not (COCO_ROOT / "test").is_dir() or not (COCO_ROOT / "manifest.csv").is_file():
        raise FileNotFoundError(f"Copy the existing exact COCO100 test/ + manifest.csv to {COCO_ROOT}. No new dataset will be substituted.")
print("Isolated controller:", CONTROLLER)
print("Free scratch SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
print("Ready. Saved images will survive model-environment cleanup.")
"""

RUN = """# Re-running resumes verified images, including after a Colab disconnect.
# Stop an older run before starting this cell in another notebook/runtime.
command = [str(CONTROLLER), "-m", "wmbench.embedding_bank", "run", "--config", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"Controller exited with code {code}. Saved records remain; inspect the logs.")
except BaseException:
    process.terminate()
    process.wait(timeout=30)
    raise
print("Review:", OUTPUT_ROOT / "_reports/REPORT.md")
print("A partial/blocked report is not a successful 32-model bank.")
"""

REPORT = """# Refresh from saved files; does not reload models or run inference.
subprocess.run([str(CONTROLLER), "-m", "wmbench.embedding_bank", "report",
    "--config", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
print("Transfer the WHOLE output folder to E:/watermarking_embedding_bank_v1, not only its PNGs.")
print("No large output ZIP is created automatically; this avoids duplicate disk usage.")
"""


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            item = zipfile.ZipInfo(name, (2026, 9, 18, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, content)
    return buffer.getvalue()


def build():
    registry = bank_registry(ROOT)
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in (ROOT / "src").rglob("*.py")}
    for model in registry.values():
        lock = model.get("validated_lock")
        if lock:
            files[lock] = (ROOT / lock).read_bytes()
    # Only the selected32 are discoverable in this standalone release.
    additions = json.loads((ROOT / "registry/bank_additions.json").read_text(encoding="utf-8"))
    additional_ids = {row["id"] for row in additions["models"]}
    files["registry/bank_models.json"] = json.dumps(
        [row for name, row in registry.items() if name not in additional_ids], indent=2
    ).encode()
    files["registry/bank_additions.json"] = (ROOT / "registry/bank_additions.json").read_bytes()
    files["docs/EMBEDDING_BANK.md"] = (ROOT / "docs/EMBEDDING_BANK.md").read_bytes()
    files["docs/EMBEDDING_BANK_VALIDATION.md"] = (
        ROOT / "docs/EMBEDDING_BANK_VALIDATION.md"
    ).read_bytes()
    files["README.md"] = files["docs/EMBEDDING_BANK.md"]
    files["THIRD_PARTY.md"] = (ROOT / "THIRD_PARTY.md").read_bytes()
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    files["compatibility/embedding_bank_hotfixes.json"] = (
        ROOT / "compatibility/embedding_bank_hotfixes.json"
    ).read_bytes()
    for relative in (
        "scripts/build_embedding_bank.py",
        "scripts/check_embedding_bank_release.py",
        "scripts/smoke_embedding_bank.py",
        "tests/test_embedding_bank.py",
        "tests/test_bank_additions.py",
        "tests/test_embedding_bank_resume.py",
        "tests/test_embedding_bank_hotfix.py",
        "tests/test_embedding_bank_release.py",
    ):
        files[relative] = (ROOT / relative).read_bytes()
    payload = zip_bytes(files)
    checksum = hashlib.sha256(payload).hexdigest()
    bootstrap = BOOTSTRAP.replace("__CHECKSUM__", checksum)
    pieces = [
        (
            "markdown",
            "# COCO100 — reusable embedding bank\n\n"
            "**Two files:** upload this notebook and the accompanying "
            "`COCO100_Embedding_Bank_source.zip`. The second cell asks for the ZIP "
            "if it is not present. Python source stays in ordinary files, not Base64.\n\n"
            "Prepare Native and PSNR39/42/45 PNGs once, then reuse them in all three channels. "
            "No attacks, training, calibration or generation-time methods run here.\n\n"
            "One image-specific 30-bit message is shared by all methods and modes. "
            "Start with 5 images, inspect the report, then change only N_IMAGES to 100. "
            "All 32 requested methods are listed, but missing weights, MATLAB and the "
            "unimplemented Camera-Shooting Instance adapter remain explicit blockers.\n\n"
            "**Colab:** use an NVIDIA GPU and high-RAM runtime if available. "
            "ChunkySeal is particularly demanding; no GPU is a compatibility guarantee. "
            "Results persist on Drive; cloud Colab cannot directly access your E: SSD.\n\n"
            "**Windows:** open this notebook in Jupyter, connect E:, and supply the existing "
            "COCO100 originals. Some upstream model environments are Linux-specific. "
            "The CPU checks do not establish compatibility of all 32 methods with Windows.",
        ),
        ("markdown", "## 1. Settings"),
        ("code", SETTINGS),
        (
            "markdown",
            "## 2. Source bundle and lightweight controller\n\n"
            "Upload the source ZIP supplied with this notebook when prompted. "
            "It contains readable Python source, not model checkpoints. Each model gets "
            "its own Python environment. Preserve the exact originals and manifest from the earlier runs.",
        ),
        ("code", bootstrap),
        (
            "markdown",
            "## 3. Embed, match, save and verify\n\n"
            "Completed images are reused by hash. Only pending work is computed. "
            "After model processing, disposable environments/weights are deleted; the bank is not. "
            "Re-running may reinstall weights but will not re-embed completed images. "
            "A technical decode failure is retryable; a measured poor BER is not re-sampled.",
        ),
        ("code", RUN),
        ("markdown", "## 4. Coverage report and SSD transfer"),
        ("code", REPORT),
        ("markdown", "## Protocol and limitations\n\n" + files["docs/EMBEDDING_BANK.md"].decode()),
    ]
    cells = []
    for index, (kind, source) in enumerate(pieces):
        cell = {
            "cell_type": kind,
            "id": f"bank-{index:02}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            ast.parse(source, filename=f"notebook_cell_{index}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
    }
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = "COCO100_Embedding_Bank_RESUMABLE.ipynb"
    contents = json.dumps(notebook, indent=1, ensure_ascii=False).encode("utf-8")
    (dist / name).write_bytes(contents)
    (ROOT / name).write_bytes(contents)
    (dist / "COCO100_Embedding_Bank_source.zip").write_bytes(payload)
    evidence = {
        "notebook_sha256": hashlib.sha256(contents).hexdigest(),
        "source_zip_sha256": checksum,
        "packaging": "external source ZIP; no Base64 in notebook",
        "models": list(registry),
        "statically_checked_code_cells": sum(c["cell_type"] == "code" for c in cells),
    }
    (dist / "embedding_bank_build.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    build()
