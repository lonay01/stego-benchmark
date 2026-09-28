"""Build diagnostics without changing the production notebook or source files."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

SETTINGS = '''from pathlib import Path
import os, sys, json, shutil, subprocess, hashlib, zipfile, io

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

N_IMAGES = 5
DEVICE = "cuda"
CHECK_BLOCKED_METHODS = True
DOWNLOAD_AUTHOR_ASSETS = True
MAX_HOURS = 12
SOURCE_ARCHIVE = Path("Embedding_Bank_DIAGNOSTICS_source.zip")

if IN_COLAB:
    BANK_ROOT = Path("/content/drive/MyDrive/watermarking_embedding_bank_v1")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/watermarking_embedding_diagnostics_v1")
    SCRATCH_ROOT = Path("/content/watermarking_embedding_diagnostics_scratch")
else:
    BANK_ROOT = Path("E:/watermarking_embedding_bank_v1")
    OUTPUT_ROOT = Path("E:/watermarking_embedding_diagnostics_v1")
    SCRATCH_ROOT = Path.home() / "watermarking_embedding_diagnostics_scratch"

SETTINGS = dict(bank=str(BANK_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
                n_images=N_IMAGES, device=DEVICE, check_blocked=CHECK_BLOCKED_METHODS,
                download_author_assets=DOWNLOAD_AUTHOR_ASSETS, max_hours=MAX_HOURS,
                cleanup=True)
print("Read-only bank:", BANK_ROOT)
print("Diagnostic results:", OUTPUT_ROOT)
print("Same historical images and messages. No attacks, training or production edits.")
'''

BOOTSTRAP = '''if IN_COLAB:
    drive.mount("/content/drive")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Drive is not mounted")
if not (BANK_ROOT / "_inputs/manifest.json").is_file():
    raise FileNotFoundError(f"Historical bank not found: {BANK_ROOT}. Set BANK_ROOT to the existing bank, not an empty folder.")
if not shutil.which("git"):
    raise RuntimeError("Git is required for the pinned author repositories")
if DEVICE == "cuda" and not shutil.which("nvidia-smi"):
    raise RuntimeError("Select a GPU runtime before running diagnostics")
for left, right in ((BANK_ROOT, OUTPUT_ROOT), (BANK_ROOT, SCRATCH_ROOT), (OUTPUT_ROOT, SCRATCH_ROOT)):
    if left.resolve().is_relative_to(right.resolve()) or right.resolve().is_relative_to(left.resolve()):
        raise ValueError("Bank, diagnostic results and scratch must be separate, non-nested folders")

SOURCE_SHA256 = "__CHECKSUM__"
payload = SOURCE_ARCHIVE.read_bytes() if SOURCE_ARCHIVE.is_file() else None
if IN_COLAB and (payload is None or hashlib.sha256(payload).hexdigest() != SOURCE_SHA256):
    from google.colab import files
    print("Upload Embedding_Bank_DIAGNOSTICS_source.zip supplied with THIS notebook.")
    uploads = files.upload()
    matching = [data for data in uploads.values() if hashlib.sha256(data).hexdigest() == SOURCE_SHA256]
    if not matching:
        raise ValueError("Source ZIP checksum mismatch")
    payload = matching[0]
if payload is None or hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("Place the matching source ZIP beside the notebook or set SOURCE_ARCHIVE")

PROJECT = SCRATCH_ROOT / "source" / SOURCE_SHA256[:16]
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for item in bundle.infolist():
        target = (PROJECT / item.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError("Unsafe archive path")
        content = bundle.read(item)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Release file was modified: {target}; use a fresh scratch directory")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

# Heavy model dependencies never enter the notebook kernel or controller.
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
ENV.update(PYTHONPATH=os.pathsep.join((str(PROJECT / "src"), str(PROJECT / "diagnostics"))),
           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
           PATH=str(Path(UV).parent) + os.pathsep + ENV.get("PATH", ""))
CONFIG = SCRATCH_ROOT / "diagnostic_settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Free scratch SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
print("Ready. Stop any older diagnostic run before proceeding.")
'''

RUN = '''# Same cell after a disconnect: measured cases are reused, including nonzero BER.
# Only technical failures/incomplete stages are retried; no new messages are sampled.
command = [str(CONTROLLER), "-m", "bank_diagnostics.controller", "run", "--settings", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", start_new_session=(os.name != "nt"))
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"Diagnostic controller exited with {code}; saved evidence remains in {OUTPUT_ROOT}")
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
print("Finished or session budget reached. Read REPORT.md; completion does not imply zero BER.")
'''

REPORT = '''subprocess.run([str(CONTROLLER), "-m", "bank_diagnostics.controller", "report",
                "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "REPORT.md").read_text(encoding="utf-8")))
print("Full stage results:", OUTPUT_ROOT / "stages.csv")
print("PSNR audit:", OUTPUT_ROOT / "matching/summary.json")
print("Author assets:", OUTPUT_ROOT / "author_assets/asset_report.json")

# A small review package: logs and JSON/CSV/Markdown, not large weights or image copies.
archive_path = OUTPUT_ROOT / "diagnostics_report.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
    for p in OUTPUT_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".csv", ".md", ".log", ".lock", ".txt"}:
            bundle.write(p, p.relative_to(OUTPUT_ROOT).as_posix())
print("Send this file for review:", archive_path)
'''


def build(recovery=False):
    files = {}
    for folder in ("src", "registry", "locks", "compatibility"):
        for p in sorted((PROJECT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".lock"):
                files[p.relative_to(PROJECT).as_posix()] = p.read_bytes()
    for p in ROOT.rglob("*.py"):
        if "__pycache__" not in p.parts:
            files[p.relative_to(PROJECT).as_posix()] = p.read_bytes()
    if recovery:
        # prepare_model's initializer deliberately uses only PROJECT/src on PYTHONPATH.
        # Include the diagnostic import shim there too, without editing production src.
        for p in (ROOT / 'bank_diagnostics').glob('*.py'):
            files['src/bank_diagnostics/' + p.name] = p.read_bytes()
    files["diagnostics/README.md"] = (ROOT / "README.md").read_bytes()
    if recovery:
        files['diagnostics/RECOVERY.md'] = (ROOT / 'RECOVERY.md').read_bytes()
        recovery_validation = ROOT / 'RECOVERY_VALIDATION.md'
        if recovery_validation.is_file():
            files['diagnostics/RECOVERY_VALIDATION.md'] = recovery_validation.read_bytes()
    validation = ROOT / "dist/VALIDATION.md"
    if validation.is_file():
        files["diagnostics/VALIDATION.md"] = validation.read_bytes()
    files["THIRD_PARTY.md"] = (PROJECT / "THIRD_PARTY.md").read_bytes()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in sorted(files.items()):
            item = zipfile.ZipInfo(name, (2026, 9, 20, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(item, content)
    zipped = payload.getvalue()
    checksum = hashlib.sha256(zipped).hexdigest()
    pieces = [
        ("markdown", "# Embedding bank — diagnostic recheck\n\nSeparate from the production notebook. The historical bank is read-only.\n\n"
         "Checks FIN and HiDDeN before/after PNG and resolution adaptation; repeats six mild-error Native methods; measures DistSeal autoencoder-only controls; audits failed scalar PSNR targets; downloads and checks available author assets for blocked methods.\n\n"
         "Upload the matching **Embedding_Bank_DIAGNOSTICS_source.zip** when asked. Source is ordinary Python files, not Base64. Use the A100/high-RAM runtime that ran your original bank if available. Each method runs alone; its temporary environment is removed afterwards.\n\n"
         "**Limits:** MATLAB licensing and unavailable Camera-Shooting Instance integration cannot be solved by this notebook. OmniGuard remains conditional on a genuine copyright checkpoint; public localization weights are not substituted."),
        ("markdown", "## 1. Paths and scope\n\nLeave N_IMAGES=5 for this check. The same images/messages are read from the existing bank. Changing the diagnostic scope requires a new OUTPUT_ROOT; this is not the production100 run."),
        ("code", SETTINGS),
        ("markdown", "## 2. Mount Drive and verify the source ZIP"),
        ("code", BOOTSTRAP.replace("__CHECKSUM__", checksum)),
        ("markdown", "## 3. Run all diagnostics\n\nProgress and model logs are on Drive. A disconnected Colab may require reinstalling dependencies, but completed diagnostic cases are reused. Technical errors and bit errors are reported separately. Do not run this output directory from two sessions at once."),
        ("code", RUN),
        ("markdown", "## 4. Review and export\n\nNo correction is applied to the main bank automatically. Review the evidence before changing its adapters or PSNR policy."),
        ("code", REPORT),
    ]
    archive_name = 'Embedding_Bank_DIAGNOSTICS_source.zip'
    notebook_name = 'Embedding_Bank_DIAGNOSTICS_RESUMABLE.ipynb'
    if recovery:
        archive_name = 'Embedding_Bank_DIAGNOSTICS_RECOVERY_source.zip'
        notebook_name = 'Embedding_Bank_DIAGNOSTICS_RECOVERY.ipynb'
        settings = SETTINGS.replace('watermarking_embedding_diagnostics_v1', 'watermarking_embedding_diagnostics_recovery_v1')
        settings = settings.replace('watermarking_embedding_diagnostics_scratch', 'watermarking_embedding_diagnostics_recovery_scratch')
        settings += '''
# Retry only the three technical failures. The eight completed methods stay untouched.
PREVIOUS_DIAGNOSTICS_ROOT = OUTPUT_ROOT.parent / "watermarking_embedding_diagnostics_v1"
SETTINGS.update(models=["editguard", "sepwater", "chunkyseal"], check_blocked=False,
                download_author_assets=False, audit_matching=False, dependency_repairs=True,
                previous_diagnostics=str(PREVIOUS_DIAGNOSTICS_ROOT), keep_failed_scratch=True,
                prepare_timeout_seconds={"chunkyseal": 10800, "editguard": 5400, "sepwater": 5400})
print("Recovery only:", SETTINGS["models"])
print("Previous diagnostic report remains in:", PREVIOUS_DIAGNOSTICS_ROOT)
'''
        pieces[0] = ('markdown', '# Diagnostic recovery — three failed methods\n\n'
                     'EditGuard: BasicSR dependency and torchvision import compatibility. '
                     'SepWater: pinned setuptools providing pkg_resources. '
                     'ChunkySeal: three-hour preparation allowance including checkpoint download.\n\n'
                     'Only these three methods are retried. The original bank and previous diagnostic '
                     'folder are read-only. Results go to a new recovery folder. '
                     'No FIN/HiDDeN or matched-quality changes are applied.\n\n'
                     'Use the same A100/high-RAM runtime. Upload the accompanying recovery source ZIP when asked. '
                     'Failed scratch is retained while the runtime survives; completed recovery cases are saved on Drive.')
        pieces[2] = ('code', settings)
        pieces[5] = ('markdown', '## 3. Retry EditGuard, SepWater and ChunkySeal\n\n'
                     'Do not run the old diagnostic notebook in parallel. The three-hour limit is a safety ceiling, not a runtime estimate.')
        pieces[-1] = ('code', REPORT.replace(
            'print("PSNR audit:", OUTPUT_ROOT / "matching/summary.json")',
            'print("Previous PSNR audit:", PREVIOUS_DIAGNOSTICS_ROOT / "matching/summary.json")'
        ).replace('print("Author assets:", OUTPUT_ROOT / "author_assets/asset_report.json")',
                  'print("Reused author weights:", OUTPUT_ROOT / "author_assets")'))
        pieces = [(kind, text.replace('Embedding_Bank_DIAGNOSTICS_source.zip', archive_name)) for kind,text in pieces]
    cells = []
    for i, (kind, source) in enumerate(pieces):
        cell = dict(cell_type=kind, id=f"diag-{i:02d}", metadata={}, source=source.splitlines(True))
        if kind == "code":
            ast.parse(source, filename=f"cell_{i}")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}})
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    content = json.dumps(notebook, ensure_ascii=False, indent=1).encode()
    (dist / notebook_name).write_bytes(content)
    (dist / archive_name).write_bytes(zipped)
    evidence = {"code_cells_compiled": sum(c["cell_type"] == "code" for c in cells),
                "source_sha256": checksum, "notebook_sha256": hashlib.sha256(content).hexdigest(),
                "production_source_policy": "Bundled verbatim; builder writes only diagnostics/dist",
                "files_in_source_bundle": len(files)}
    (dist / ('recovery_build.json' if recovery else 'build.json')).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', action='store_true')
    build(parser.parse_args().recovery)
