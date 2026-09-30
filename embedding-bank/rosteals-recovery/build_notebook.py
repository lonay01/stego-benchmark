"""Build a separate release; never rewrite the previous evidence/notebook ZIP."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PARENT_SHA = "42ccd248d1764e78cc4a662626b07dab63778d83c56037bf47f60dbcca11c8dd"
NAME = "RoSteALS_AUTHOR_FIX"


def inherited_files():
    parent = ROOT.parent / "author-recovery/dist/RoSteALS_SepWater_RECOVERY_source.zip"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Frozen recovery ZIP changed")
    with zipfile.ZipFile(parent) as archive:
        files = {n: archive.read(n) for n in archive.namelist()
                 if n.startswith(("authornight/", "authorrecovery/", "reference/")) or n == "locks/rosteals.txt"}
    original = files["authornight/environment.py"].decode()
    replacements = {
        'checkout("imagenetc", root, scratch) / "ImageNet-C"':
        'checkout("imagenetc", root, scratch) / "ImageNet-C" / "imagenet_c"',
        '("imagenetc", "ImageNet-C")': '("imagenetc", "ImageNet-C/imagenet_c")',
    }
    fixed = original
    for before, after in replacements.items():
        if fixed.count(before) != 1:
            raise ValueError("Expected import-root expression not found exactly once")
        fixed = fixed.replace(before, after)
    files["authornight/environment.py"] = fixed.encode()
    files["reference/import_fix.json"] = json.dumps({"parent_release": PARENT_SHA,
        "changed_file": "authornight/environment.py", "changes": replacements,
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
        "author_repositories_changed": False}, indent=2).encode()
    return files


def build():
    files = inherited_files()
    for folder in ("rostealsfix", "tests"):
        for path in sorted((ROOT / folder).glob("*.py")):
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("build_notebook.py", "README.md", "validate_release.py", "smoke_imports.py"):
        files[name] = (ROOT / name).read_bytes()
    for name, data in files.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = stream.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    settings = '''from pathlib import Path
import hashlib, io, os, shutil, subprocess, sys, zipfile

MAX_HOURS = 8  # Session budget; same notebook resumes unfinished work.
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name != "posix":
    raise RuntimeError("Use Colab A100 + high RAM or a Linux NVIDIA server.")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
PARENT_ROOT = BASE / "watermarking_rosteals_sepwater_recovery_v1"  # READ ONLY
OUTPUT_ROOT = BASE / "watermarking_rosteals_author_fix_v1"        # NEW RUN
SCRATCH_ROOT = Path("/content/rosteals_author_fix_scratch_v1") if IN_COLAB else Path("/workspace/rosteals-fix-scratch")
SOURCE_ARCHIVE = Path("RoSteALS_AUTHOR_FIX_source.zip")
SETTINGS = dict(parent=str(PARENT_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT), max_hours=MAX_HOURS)
print("RoSteALS only: same 100 original COCO images, 5 messages, 256px author input.")
print("500 raw100 + 500 BCH-text cases, then 35,000 raw100 corruption conditions.")
print("SepWater and DistSeal will NOT run. Previous results remain unchanged.")
print("Results:", OUTPUT_ROOT)
'''
    setup = '''if IN_COLAB:
    drive.mount("/content/drive")
from IPython.display import Markdown, display
paths = [p.resolve() for p in (PARENT_ROOT, OUTPUT_ROOT, SCRATCH_ROOT)]
for i, left in enumerate(paths):
    for right in paths[i+1:]:
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Parent, output and scratch must not overlap")
if not (PARENT_ROOT / "plan.json").is_file():
    raise FileNotFoundError(f"Need the previous recovery folder, not just its report ZIP: {PARENT_ROOT}")
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(SCRATCH_ROOT).free < 35 * 1024**3:
    raise RuntimeError("Need 35 GiB free temporary SSD. Do not delete Drive results.")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True))
EXPECTED_SHA = "__SHA__"
cache = OUTPUT_ROOT / "_release" / EXPECTED_SHA / SOURCE_ARCHIVE.name
payload = None
for candidate in (cache, SOURCE_ARCHIVE):
    if candidate.is_file():
        content = candidate.read_bytes()
        if hashlib.sha256(content).hexdigest() == EXPECTED_SHA:
            payload = content
            break
        print("Ignoring mismatched source ZIP (not deleting):", candidate)
if payload is None:
    if not IN_COLAB:
        raise FileNotFoundError("Place the matching source ZIP next to this notebook")
    print("Upload RoSteALS_AUTHOR_FIX_source.zip — NOT the old recovery ZIP.")
    uploaded = files.upload()
    candidates = [data for data in uploaded.values() if hashlib.sha256(data).hexdigest() == EXPECTED_SHA]
    if len(candidates) != 1:
        raise ValueError("No unique matching source ZIP")
    payload = candidates[0]
PROJECT = SCRATCH_ROOT / ("release-" + EXPECTED_SHA[:16])
# A notebook that previously imported another release must restart its kernel.
for module_name in ("authornight", "authorrecovery", "rostealsfix"):
    loaded = sys.modules.get(module_name)
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Restart the notebook kernel: another release is already imported")
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        target = (PROJECT / entry.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()) or target == PROJECT.resolve():
            raise ValueError("Unsafe ZIP path")
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("ZIP symlink rejected")
        if not entry.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
sys.path.insert(0, str(PROJECT))
from rostealsfix.migration import make_plan
from authornight.storage import atomic
PLAN = make_plan(SETTINGS, EXPECTED_SHA)
atomic(cache, payload)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
if IN_COLAB:
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "libmagickwand-dev", "build-essential", "libgl1"], check=True)
print("Ready. Next: real imports + 70 attack API checks -> full author GPU preflight -> missing work.")
'''.replace("__SHA__", checksum)
    queue = '''from rostealsfix.runner import run
from rostealsfix.report import bundle
try:
    SUMMARY = run(SETTINGS, EXPECTED_SHA)
finally:
    REPORT_ZIP = bundle(OUTPUT_ROOT)
    print("Partial/final report saved:", REPORT_ZIP)
'''
    report = '''from rostealsfix.report import report, bundle
from IPython.display import Markdown, display, FileLink
SUMMARY = report(OUTPUT_ROOT)
display(Markdown((OUTPUT_ROOT / "_reports/report.md").read_text()))
REPORT_ZIP = bundle(OUTPUT_ROOT)
print("Send this ZIP:", REPORT_ZIP)
if IN_COLAB:
    files.download(str(REPORT_ZIP))
else:
    display(FileLink(str(REPORT_ZIP)))
'''
    sections = [
        ("markdown", "# RoSteALS — author-path import fix\n\n"
         "**Colab A100 + high RAM, matching the previous run.** This notebook fixes the ImageNet-C package root. "
         "It runs RoSteALS only and writes a new resumable folder. Old results, weights and scientific settings are not overwritten. "
         "The ZIP contains readable Python. Local checks do not establish full GPU success; the runtime preflight does."),
        ("markdown", "## 1. Settings\n\nKeep the default paths. The previous recovery folder must still be on Drive. MAX_HOURS is a budget, not an ETA."),
        ("code", settings),
        ("markdown", "## 2. Setup\n\nAuthorize Drive; upload the matching **RoSteALS_AUTHOR_FIX_source.zip** when prompted. "
         "Use the same A100 model as before. CPU-only execution is not silently substituted."),
        ("code", setup),
        ("markdown", "## 3. Run / resume\n\nReuse only inputs and RoSteALS caches → validate real imports and all 70 synthetic attack conditions → "
         "one-image author preflight (10 cases, PNG roundtrip, unmodified CLI) → 1,000 native cases → 35,000 attacks. "
         "Before leaving, wait for **PREFLIGHT PASSED: rosteals**. A failure stops the queue with a saved log. "
         "Each finished case/grid is checkpointed. If Colab disconnects, select the same GPU and Run All again. "
         "Do not run two notebook copies against this folder."),
        ("code", queue),
        ("markdown", "## 4. Report\n\nSend **RoSteALS_AUTHOR_FIX_report.zip**, including on failure. "
         "Raw BER, exact payload recovery and legacy BCH text recovery are separate. No promise of zero BER or exact paper reproduction."),
        ("code", report)]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"rosteals-fix-{index}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            compile(source, f"cell-{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}}
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    (destination / (NAME + "_source.zip")).write_bytes(payload)
    (destination / (NAME + ".ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (destination / "README.md").write_bytes((ROOT / "README.md").read_bytes())
    (destination / "build.json").write_text(json.dumps({"sha256": checksum, "parent_sha256": PARENT_SHA,
        "code_cells": 4, "python_files": sum(n.endswith(".py") for n in files), "bytes": len(payload)}, indent=2))
    print("Built", NAME, checksum)


if __name__ == "__main__":
    build()
