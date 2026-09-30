"""Bundle readable sources with a checksum-locked parent implementation."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "RoSteALS_SepWater_RECOVERY"
PARENT_SHA = "9f186b48fe81c1443b6db48d91a7c0b828b2ebe2b9246b732b59bb08ea0e8f61"


def inherited_files():
    parent = ROOT.parent / "author-night/dist/Three_Method_AUTHOR_NIGHT_source.zip"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("The parent release changed; do not silently vendor different code")
    with zipfile.ZipFile(parent) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()
                    if name.startswith(("authornight/", "reference/"))
                    or name in ("locks/rosteals.txt", "locks/sepwater.txt")}
    # Only the batch orchestration changes. Author repository files are untouched.
    source = contents["authornight/runner.py"].decode()
    before = '"-m", "authornight.worker", str(job_path)'
    assert source.count(before) == 1
    contents["authornight/runner.py"] = source.replace(before, '"-m", job.get("worker_module", "authornight.worker"), str(job_path)').encode()
    source = contents["authornight/latent.py"].decode()
    before = '    originals = read(root / "_inputs/originals.json")\n'
    assert source.count(before) == 1
    contents["authornight/latent.py"] = source.replace(before, before + '    originals = originals[:job.get("image_limit", len(originals))]\n').encode()
    return contents


def build():
    contents = inherited_files()
    for directory, pattern in (("authorrecovery", "*.py"), ("tests", "*.py"),
                               ("locks", "*"), ("validation", "*.json")):
        for path in sorted((ROOT / directory).glob(pattern)):
            if path.is_file() and path.suffix != ".log" and path.name != "packaged_smoke.json":
                contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("README.md", "DISTSEAL_RESUME.md", "build_notebook.py", "validate_release.py", "resolve_locks.py", "smoke_release.py"):
        contents[name] = (ROOT / name).read_bytes()
    assert "slackclient==2.9.3" in contents["locks/rosteals.txt"].decode()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = stream.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    settings = '''from pathlib import Path
import hashlib, io, os, shutil, subprocess, sys, zipfile

MAX_HOURS = 8  # Session budget, not a promise of completion. May change on resume.
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name != "posix":
    raise RuntimeError("Use Colab A100 + high RAM or a Linux NVIDIA server.")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
PARENT_ROOT = BASE / "watermarking_author_reproduction_v1"  # READ ONLY
OUTPUT_ROOT = BASE / "watermarking_rosteals_sepwater_recovery_v1"
SCRATCH_ROOT = Path("/content/author_recovery_scratch_v1") if IN_COLAB else Path("/workspace/author-recovery-scratch")
SOURCE_ARCHIVE = Path("RoSteALS_SepWater_RECOVERY_source.zip")
SETTINGS = dict(parent=str(PARENT_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT), max_hours=MAX_HOURS)
print("Fixed cohort inherited from the reviewed run: 100 COCO images, 5 messages, full 100-bit payloads.")
print("RoSteALS 256px; SepWater 224px. No DistSeal jobs. No old results will be deleted.")
print("Results:", OUTPUT_ROOT)
'''
    setup = '''if IN_COLAB:
    drive.mount("/content/drive")
from IPython.display import Markdown, display
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(SCRATCH_ROOT).free < 35 * 1024**3:
    raise RuntimeError("Need at least 35 GiB free temporary disk. Do not delete Drive results.")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True))
EXPECTED_SHA = "__SHA__"
cache = OUTPUT_ROOT / "_release" / SOURCE_ARCHIVE.name
if cache.is_file():
    payload = cache.read_bytes()
elif SOURCE_ARCHIVE.is_file():
    payload = SOURCE_ARCHIVE.read_bytes()
elif IN_COLAB:
    print("Upload RoSteALS_SepWater_RECOVERY_source.zip (the matching new source ZIP).")
    uploaded = files.upload()
    candidates = [value for value in uploaded.values() if hashlib.sha256(value).hexdigest() == EXPECTED_SHA]
    if len(candidates) != 1:
        raise ValueError("Upload the matching recovery source ZIP")
    payload = candidates[0]
else:
    raise FileNotFoundError(SOURCE_ARCHIVE)
if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA:
    raise ValueError("Release checksum mismatch; do not mix notebook versions")
# Check path separation before any persistent writes.
paths = [p.resolve() for p in (PARENT_ROOT, OUTPUT_ROOT, SCRATCH_ROOT)]
for i, left in enumerate(paths):
    for right in paths[i+1:]:
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Parent, recovery and scratch directories must be separate")
if not (PARENT_ROOT / "plan.json").is_file():
    raise FileNotFoundError(f"Original author-night run not found: {PARENT_ROOT}")
PROJECT = SCRATCH_ROOT / ("release-" + EXPECTED_SHA[:16])
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
from authorrecovery.migration import make_plan
from authornight.storage import atomic
PLAN = make_plan(SETTINGS, EXPECTED_SHA)
atomic(cache, payload)
atomic(OUTPUT_ROOT / "DISTSEAL_RESUME.md", (PROJECT / "DISTSEAL_RESUME.md").read_bytes())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
if IN_COLAB:
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "libmagickwand-dev", "build-essential", "libgl1"], check=True)
print("Ready. Run the next cell; imports are resumable and logged every 25 records.")
print("DistSeal handoff saved:", OUTPUT_ROOT / "DISTSEAL_RESUME.md")
'''.replace("__SHA__", checksum)
    queue = '''from authorrecovery.runner import run
from authorrecovery.report import bundle
try:
    SUMMARY = run(SETTINGS, EXPECTED_SHA)
finally:
    REPORT_ZIP = bundle(OUTPUT_ROOT)
    print("Partial/final report saved:", REPORT_ZIP)
'''
    report = '''from authorrecovery.report import report, bundle
from IPython.display import Markdown, display, FileLink
SUMMARY = report(OUTPUT_ROOT)
display(Markdown((OUTPUT_ROOT / "_reports/report.md").read_text()))
REPORT_ZIP = bundle(OUTPUT_ROOT)
print("Send this report ZIP:", REPORT_ZIP)
if IN_COLAB:
    files.download(str(REPORT_ZIP))
else:
    display(FileLink(str(REPORT_ZIP)))
'''
    sections = [
        ("markdown", "# RoSteALS / SepWater — targeted recovery\n\n"
         "**A100 (same GPU as the old run) + high RAM.** Resume the reviewed 100-image author-path experiment. "
         "Old files are read-only. DistSeal is not run. The separate ZIP contains readable Python, not Base64.\n\n"
         "Local checks include real BCH replay on the previous SepWater records. Full CUDA execution remains to be verified in Colab."),
        ("markdown", "## 1. Settings\n\nLeave paths unchanged if the original run is in MyDrive. Only MAX_HOURS is a runtime budget."),
        ("code", settings),
        ("markdown", "## 2. Drive and source setup\n\nAuthorize Drive and upload the new matching source ZIP when asked. The old 3-method ZIP is not the new ZIP."),
        ("code", setup),
        ("markdown", "## 3. Run / resume\n\nImport verified results → CPU BCH replay → two isolated GPU preflights → missing RoSteALS COCO cases → remaining SepWater/RoSteALS attacks. "
         "Only one GPU worker runs at once. Completed verified cases and attack conditions are reused. "
         "Before leaving, check for PREFLIGHT PASSED in persistent logs. A time limit means partial coverage, not model failure. "
         "If Colab stops, reconnect with the same A100 and run cells 1–3 again. Do not run two copies against one folder."),
        ("code", queue),
        ("markdown", "## 4. Report\n\nFull-payload BER, corrected BCH text recovery, technical checks and incomplete coverage are separate. "
         "DistSeal continuation instructions are saved in OUTPUT_ROOT/DISTSEAL_RESUME.md."),
        ("code", report),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = dict(cell_type=kind, id=f"recovery-{index:02d}", metadata={}, source=source.splitlines(True))
        if kind == "code":
            compile(source, f"cell-{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        "kernelspec": dict(display_name="Python 3", language="python", name="python3"),
        "language_info": dict(name="python"), "accelerator": "GPU"})
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    (output / (NAME + ".ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / (NAME + "_source.zip")).write_bytes(payload)
    (output / "DISTSEAL_RESUME.md").write_bytes((ROOT / "DISTSEAL_RESUME.md").read_bytes())
    (output / "build.json").write_text(json.dumps({"sha256": checksum, "parent_sha256": PARENT_SHA,
        "code_cells": 4, "python_files": sum(n.endswith('.py') for n in contents), "bytes": len(payload)}, indent=2))
    print("Built", NAME, checksum)


if __name__ == "__main__":
    build()
