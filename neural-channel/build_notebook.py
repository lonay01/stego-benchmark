"""Build a self-contained readable source distribution, without embedded code blobs."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
NAME = "COCO_Neural_Channel_RESUMABLE"
DIGITAL_SHA = "fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff"

SETTINGS = '''from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile

N_IMAGES = 5                  # After the canary: change ONLY this to 100.
MODELS = "all"                # All 29 reviewed production entries.
GPU_BATCH_SIZE = 4            # Freeze this after the first canary; A100 high RAM recommended.
MAX_HOURS = 16                # Session budget; not a guarantee of Colab uptime.
SSD_RESERVE_GIB = 60
CAPTIONS_FILE = None          # Optional official captions_val2017.json; otherwise downloaded once.
SOURCE_ARCHIVE = Path("COCO_Neural_Channel_RESUMABLE_source.zip")

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name != "posix":
    raise RuntimeError("Use Colab or Linux/NVIDIA; the complete release cannot run natively on Windows.")
if IN_COLAB:
    drive.mount("/content/drive")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
BANK_ROOT = BASE / "watermarking_production_bank_v1"   # READ ONLY
OUTPUT_ROOT = BASE / "watermarking_neural_channel_v1" # Separate from digital!
SCRATCH_ROOT = Path("/content/wmbench_neural_scratch") if IN_COLAB else Path("/workspace/wmbench_neural_scratch")
SETTINGS = dict(n_images=N_IMAGES, models=MODELS, gpu_batch_size=GPU_BATCH_SIZE, max_hours=MAX_HOURS,
    ssd_reserve_gib=SSD_RESERVE_GIB, device="cuda", bank=str(BANK_ROOT), output=str(OUTPUT_ROOT),
    scratch=str(SCRATCH_ROOT), captions_file=CAPTIONS_FILE, pairwise=True, download_author_assets=True,
    manual_assets_root=str(BANK_ROOT / "_manual_assets"))
print("Input bank (unchanged):", BANK_ROOT)
print("Durable results:", OUTPUT_ROOT)
print("Temporary attack ZIPs on Drive: one method/pipeline stage, deleted after decoding.")
print("Do not use the runtime that is currently executing the digital experiment.")
'''

BOOTSTRAP = '''EXPECTED_SHA = "__ZIP_SHA__"
SAVED_ZIP = OUTPUT_ROOT / "_release" / SOURCE_ARCHIVE.name
if not SOURCE_ARCHIVE.is_file() and SAVED_ZIP.is_file():
    SOURCE_ARCHIVE = SAVED_ZIP
if not SOURCE_ARCHIVE.is_file():
    if not IN_COLAB:
        raise FileNotFoundError("Place the source ZIP beside this notebook or set SOURCE_ARCHIVE.")
    from google.colab import files
    uploaded = files.upload()
    matches = [Path(n) for n, data in uploaded.items() if hashlib.sha256(data).hexdigest() == EXPECTED_SHA]
    if len(matches) != 1:
        raise ValueError("Upload the matching source ZIP supplied with this notebook.")
    SOURCE_ARCHIVE = matches[0]
data = SOURCE_ARCHIVE.read_bytes()
if hashlib.sha256(data).hexdigest() != EXPECTED_SHA:
    raise ValueError("Wrong source ZIP: checksum mismatch")
SAVED_ZIP.parent.mkdir(parents=True, exist_ok=True)
if not SAVED_ZIP.exists():
    SAVED_ZIP.write_bytes(data)
if hashlib.sha256(SAVED_ZIP.read_bytes()).hexdigest() != EXPECTED_SHA:
    raise ValueError("Saved source ZIP readback failed")
PROJECT = SCRATCH_ROOT / "release"
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
    for member in archive.infolist():
        target = (PROJECT / member.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()) or member.is_dir():
            raise ValueError("Unsafe source ZIP entry")
        content = archive.read(member)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Source changed: {target}; use a different empty scratch, keep results.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
import uv
UV = str(uv.find_uv_bin())
CONTROLLER = SCRATCH_ROOT / "controller/bin/python"
if not CONTROLLER.exists():
    subprocess.run([UV, "venv", "--python", "3.10.12", str(CONTROLLER.parent.parent)], check=True)
lock = OUTPUT_ROOT / "_release/controller-resolved.lock"
if lock.is_file():
    subprocess.run([UV, "pip", "sync", "--python", str(CONTROLLER), str(lock)], check=True)
else:
    subprocess.run([UV, "pip", "install", "--python", str(CONTROLLER), "-r", str(PROJECT / "requirements.txt")], check=True)
    resolved = subprocess.check_output([UV, "pip", "freeze", "--python", str(CONTROLLER)], text=True)
    lock.write_text(resolved, encoding="utf-8")
    if lock.read_text(encoding="utf-8") != resolved:
        raise ValueError("Resolved environment lock readback failed")
ENV = os.environ.copy()
ENV.update(PYTHONPATH=os.pathsep.join([str(PROJECT / "src"), str(PROJECT / "production/src")]),
    PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
    PATH=str(Path(UV).parent)+os.pathsep+ENV.get("PATH", ""),
    UV_PYTHON_INSTALL_DIR=str(SCRATCH_ROOT / "python"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
CONFIG = SCRATCH_ROOT / "neural-settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
print("Free SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
'''

RUN = '''command = [str(CONTROLLER), "-m", "wmbench_neural.runner", "run", "--settings", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", start_new_session=True)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Run stopped: inspect persistent logs. Completed rows and pending ZIPs are safe.")
except BaseException:
    if process.poll() is None:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
    raise
'''


def build():
    data = (REPO / "digital-channel/dist/COCO_Digital_Channel_RESUMABLE_source.zip").read_bytes()
    if hashlib.sha256(data).hexdigest() != DIGITAL_SHA:
        raise ValueError("Reviewed digital/production capsule changed")
    with zipfile.ZipFile(io.BytesIO(data)) as old:
        contents = {n: old.read(n) for n in old.namelist() if n.startswith(("production/", "src/"))}
    for name in ("stegastamp_cuda_hotfix.py", "stegastamp_paired_hotfix.py"):
        contents["src/"+name] = (REPO / "digital-channel" / name).read_bytes()
    for path in sorted((ROOT / "src").rglob("*.py")):
        contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted((ROOT / "tests").glob("*.py")):
        contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("README.md", "VALIDATION.md"):
        contents[name] = (ROOT/name).read_bytes()
    for name in ("neural-channel-protocol.md", "digital-channel-protocol.md", "evaluation-baselines.md", "model-catalogue.md"):
        contents["reference/"+name] = (REPO / "docs" / name).read_bytes()
    contents["requirements.txt"] = (ROOT / "requirements.txt").read_bytes()
    # Preserve the pre-existing tested diffusion lock; decoder locks live in production/.
    contents["locks/channel.lock"] = (ROOT / "locks/channel.lock").read_bytes()
    release = dict(version="1.0.0", base_digital_zip_sha256=DIGITAL_SHA,
        protocol_commit="bce02b45d9a1146164248dd5c736cd09dc343672",
        remote_verified=False, files={n: hashlib.sha256(v).hexdigest() for n, v in contents.items() if not n.startswith("production/")})
    contents["neural-release.json"] = (json.dumps(release, indent=2)+"\n").encode()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / (NAME+"_source.zip")
    for name, content in contents.items():
        if name.endswith(".py"):
            ast.parse(content.decode("utf-8"), filename=name)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(contents.items()):
            archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), content,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("production_builder", REPO / "embedding-bank/production/build_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    sections = [
        ("markdown", "# COCO neural channel — resumable, batched\n\n29 reviewed decoders. Same production bank as digital; five full messages and Native / Matched 39, 42, 45. **No new embedding.**\n\nUpload the matching source ZIP; code is readable Python. Start with **5 images, A100 high RAM**. Do not interrupt the digital experiment or start both in one runtime.\n"),
        ("markdown", "## 0. Configuration\n\nChange only `N_IMAGES` from 5 to 100 after reviewing the canary. Keep the same output folder and batch size.\n"),
        ("code", SETTINGS),
        ("markdown", "## 1. Restore the release and isolated controller\n\nNo model weights are stored in this small ZIP. Checkpoints download once per Colab session and are shared between methods. Compatible neural pipelines use the controller environment; each watermark decoder uses its own frozen environment.\n"),
        ("code", BOOTSTRAP.replace("__ZIP_SHA__", checksum)),
        ("markdown", "## 2. System libraries\n\nRestore the already reviewed MagickWand/GL dependencies for author implementations.\n"),
        ("code", builder.AUTHOR_SYSTEM_DEPENDENCIES),
        ("markdown", "## 3. Validate bank and freeze captions/seeds\n\nNo images are re-embedded. A missing bank/caption or changed release fails explicitly.\n"),
        ("code", 'subprocess.run([str(CONTROLLER), "-m", "wmbench_neural.runner", "validate", "--settings", str(CONFIG)], env=ENV, check=True)\n'),
        ("markdown", "## 4. Run\n\nClean preflight → GPU attack batches → saved ZIPs → unchanged decoder → durable scores → delete ZIPs. Repeat per pipeline and method. CPU input preparation/PNG encoding/Drive writes overlap GPU batches.\n\nTemporary Drive space is needed for one method/pipeline stage (potentially tens of GiB). Logs and completed results survive disconnects. A failed call stops safely; restart the same cells.\n"),
        ("code", RUN),
        ("markdown", "## 5. Report\n\nRebuilding reports performs no inference. Missing or safety-blocked generations are not decoding errors. The additional resize-only diagnostic is excluded from the 21-condition experiment.\n"),
        ("code", '''subprocess.run([str(CONTROLLER), "-m", "wmbench_neural.runner", "report", "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
print("Aggregate report:", OUTPUT_ROOT / "_reports")
print("Per-method tables:", OUTPUT_ROOT / "<method>/tables")
print("Decoded bits and task provenance:", OUTPUT_ROOT / "<method>/records")
'''),
        ("markdown", "## Interpretation and limits\n\nThe canary has up to **60,900** evaluations; N=100 has up to **1,218,000**, fewer when a matched carrier is unavailable. A fixed batch improves throughput, not the scientific task count. This canary itself may span multiple sessions.\n\nStatic and CPU smoke checks are recorded in VALIDATION.md. Real CUDA pipelines still need this Colab acceptance run; no full-GPU success or runtime estimate is invented. Protocol source: bundled committed revision; live GitHub access was unavailable during preparation.\n"),
    ]
    cells = []
    for i, (kind, content) in enumerate(sections):
        cell = dict(cell_type=kind, id=f"neural-{i:02d}", metadata={}, source=content.splitlines(True))
        if kind == "code":
            ast.parse(content)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                      language_info=dict(name="python"), colab=dict(name=NAME+".ipynb")))
    (dist/(NAME+".ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    (dist/(NAME+"_SHA256.txt")).write_text(checksum+"  "+target.name+"\n", encoding="ascii")
    print("Built:", dist/(NAME+".ipynb"), "source SHA256:", checksum)


if __name__ == "__main__":
    build()
