"""Build a readable diagnostic capsule without modifying the frozen neural ZIP."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "InvisMark_ChunkySeal_NEURAL_DIAGNOSTICS"
BASE_SHA = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"

SETTINGS = '''from pathlib import Path
import hashlib, io, json, os, shutil, subprocess, sys, zipfile

N_IMAGES = 5  # All five messages for each of these historical images; Native only.
MAX_HOURS = 8
SOURCE_ARCHIVE = Path("InvisMark_ChunkySeal_NEURAL_DIAGNOSTICS_source.zip")

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name != "posix":
    raise RuntimeError("Use Colab or Linux/NVIDIA. Local Windows is for source inspection, not this GPU run.")
if IN_COLAB:
    drive.mount("/content/drive")
BASE = Path("/content/drive/MyDrive") if IN_COLAB else Path("/workspace/data")
BANK_ROOT = BASE / "watermarking_production_bank_v1"  # READ ONLY
NEURAL_ROOT = BASE / "watermarking_neural_channel_v1" # READ ONLY
OUTPUT_ROOT = BASE / "watermarking_neural_two_method_check_v1"
SCRATCH_ROOT = Path("/content/wmbench_neural_two_method_check") if IN_COLAB else Path("/workspace/neural_two_method_scratch")
roots = [p.resolve() for p in (BANK_ROOT, NEURAL_ROOT, OUTPUT_ROOT, SCRATCH_ROOT)]
for i, p in enumerate(roots):
    for q in roots[i+1:]:
        if p.is_relative_to(q) or q.is_relative_to(p):
            raise ValueError("All four roots must be separate, non-overlapping directories")
if not 1 <= N_IMAGES <= 5:
    raise ValueError("Diagnostic only: choose 1..5 images, not 100")
for p in (BANK_ROOT / "trial.json", NEURAL_ROOT / "neural-run.json"):
    if not p.is_file():
        raise FileNotFoundError(f"Required historical input missing: {p}")
print("Read-only bank:", BANK_ROOT)
print("Read-only old neural run:", NEURAL_ROOT)
print("New diagnostics:", OUTPUT_ROOT)
print("Do not run this in the runtime that is executing the digital experiment.")
'''

BOOTSTRAP = '''EXPECTED_SHA = "__SOURCE_SHA__"
BASE_SHA = "__BASE_SHA__"
SAVED = OUTPUT_ROOT / "_release" / SOURCE_ARCHIVE.name
if not SOURCE_ARCHIVE.is_file() and SAVED.is_file():
    SOURCE_ARCHIVE = SAVED
if not SOURCE_ARCHIVE.is_file():
    if not IN_COLAB:
        raise FileNotFoundError("Place the matching source ZIP next to this notebook or set SOURCE_ARCHIVE")
    from google.colab import files
    uploaded = files.upload()
    matches = [Path(name) for name, data in uploaded.items() if hashlib.sha256(data).hexdigest() == EXPECTED_SHA]
    if len(matches) != 1:
        raise ValueError("Upload the exact diagnostic _source.zip supplied with this notebook")
    SOURCE_ARCHIVE = matches[0]
data = SOURCE_ARCHIVE.read_bytes()
if hashlib.sha256(data).hexdigest() != EXPECTED_SHA:
    raise ValueError("Wrong or changed diagnostic source ZIP")
SAVED.parent.mkdir(parents=True, exist_ok=True)
if SAVED.exists() and SAVED.read_bytes() != data:
    raise ValueError("A different diagnostic release is already saved. Keep results; use a new output root")
SAVED.write_bytes(data)
if SAVED.read_bytes() != data:
    raise RuntimeError("Drive source ZIP readback failed")

def extract_checked(blob, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            path = (destination / entry.filename).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive entry")
            value = archive.read(entry)
            if path.exists() and path.read_bytes() != value:
                raise ValueError(f"Local source changed: {path}; use a new scratch folder, keep Drive results")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)

PACKAGE = SCRATCH_ROOT / "diagnostics"
extract_checked(data, PACKAGE)
capsule = (PACKAGE / "base.zip").read_bytes()
if hashlib.sha256(capsule).hexdigest() != BASE_SHA:
    raise ValueError("Original frozen neural capsule checksum mismatch")
PROJECT = SCRATCH_ROOT / "base"
extract_checked(capsule, PROJECT)
for rel, expected in json.loads((PACKAGE / "manifest.json").read_text())["files"].items():
    if hashlib.sha256((PACKAGE / rel).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Diagnostic source changed: {rel}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.15"], check=True)
import uv
UV = str(uv.find_uv_bin())
ENV = os.environ.copy()
ENV.update(PATH=str(Path(UV).parent)+os.pathsep+ENV.get("PATH", ""),
    UV_PYTHON_INSTALL_DIR=str(SCRATCH_ROOT / "python"),
    UV_CACHE_DIR=str(SCRATCH_ROOT / "uv-cache"), PYTHONUNBUFFERED="1",
    OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
    CUBLAS_WORKSPACE_CONFIG=":4096:8")
CONTROLLER = SCRATCH_ROOT / "controller/bin/python"
if not CONTROLLER.is_file():
    subprocess.run([UV, "venv", "--python", "3.10.12", str(CONTROLLER.parent.parent)], env=ENV, check=True)
# Reproduce the reviewed neural controller, not today's dependency resolution.
old_lock = NEURAL_ROOT / "_release/controller-resolved.lock"
if not old_lock.is_file():
    raise FileNotFoundError(f"Historical controller lock required: {old_lock}")
lock = OUTPUT_ROOT / "_release/controller-resolved.lock"
lock_data = old_lock.read_bytes()
if lock.exists() and lock.read_bytes() != lock_data:
    raise ValueError("Historical controller lock changed")
lock.write_bytes(lock_data)
print("Restoring isolated controller from historical lock...", flush=True)
subprocess.run([UV, "pip", "sync", "--python", str(CONTROLLER), str(lock)], env=ENV, check=True)
ENV["PYTHONPATH"] = os.pathsep.join([str(PACKAGE / "src"), str(PROJECT / "src"), str(PROJECT / "production/src")])
old_run = json.loads((NEURAL_ROOT / "neural-run.json").read_text())
SETTINGS = dict(n_images=N_IMAGES, models=["invismark", "chunkyseal"],
    gpu_batch_size=old_run["gpu_batch_size"], max_hours=MAX_HOURS,
    bank=str(BANK_ROOT), neural=str(NEURAL_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
    project=str(PROJECT), device="cuda", download_author_assets=True,
    manual_assets_root=str(BANK_ROOT / "_manual_assets"))
CONFIG = SCRATCH_ROOT / "diagnostic-settings.json"
CONFIG.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
print("Free local SSD: %.1f GiB" % (shutil.disk_usage(SCRATCH_ROOT).free / 1024**3))
'''

RUN = '''command = [str(CONTROLLER), "-m", "neural_io_check.runner", "run", "--settings", str(CONFIG)]
process = subprocess.Popen(command, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", start_new_session=True)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Diagnostic stopped. Read the saved log/REPORT.md; existing results are safe.")
except BaseException:
    if process.poll() is None:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
    raise
'''

REPORT = '''subprocess.run([str(CONTROLLER), "-m", "neural_io_check.runner", "report", "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import display, Markdown, Image as DisplayImage
display(Markdown((OUTPUT_ROOT / "REPORT.md").read_text(encoding="utf-8")))
for preview in sorted((OUTPUT_ROOT / "gallery").glob("*.jpg")):
    print(preview.name)
    display(DisplayImage(filename=str(preview), width=1100))
print("Report:", OUTPUT_ROOT / "REPORT.md")
print("Rows:", OUTPUT_ROOT / "per_case.csv")
print("Detailed evidence:", OUTPUT_ROOT / "<method>/checks")
# Small review bundle. Exact PNGs/float arrays remain separately on Drive.
review = OUTPUT_ROOT / "diagnostic_review.zip"
with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        rel = path.relative_to(OUTPUT_ROOT)
        if any(part in {"_runtime", "_release", "images", "float"} for part in rel.parts):
            continue
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".log", ".jpg"}:
            z.write(path, rel)
print("Share this review ZIP:", review)
'''


def build():
    capsule = ROOT.parent / "dist/COCO_Neural_Channel_RESUMABLE_source.zip"
    raw = capsule.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError("Frozen base capsule changed; do not rebuild it")
    files = {"base.zip": raw, "README.md": (ROOT / "README.md").read_bytes(),
             "VALIDATION.md": (ROOT / "VALIDATION.md").read_bytes(),
             "src/neural_sdxl_identity_fix.py": (ROOT.parent / "repairs/neural_sdxl_identity_fix.py").read_bytes()}
    for folder in ("src", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name, value in files.items():
        if name.endswith(".py"):
            ast.parse(value.decode("utf-8"), filename=name)
            compile(value, name, "exec")
    manifest = {"files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
                "version": "two-method-io-v1"}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / (NAME + "_source.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in sorted(files.items()):
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 13, 0, 0, 0)), value,
                       compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    sections = [
        ("markdown", "# InvisMark / ChunkySeal: neural IO diagnostics\n\nОтдельная проверка на тех же пяти изображениях и пяти полных сообщениях. Старые банк и нейросетевой прогон **не изменяются**. Используйте A100 того же типа, что в канарейке, и большой объём ОЗУ. Не запускайте в среде, где считается цифровой канал.\n\nНужны этот notebook и соответствующий `_source.zip`. Исходники — обычные Python-файлы, без Base64.\n"),
        ("markdown", "## 1. Paths\n\nПо умолчанию пути такие же, как в завершённом прогоне; новый результат сохраняется отдельно. Оставьте `N_IMAGES=5`.\n"),
        ("code", SETTINGS),
        ("markdown", "## 2. Restore the exact release\n\nОдин раз загрузите приложенный диагностический ZIP. Затем он будет браться с Drive. Веса и изолированные окружения восстанавливаются отдельно; содержимое прежних папок не переписывается.\n"),
        ("code", BOOTSTRAP.replace("__SOURCE_SHA__", checksum).replace("__BASE_SHA__", BASE_SHA)),
        ("markdown", "## 3. Validate historical inputs\n\nПроверка контрольных сумм, сообщений, исторических записей и порядка batch. Это не повторное встраивание.\n"),
        ("code", 'subprocess.run([str(CONTROLLER), "-m", "neural_io_check.runner", "validate", "--settings", str(CONFIG)], env=ENV, check=True)\n'),
        ("markdown", "## 4. Run the two methods\n\nДля каждого: Clean/PNG/normalization/resize/JPEG/оригинал → VAE1/2 → SDXL0.15 → IP2P lighting. Прямой декодер сравнивается с адаптером; все PNG сохраняются. Float VAE — отдельный контроль округления.\n\nЯчейка остаётся выполняющейся; heartbeat и логи идут на Drive. После обрыва повторите те же ячейки. При расхождении IO проверка остановится, не подбирая лучший результат.\n"),
        ("code", RUN),
        ("markdown", "## 5. Review\n\nЭту ячейку можно выполнить и после ошибки: она не запускает модели. Смотрите таблицу, вердикт IO и контактные листы. Для разбора пришлите `diagnostic_review.zip`.\n"),
        ("code", REPORT),
        ("markdown", "## Limits\n\nЛокально проверяются синтаксис, CPU-контракты и mock smoke-тесты. Запуск настоящих весов на CUDA проверяется этим notebook, не заявлен заранее. Совпадение двух путей IO не доказывает воспроизведение статей. BER около 50% при исправном IO остаётся измерением, а не основанием менять пороги.\n"),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = dict(cell_type=kind, id=f"io-check-{index:02d}", metadata={}, source=source.splitlines(True))
        if kind == "code":
            ast.parse(source, filename=f"cell-{index}")
            compile(source, f"cell-{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(name="python3", display_name="Python 3", language="python"),
        language_info=dict(name="python"), accelerator="GPU"))
    path = dist / (NAME + ".ipynb")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps(dict(source_sha256=checksum, base_sha256=BASE_SHA,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        code_cells=sum(c["cell_type"] == "code" for c in cells),
        python_modules=sum(k.endswith(".py") for k in files), gpu_tested=False), indent=2) + "\n")
    print(path)
    print(archive)


if __name__ == "__main__":
    build()
