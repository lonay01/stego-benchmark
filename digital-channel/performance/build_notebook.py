"""Build an immutable-base overlay plus a small, readable Colab notebook."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIGITAL = ROOT.parent
NAME = "COCO_Digital_Channel_PARALLEL_FAST_RESUME"
BASE_SHA = "fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build():
    old = DIGITAL/"dist/COCO_Digital_Channel_RESUMABLE_source.zip"
    if hashlib.sha256(old.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Frozen base changed")
    with zipfile.ZipFile(old) as zipped:
        content = {n: zipped.read(n) for n in zipped.namelist()}
    added = {}
    for path in sorted((ROOT/"src").rglob("*.py")):
        added[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("digital_resume_hotfix.py", "stegastamp_clean_hotfix.py", "stegastamp_cuda_hotfix.py", "stegastamp_paired_hotfix.py"):
        added["src/"+name] = (DIGITAL/name).read_bytes()
    for path in sorted((ROOT/"tests").glob("*.py")):
        added["performance-tests/"+path.name] = path.read_bytes()
    # Keep original README and all frozen files unchanged.
    added["PERFORMANCE.md"] = (ROOT/"README.md").read_bytes()
    if set(content) & set(added):
        raise ValueError("Overlay must not replace a frozen file")
    manifest = dict(version="1.0.0", base_zip_sha256=BASE_SHA,
                    files={n: hashlib.sha256(data).hexdigest() for n, data in added.items()})
    content.update(added)
    content["performance-release.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    dist = DIGITAL/"dist"
    target = dist/(NAME+"_source.zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name, data in sorted(content.items()):
            if name.endswith(".py"):
                ast.parse(data, filename=name)
            zipped.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                            compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    original = load(DIGITAL/"build_notebook.py", "base_notebook")
    production = load(DIGITAL.parent/"embedding-bank/production/build_notebook.py", "production_notebook")
    settings = original.SETTINGS.replace("N_IMAGES = 5", "N_IMAGES = 100").replace(
        'SOURCE_ARCHIVE = Path("COCO_Digital_Channel_RESUMABLE_source.zip")',
        f'SOURCE_ARCHIVE = Path("{NAME}_source.zip")\nDECODE_WORKERS = 2\nATTACK_WORKERS = 2\nFULL_AUDIT = False')
    settings += '\nSETTINGS.update(decode_workers=DECODE_WORKERS, attack_workers=ATTACK_WORKERS, full_audit=FULL_AUDIT)\n'
    bootstrap = original.BOOTSTRAP.replace("__ZIP_SHA__", checksum).replace(
        'PROJECT = SCRATCH_ROOT / "release"', 'PROJECT = SCRATCH_ROOT / "performance_release_v1"')
    bootstrap = bootstrap.replace("Use a different empty SCRATCH_ROOT; retain OUTPUT_ROOT.",
                                  "Do not overwrite it; keep bank/output paths and inspect the source conflict.")
    run = '''# Stop the OLD notebook runner first. Repeating this cell will not start a duplicate fast runner.
subprocess.run([str(CONTROLLER), "-m", "wmbench_fast.launch", "start", "--settings", str(CONFIG)],
               env=ENV, check=True)
'''
    status = '''# Read-only, bounded log view. Re-run whenever you want; this does not rebuild reports.
import time
from IPython.display import Markdown, display
launch = json.loads((OUTPUT_ROOT / "_reports/fast_launcher.json").read_text())
log = Path(launch["log"])
if log.exists():
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 16000))
        print(handle.read().decode("utf-8", errors="replace"))
report = OUTPUT_ROOT / "_reports/FAST_REPORT.md"
if report.exists():
    display(Markdown(report.read_text(encoding="utf-8")))
else:
    print("Controller starting; rerun this cell shortly.")
exit_file = Path(launch["config"]).with_name(Path(launch["config"]).stem + "_exit.json")
print("Controller exit:", exit_file.read_text() if exit_file.exists() else "not yet recorded")
print("Worker logs:", OUTPUT_ROOT / "<method>/logs")
'''
    stop = '''# OPTIONAL. Set True only when you explicitly want to stop the background calculation.
STOP_BACKGROUND = False
if STOP_BACKGROUND:
    subprocess.run([str(CONTROLLER), "-m", "wmbench_fast.launch", "stop", "--settings", str(CONFIG)],
                   env=ENV, check=True)
'''
    sections = [
        ("markdown", "# Digital channel — parallel, fast resume\n\nSame reviewed 29 methods and 79 conditions. Existing bank/results stay in place. This is an execution optimization, not a new experiment protocol.\n\n**Stop the old runner before starting this notebook.** Use the same GPU type and same paths. Download the matching source ZIP too. Start with two workers; StegaStamp stays serial. No model CUDA run has been executed locally.\n"),
        ("code", settings),
        ("markdown", "## Install\n\nOne small source ZIP, uploaded once and saved on Drive. No Base64. Frozen model environments remain isolated.\n"),
        ("code", bootstrap),
        ("code", production.AUTHOR_SYSTEM_DEPENDENCIES),
        ("code", 'subprocess.run([str(CONTROLLER), "-m", "wmbench_fast.runner", "validate", "--settings", str(CONFIG)], env=ENV, check=True)\n'),
        ("markdown", "## Start / resume\n\nFirst use indexes legacy records once, with visible progress. Later resumes reuse checksummed catalogs and size/mtime inventories. Do not edit the bank or result files externally in fast mode. `FULL_AUDIT=True` rehashes the underlying evidence and is intentionally slow. Pending inputs and attack ZIPs remain checksum-checked.\n\nThe controller is detached: a browser disconnect or stopping the log-view cell does not intentionally kill it. Colab VM termination still stops work; rerun with the same paths after reconnect. This cell returns immediately while computation continues.\n"),
        ("code", run), ("markdown", "## Progress\n\nRefresh this cell manually. Per-worker telemetry is under each method's `logs/`. Reports are generated after a method finishes. Historical canary tables are not completion evidence for 100 images.\n"),
        ("code", status), ("markdown", "## Optional stop\n\nCompleted checkpoints remain on Drive; an interrupted case resumes from its last checkpoint. The embedding bank is read-only.\n"),
        ("code", stop),
        ("markdown", "## Tuning and integrity\n\nDefault: two CPU package workers and up to two GPU decoders for one method. Decoder admission waits for warm-up and memory headroom. Missing GPU telemetry means sequential admission. On A100 80 GB, test `DECODE_WORKERS=4` after reviewing two-worker throughput. Memory headroom is a heuristic, not an OOM guarantee. Do not compare concurrent decode latency with old isolated latency.\n\nThe serial reference and additional workers must agree on four fixed probes; the original per-case Clean gate remains. No valid poor result is retried. Package deletion still requires original raw-result verification. Completed methods get fast receipts so their reports and environments are not rebuilt on every resume. See `PERFORMANCE.md` inside the source ZIP for the full policy and limitations.\n")]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = dict(cell_type=kind, id=f"parallel-{index:02d}", metadata={}, source=source.splitlines(True))
        if kind == "code":
            ast.parse(source)
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"name": NAME+".ipynb"}})
    (dist/(NAME+".ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    (dist/(NAME+"_SHA256.txt")).write_text(checksum+"  "+target.name+"\n", encoding="ascii")
    print(dist/(NAME+".ipynb"))
    print(checksum)


if __name__ == "__main__":
    build()
