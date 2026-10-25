"""Repackage v3 with report I/O only; never rewrite the delivered v3 archive."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "dist"
BASE = "COCO_Digital_Channel_RECOVERED_FAST_RESUME"
BASE_SHA = "c94a9cf9e21b08754a0800af538325ea270aa42b1e4b3e02020426c49dbbfddf"
NAME = "COCO_Digital_Channel_REPORT_IO_FIX"


def build():
    original = DIST / (BASE + "_source.zip")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == BASE_SHA
    with zipfile.ZipFile(original) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["performance-release.json"])
    updates = {
        "src/wmbench_fast/runner.py": ROOT / "src/wmbench_fast/runner.py",
        "src/wmbench_fast/report_io.py": ROOT / "src/wmbench_fast/report_io.py",
        "performance-tests/test_report_io.py": ROOT / "tests/test_report_io.py",
        "PERFORMANCE.md": ROOT / "README.md",
    }
    for name, path in updates.items():
        files[name] = path.read_bytes()
        manifest["files"][name] = hashlib.sha256(files[name]).hexdigest()
    manifest["execution_repair"] = "verified-ssd-report-publication-v4"
    files["performance-release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    target = DIST / (NAME + "_source.zip")
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in sorted(files.items()):
            if name.endswith(".py"):
                ast.parse(data, filename=name)
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    nb = json.loads((DIST / (BASE + ".ipynb")).read_text("utf-8"))
    nb["metadata"]["colab"]["name"] = NAME + ".ipynb"
    for cell in nb["cells"]:
        source = "".join(cell["source"]).replace(BASE, NAME)
        source = source.replace("performance_release_recovered_v3", "performance_release_report_io_v4")
        source = source.replace(BASE_SHA, checksum)
        if cell["id"] == "parallel-00":
            source += """
Report-only repair: build tables on local SSD, validate all six files, preserve
the previous tables, and publish SHA256-checked copies to Drive. An interrupted
publication reuses the staged report without recomputing metrics. Model code,
attacks and all saved raw measurements are unchanged. The final wait cell
waits for this actual computation; it cannot guarantee Colab VM lifetime.
"""
        if cell["cell_type"] == "code":
            ast.parse(source)
            cell["outputs"], cell["execution_count"] = [], None
        cell["source"] = source.splitlines(True)
    wait = '''# Wait for the controller that Start / resume already launched.
# This does not start, restart or stop benchmark work.
monitor = r"""
import sys, time
from pathlib import Path
from wmbench.io import read_json
from wmbench_fast.launch import alive
record = read_json(Path(sys.argv[1]) / "_reports/fast_launcher.json")
config = Path(record["config"])
exit_file = config.with_name(config.stem + "_exit.json")
print("Waiting for benchmark controller", record["pid"], flush=True)
while alive(record):
    time.sleep(30)
if not exit_file.is_file():
    raise RuntimeError("Controller stopped without exit status. Inspect its log; preserve results.")
status = read_json(exit_file)
print("Controller finished:", status, flush=True)
raise SystemExit(status["exit_code"])
"""
with subprocess.Popen([str(CONTROLLER), "-u", "-c", monitor, str(OUTPUT_ROOT)],
                      env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True) as watching:
    for line in watching.stdout:
        print(line, end="", flush=True)
    if watching.wait():
        raise RuntimeError("Controller monitor reported an incomplete run. Inspect the log.")
'''
    ast.parse(wait)
    nb["cells"].append(dict(cell_type="code", id="wait-current-controller", metadata={},
                            source=wait.splitlines(True), outputs=[], execution_count=None))
    (DIST / (NAME + ".ipynb")).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", "utf-8")
    (DIST / (NAME + "_SHA256.txt")).write_text(checksum + "  " + target.name + "\n", "utf-8")
    print(target, checksum)


if __name__ == "__main__":
    build()
