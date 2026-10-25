"""Build the narrow lease/storage repair on the exact report-I/O release."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "dist"
BASE = "COCO_Digital_Channel_REPORT_IO_FIX"
BASE_SHA = "2198afa282de2c4c352700c44593d245adfa4e4e2d79194e3f0c6c205505a7f5"
NAME = "COCO_Digital_Channel_LEASE_IO_FIX"


def build():
    original = DIST / (BASE + "_source.zip")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == BASE_SHA
    with zipfile.ZipFile(original) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["performance-release.json"])
    updates = {
        "src/wmbench_fast/runner.py": ROOT / "src/wmbench_fast/runner.py",
        "src/wmbench_fast/launch.py": ROOT / "src/wmbench_fast/launch.py",
        "src/wmbench_fast/lease_io.py": ROOT / "src/wmbench_fast/lease_io.py",
        "performance-tests/test_lease_io.py": ROOT / "tests/test_lease_io.py",
        "LEASE_IO_FIX.md": ROOT / "LEASE_IO_FIX.md",
    }
    for name, path in updates.items():
        files[name] = path.read_bytes()
        manifest["files"][name] = hashlib.sha256(files[name]).hexdigest()
    manifest["execution_repair"] = "bounded-lease-read-recovery-v5"
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
        source = "".join(cell["source"]).replace(BASE, NAME).replace(BASE_SHA, checksum)
        source = source.replace("performance_release_report_io_v4", "performance_release_lease_io_v5")
        if cell["id"] == "parallel-00":
            source += "\nLease/storage repair: bounded control-file retries with ownership checks intact. " \
                      "Existing bank and results are retained. See LEASE_IO_FIX.md in the source ZIP.\n"
        if cell["id"] == "parallel-01":
            source = source.replace("MAX_HOURS = 16", "MAX_HOURS = 24")
            source = source.replace("DECODE_WORKERS = 2", "DECODE_WORKERS = 4")
            source = source.replace("ATTACK_WORKERS = 2", "ATTACK_WORKERS = 4")
            source = source.replace('    drive.mount("/content/drive")',
                '    drive.mount("/content/drive")\n'
                '    if not os.path.ismount("/content/drive"):\n'
                '        raise RuntimeError("Drive is not mounted; authorize it before continuing.")')
        if cell["cell_type"] == "code":
            ast.parse(source)
            cell["outputs"], cell["execution_count"] = [], None
        cell["source"] = source.splitlines(True)
    (DIST / (NAME + ".ipynb")).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", "utf-8")
    (DIST / (NAME + "_SHA256.txt")).write_text(checksum + "  " + target.name + "\n", "utf-8")
    print(target, checksum)


if __name__ == "__main__":
    build()
