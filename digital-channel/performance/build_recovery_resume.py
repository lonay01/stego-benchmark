"""Build the October 16 recovery/fast-resume release without changing frozen science."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIGITAL = ROOT.parent
DIST = DIGITAL / "dist"
NAME = "COCO_Digital_Channel_RECOVERY_FAST_RESUME"
BASE = "COCO_Digital_Channel_PARALLEL_FAST_RESUME"
BASE_SHA = "3115ddb4920c64f9e30c5e4ace91ffb521b5f449173bcbaeb863ec442b3fad94"


def build():
    original = DIST / (BASE + "_source.zip")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == BASE_SHA
    with zipfile.ZipFile(original) as archive:
        content = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(content["performance-release.json"])
    changed = []
    for path in sorted((ROOT / "src/wmbench_fast").glob("*.py")):
        member = "src/wmbench_fast/" + path.name
        data = path.read_bytes()
        if content.get(member) != data:
            changed.append(member)
        content[member] = data
        manifest["files"][member] = hashlib.sha256(data).hexdigest()
    for filename in ("stegastamp_byte_recovery.py", "stegastamp_recovery_driver.py"):
        member = "src/" + filename
        content[member] = (DIGITAL / filename).read_bytes()
        manifest["files"][member] = hashlib.sha256(content[member]).hexdigest()
        changed.append(member)
    for filename in ("test_resume_v2.py", "test_psutil_fix.py"):
        member = "performance-tests/" + filename
        content[member] = (ROOT / "tests" / filename).read_bytes()
        manifest["files"][member] = hashlib.sha256(content[member]).hexdigest()
    content["PERFORMANCE.md"] = (ROOT / "README.md").read_bytes()
    manifest["files"]["PERFORMANCE.md"] = hashlib.sha256(content["PERFORMANCE.md"]).hexdigest()
    manifest["execution_repair"] = "stegastamp-byte-recovery-and-resume-v2"
    content["performance-release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    archive_path = DIST / (NAME + "_source.zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member, data in sorted(content.items()):
            if member.endswith(".py"):
                ast.parse(data, filename=member)
            archive.writestr(zipfile.ZipInfo(member, date_time=(2026, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    nb = json.loads((DIST / ("COCO_Digital_Channel_PARALLEL_PSUTIL_FIX.ipynb")).read_text("utf-8"))
    nb["metadata"]["colab"]["name"] = NAME + ".ipynb"
    cells = []
    recovery = '''# One-time repair. A repeat run checks the four original hashes and skips.
# Only known zero-byte StegaStamp files can be restored. Other inputs/results stay unchanged.
subprocess.run([str(CONTROLLER), "-u", "-m", "stegastamp_recovery_driver",
    "--project", str(PROJECT / "production"), "--bank", str(BANK_ROOT),
    "--work", str(BANK_ROOT.parent / "watermarking_stegastamp_recovery_20261016"),
    "--scratch", str(SCRATCH_ROOT / "stegastamp_recovery"),
    "--results", str(OUTPUT_ROOT)], env=ENV, check=True)
'''
    for cell in nb["cells"]:
        if cell.get("id") in {"psutil-fix-note", "psutil-fix-setup"}:
            continue
        source = "".join(cell["source"])
        source = source.replace(BASE + "_source.zip", NAME + "_source.zip").replace(BASE_SHA, checksum)
        source = source.replace('SCRATCH_ROOT / "performance_release_v1"', 'SCRATCH_ROOT / "performance_release_recovery_v2"')
        if cell.get("id") == "parallel-00":
            source = "# Digital channel — StegaStamp recovery and faster resume\n\n" \
                "Same frozen 29 methods, messages and 79 conditions. This release fixes four known empty " \
                "StegaStamp PNGs only when their original hashes and Clean bits reproduce exactly. " \
                "No reference hash or measured result is changed. Stop any old active runner first.\n\n" \
                "Upload the matching new source ZIP once; it stays on Drive. Use the same bank/output " \
                "folders and GPU type. Run all cells. Existing complete methods are reused; " \
                "the first run creates compact completion receipts, later runs avoid large catalogue reads.\n"
        if cell["cell_type"] == "code":
            ast.parse(source)
            cell.update(outputs=[], execution_count=None)
        cell["source"] = source.splitlines(True)
        cells.append(cell)
        if '"wmbench_fast.runner", "validate"' in source:
            cells.append(dict(cell_type="markdown", id="byte-recovery-note", metadata={}, source=[
                "## Recover the four StegaStamp inputs\n\nOriginal checkpoint, locked dependencies and recorded "
                "payloads are used. Candidate PNGs and receipts are retained in a separate Drive recovery folder. "
                "A mismatch stops without replacing inputs. This step does not rebuild digital results.\n"]))
            cells.append(dict(cell_type="code", id="byte-recovery-run", metadata={},
                              source=recovery.splitlines(True), execution_count=None, outputs=[]))
    nb["cells"] = cells
    target = DIST / (NAME + ".ipynb")
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (DIST / (NAME + "_SHA256.txt")).write_text(checksum + "  " + archive_path.name + "\n")
    print(target)
    print(archive_path)
    print(checksum)
    return changed


if __name__ == "__main__":
    build()
