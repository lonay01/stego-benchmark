"""Package only the missing-psutil telemetry repair in the submitted notebook."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "dist"
NAME = "COCO_Digital_Channel_PARALLEL_PSUTIL_FIX"
BASE_SHA = "3115ddb4920c64f9e30c5e4ace91ffb521b5f449173bcbaeb863ec442b3fad94"


def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def repair_cell():
    archive_path = DIST / "COCO_Digital_Channel_PARALLEL_FAST_RESUME_source.zip"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == BASE_SHA
    patches = {}
    with zipfile.ZipFile(archive_path) as archive:
        for filename, function in (("worker.py", "resources"), ("scheduler.py", "invoke")):
            member = "src/wmbench_fast/" + filename
            before = archive.read(member).decode("utf-8").replace("\r\n", "\n")
            after = (ROOT / member).read_text("utf-8").replace("\r\n", "\n")
            old, new = function_source(before, function), function_source(after, function)
            assert old != new and before.count(old) == 1
            assert before.replace(old, new, 1) == after, "Unrelated source change: " + member
            patches[member] = (old, new)

    source = '''# Fix only resource telemetry when an isolated decoder has no psutil.
# No pip installs into model environments, no bank/result edits, no new ZIP.
PSUTIL_FIX_PATCHES = __PATCHES__
PSUTIL_FIX_PROJECT = SCRATCH_ROOT / "performance_release_v1_psutil_fix_v1"
if hashlib.sha256(SOURCE_ARCHIVE.read_bytes()).hexdigest() != EXPECTED_SHA:
    raise ValueError("Original source ZIP checksum mismatch")
with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
    fixed_files = {name: archive.read(name) for name in archive.namelist()}
manifest = json.loads(fixed_files["performance-release.json"])
for name, expected in manifest["files"].items():
    if hashlib.sha256(fixed_files[name]).hexdigest() != expected:
        raise ValueError(f"Original performance source checksum mismatch: {name}")
for name, (old, new) in PSUTIL_FIX_PATCHES.items():
    source = fixed_files[name].decode("utf-8").replace("\\r\\n", "\\n")
    if source.count(old) != 1:
        raise ValueError(f"Unexpected original telemetry source: {name}")
    fixed_files[name] = source.replace(old, new, 1).encode("utf-8")
    manifest["files"][name] = hashlib.sha256(fixed_files[name]).hexdigest()
manifest["telemetry_repair"] = "psutil-controller-fallback-v1"
fixed_files["performance-release.json"] = (json.dumps(manifest, indent=2) + "\\n").encode("utf-8")
# Use a separate source directory: the old installed release stays untouched.
# Preflight every path and existing file before writing any missing source.
for name, data in fixed_files.items():
    target = (PSUTIL_FIX_PROJECT / name).resolve()
    if not target.is_relative_to(PSUTIL_FIX_PROJECT.resolve()):
        raise ValueError(f"Unsafe source path: {name}")
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f"Different repair already exists: {target}; nothing overwritten")
for name, data in fixed_files.items():
    target = PSUTIL_FIX_PROJECT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    if target.read_bytes() != data:
        raise ValueError(f"Repair source readback failed: {target}")
source_paths = {
    str(PROJECT / "src"): str(PSUTIL_FIX_PROJECT / "src"),
    str(PROJECT / "production/src"): str(PSUTIL_FIX_PROJECT / "production/src"),
}
ENV["PYTHONPATH"] = os.pathsep.join(source_paths.get(p, p) for p in ENV["PYTHONPATH"].split(os.pathsep))
PROJECT = PSUTIL_FIX_PROJECT
print("psutil telemetry repair ready; model dependencies, settings and saved results unchanged.")
'''.replace("__PATCHES__", pprint.pformat(patches, width=100, sort_dicts=True))
    ast.parse(source)
    return source


def build(source_path):
    notebook = json.loads(source_path.read_text("utf-8"))
    source = repair_cell()
    note = """## Missing psutil: telemetry-only repair

This fixes the InvisMark startup error in the parallel worker's resource probe.
If a decoder does not have psutil, the existing controller Python measures that
worker's RAM. Model environments, weights, attacks, scoring, parallel worker
limits and resume checks are unchanged. Telemetry failure still stops the run.

Use the same source ZIP and bank/results folders. No new ZIP is needed if the
original archive is already saved on Drive. Run all cells in the same GPU type.
The repair uses a separate source directory and preserves the old release.
Completed results are reused. Stop any older active runner first.
"""
    cells, inserted = [], 0
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and '"wmbench_fast.runner", "validate"' in text:
            cells.extend([
                dict(cell_type="markdown", id="psutil-fix-note", metadata={}, source=note.splitlines(True)),
                dict(cell_type="code", id="psutil-fix-setup", metadata={}, source=source.splitlines(True),
                     outputs=[], execution_count=None),
            ])
            inserted += 1
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    assert inserted == 1, "Expected exactly one validation cell"
    notebook["cells"] = cells
    notebook["metadata"].setdefault("colab", {})["name"] = NAME + ".ipynb"
    target = DIST / (NAME + ".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    build(parser.parse_args().source)
