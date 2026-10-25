"""Validate the VideoSeal notebook/overlay against the immutable source ZIP."""

import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
from unittest.mock import patch

import nbformat
from build_videoseal_fix import (ROOT, NAME, PREVIOUS, BASE_NAME, BASE_SHA256,
    PREVIOUS_SHA256, OLD_N_IMAGES, NEW_N_IMAGES, build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional downloaded checkpoint for a real-byte cache/resume smoke test")
    args = parser.parse_args()
    notebook_path = build()
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    previous = nbformat.read(ROOT / "dist" / (PREVIOUS + ".ipynb"), as_version=4)
    old_cells = {cell.id: cell for cell in previous.cells}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        ast.parse(cell.source, filename=cell.id)
        if cell.id in old_cells:
            expected = old_cells[cell.id].source
            if cell.id in {"neural-08", "neural-10", "neural-12"}:
                expected = expected.replace('"neural_sdxl_identity_fix"', '"neural_videoseal_download_fix"')
            if cell.id == "neural-02":
                expected = expected.replace(OLD_N_IMAGES, NEW_N_IMAGES)
            assert cell.source == expected, f"Unexpected existing-cell change: {cell.id}"
    archive_path = ROOT / "dist" / (BASE_NAME + "_source.zip")
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == BASE_SHA256
    assert hashlib.sha256((ROOT / "dist" / (PREVIOUS + ".ipynb")).read_bytes()).hexdigest() == PREVIOUS_SHA256
    module_checksums = {}
    checkpoint_audit = None
    with tempfile.TemporaryDirectory(prefix="neural-videoseal-validation-") as directory:
        target = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            files, parsed = {}, 0
            for name in archive.namelist():
                assert (target / name).resolve().is_relative_to(target.resolve())
                content = archive.read(name)
                files[name] = hashlib.sha256(content).hexdigest()
                if name.endswith(".py"):
                    ast.parse(content.decode("utf-8"), filename=name)
                    parsed += 1
            archive.extractall(target)
        scope = dict(PROJECT=target, hashlib=hashlib, ENV={})
        for cell_id, module in (("identity-repair-code", "neural_sdxl_identity_fix.py"),
                                ("videoseal-repair-code", "neural_videoseal_download_fix.py")):
            cell = next(c.source for c in notebook.cells if c.id == cell_id)
            module_path = target / "src" / module
            expected = (ROOT / "repairs" / module).read_text(encoding="utf-8").encode("utf-8")
            for _ in range(2):
                exec(compile(cell, cell_id, "exec"), scope)
                assert module_path.read_bytes() == expected
            module_path.write_bytes(b"different existing recovery code")
            try:
                exec(compile(cell, cell_id, "exec"), scope)
            except ValueError:
                assert module_path.read_bytes() == b"different existing recovery code"
            else:
                raise AssertionError("Existing different recovery must not be overwritten")
            module_path.write_bytes(expected)
            module_checksums[module] = hashlib.sha256(expected).hexdigest()
        for name in ("test_identity_fix.py", "test_videoseal_download_fix.py"):
            content = (ROOT / "repairs" / name).read_bytes()
            ast.parse(content.decode("utf-8"), filename=name)
            (target / "tests" / name).write_bytes(content)
        fixtures = target / "tests/fixtures"
        fixtures.mkdir()
        (fixtures / "videoseal_bank_identity.json").write_bytes(
            (ROOT / "repairs/fixtures/videoseal_bank_identity.json").read_bytes())
        env = {**os.environ, **scope["ENV"], "PYTHONPATH": os.pathsep.join([
            str(target / "src"), str(target / "production/src")]), "PYTHONIOENCODING": "utf-8",
            "HF_HOME": str(target / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(target / "tests"), "-v"],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        if args.checkpoint:
            # No unpickling or execution of downloaded checkpoint contents.
            sys.path[:0] = [str(target / "src"), str(target / "production/src")]
            import neural_videoseal_download_fix as fix
            from wmbench_bank.contracts import models_for_run
            bank = target / "audit-bank"
            identity = bank / "videoseal/_meta/model_identity.json"
            identity.parent.mkdir(parents=True)
            identity.write_bytes((fixtures / "videoseal_bank_identity.json").read_bytes())
            before = identity.read_bytes()
            job = dict(model=models_for_run(target / "production")["videoseal"], bank=str(bank),
                results=str(target / "audit-results"), slot=str(target / "audit-slot"),
                session="123", deadline=0)  # No network even with an expired deadline.
            local = Path(job["slot"]) / "assets/videoseal" / fix.CHECKPOINT_KEY
            fix.publish(args.checkpoint, local, fix.CHECKPOINT_SHA256)
            with patch.object(fix.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
                fix.ensure_videoseal_checkpoint(job)
                resumed = fix.ensure_videoseal_checkpoint({**job, "session": "456",
                    "slot": str(target / "audit-new-runtime")})
            assert fix.file_sha256(resumed) == fix.CHECKPOINT_SHA256
            assert identity.read_bytes() == before
            checkpoint_audit = dict(sha256=fix.CHECKPOINT_SHA256, bytes=resumed.stat().st_size,
                saved_bank_registry_matches=True, real_byte_cache_resume_tested=True, checkpoint_loaded=False)
        assert all(hashlib.sha256((target / name).read_bytes()).hexdigest() == checksum
                   for name, checksum in files.items()), "Original release file modified"
    output = result.stdout + result.stderr
    match = re.search(r"Ran (\d+) tests?", output)
    value = dict(notebook_valid=True, notebook_code_cells=count, original_python_files_checked=parsed,
        original_release_files_unchanged=True, previous_notebook_unchanged=True,
        previous_code_changes="3 entrypoint names; N_IMAGES=100 to retain the active full-run selection",
        base_archive_sha256=BASE_SHA256, module_sha256=module_checksums,
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        offline_tests_exit_code=result.returncode, tests_run=int(match[1]) if match else None,
        test_output=output, checkpoint_audit=checkpoint_audit,
        cuda_tested=False, official_weight_inference_tested=False, python=sys.version)
    report = ROOT / "dist" / (NAME + "_validation.json")
    report.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({k: v for k, v in value.items() if k != "test_output"}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
