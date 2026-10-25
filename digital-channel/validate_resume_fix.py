"""Run the original and repair suites with both in-memory fixes installed."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP_SHA = "fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff"


def main():
    started = time.monotonic()
    archive_path = ROOT/"dist/COCO_Digital_Channel_RESUMABLE_source.zip"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == ZIP_SHA
    reports = []
    with tempfile.TemporaryDirectory(prefix="wm-resume-fix-tests-") as folder:
        release = Path(folder)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (release/member.filename).resolve()
                assert target.is_relative_to(release.resolve()) and not member.is_dir()
                data = archive.read(member)
                if member.filename.endswith(".py"):
                    ast.parse(data.decode("utf-8"), member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for name, prefix in [("digital-release.json", release), ("production/release.json", release/"production")]:
            for relative, checksum in json.loads((release/name).read_text("utf-8"))["files"].items():
                assert hashlib.sha256((prefix/relative).read_bytes()).hexdigest() == checksum
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT, release/"src", release/"production/src", release/"tests"])),
                   WM_DIGITAL_TEST_RELEASE=str(release), PYTHONIOENCODING="utf-8")
        bootstrap = ("import unittest,sys; import stegastamp_clean_hotfix as s; "
                     "import digital_resume_hotfix as r; s.install_worker_patch(); r.install_cleanup_patch(); "
                     "suite=unittest.defaultTestLoader.discover(sys.argv[1]); "
                     "result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())")
        for name, tests in [("original_regressions", release/"tests"), ("both_repair_suites", ROOT/"repair-tests")]:
            result = subprocess.run([sys.executable, "-c", bootstrap, str(tests)], env=env,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            reports.append(dict(name=name, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
            print(name, result.returncode)
            print(result.stderr[-4000:])
    report = dict(original_zip_sha256=ZIP_SHA, elapsed_seconds=round(time.monotonic()-started, 2),
                  suites=reports, cuda_run="not executed locally; cache-reset/resume reproduced offline")
    (ROOT/"dist/resume-fix-validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if any(r["exit_code"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
