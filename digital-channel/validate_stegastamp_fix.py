"""Validate the narrow repair against the byte-identical original release."""

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
EXPECTED = "fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff"


def main():
    started = time.monotonic()
    archive_path = ROOT/"dist/COCO_Digital_Channel_RESUMABLE_source.zip"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == EXPECTED
    reports = []
    with tempfile.TemporaryDirectory(prefix="wm-stegastamp-replay-tests-") as directory:
        release = Path(directory)
        modules = 0
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (release/member.filename).resolve()
                assert target.is_relative_to(release.resolve()) and not member.is_dir()
                data = archive.read(member)
                if member.filename.endswith(".py"):
                    ast.parse(data.decode("utf-8"), member.filename)
                    modules += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for filename, prefix in [("digital-release.json", release), ("production/release.json", release/"production")]:
            manifest = json.loads((release/filename).read_text("utf-8"))
            for name, checksum in manifest["files"].items():
                assert hashlib.sha256((prefix/name).read_bytes()).hexdigest() == checksum
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT, release/"src", release/"production/src", release/"tests"])),
                   WM_DIGITAL_TEST_RELEASE=str(release), PYTHONIOENCODING="utf-8")
        regression = ("import unittest,sys; import stegastamp_clean_hotfix as fix; fix.install_worker_patch(); "
                      "suite=unittest.defaultTestLoader.discover(sys.argv[1]); "
                      "result=unittest.TextTestRunner(verbosity=2).run(suite); "
                      "sys.exit(not result.wasSuccessful())")
        commands = [
            ("original_release_tests_with_repair_installed", [sys.executable, "-c", regression, str(release/"tests")]),
            ("repair_tests", [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT/"repair-tests"), "-v"]),
        ]
        for name, command in commands:
            result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            reports.append(dict(name=name, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
            print(name, "exit=", result.returncode)
            print(result.stderr[-3500:])
    report = dict(original_zip_sha256=EXPECTED, original_release_modules_parsed=modules,
                  elapsed_seconds=round(time.monotonic()-started, 2), suites=reports,
                  gpu_replay="not executed locally; repeated GPU Clean checks occur in the notebook")
    (ROOT/"dist/stegastamp-fix-validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if any(row["exit_code"] for row in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
