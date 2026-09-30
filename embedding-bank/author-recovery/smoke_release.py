"""Execute tests and saved-bit replay from the distributed ZIP, not checkout code."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-zip", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wm-author-recovery-smoke-") as directory:
        project = Path(directory).resolve()
        with zipfile.ZipFile(ROOT / "dist/RoSteALS_SepWater_RECOVERY_source.zip") as archive:
            for entry in archive.infolist():
                path = (project / entry.filename).resolve()
                if not path.is_relative_to(project) or path == project:
                    raise ValueError("Unsafe release ZIP")
            archive.extractall(project)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project)
        tests = subprocess.run([sys.executable, "-m", "pytest", str(project / "tests"), "-q"],
                               cwd=project, env=env, capture_output=True, text=True, check=True)
        print(tests.stdout)
        result_path = project / "replay.json"
        subprocess.run([sys.executable, "-m", "authorrecovery.ecc", "--report-zip", args.report_zip,
                        "--output", str(result_path)], cwd=project, env=env, check=True)
        result = json.loads(result_path.read_text())
        assert result["totals"]["author_png"]["cases"] == 500
        assert result["totals"]["author_png"]["data_exact"] == 496
        assert result["totals"]["raw_internal"]["data_exact"] == 498
        summary = {"scope": "Distributed ZIP execution on CPU, not CUDA", "tests": tests.stdout.strip(),
                   "saved_bit_replay": result, "gpu_run": False}
        (ROOT / "validation/packaged_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
