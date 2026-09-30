"""Add one missing author dependency without updating the old RoSteALS stack."""

from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "author-night/dist/Three_Method_AUTHOR_NIGHT_source.zip"


def main():
    (ROOT / "locks").mkdir(exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        old = archive.read("locks/rosteals.txt").decode()
    incoming = ROOT / "locks/rosteals.in"
    incoming.write_text(old + "\nslackclient==2.9.3\n", encoding="utf-8")
    command = [sys.executable, "-m", "uv", "pip", "compile", str(incoming), "--cache-dir", str(ROOT / ".uv-cache"),
               "--python", sys.executable, "--python-version", "3.9.23", "--python-platform", "x86_64-manylinux_2_35",
               "--no-header", "--no-annotate", "--emit-index-url", "--index-strategy", "unsafe-best-match",
               "-o", str(ROOT / "locks/rosteals.txt")]
    with (ROOT / "locks/resolve.log").open("w", encoding="utf-8") as logfile:
        subprocess.run(command, check=True, stdout=logfile, stderr=subprocess.STDOUT)
    locked = (ROOT / "locks/rosteals.txt").read_text()
    previous = {line for line in old.splitlines() if "==" in line}
    current = {line for line in locked.splitlines() if "==" in line}
    if not previous.issubset(current):
        raise RuntimeError("Existing environment pins changed unexpectedly")
    print("New pins:", sorted(current - previous))


if __name__ == "__main__":
    main()
