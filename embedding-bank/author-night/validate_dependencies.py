"""Resolve Linux wheels without installing the three GPU stacks on Windows."""
import subprocess
import sys
from pathlib import Path
from authornight.environment import REQUIREMENTS, PYTHON
from authornight.storage import atomic

root = Path(__file__).resolve().parent
for name in sys.argv[1:] or REQUIREMENTS:
    incoming = root / "locks" / (name + ".in")
    atomic(incoming, ("\n".join(REQUIREMENTS[name])+"\n").encode())
    args = [sys.executable, "-m", "uv", "pip", "compile", str(incoming),
            "--python", sys.executable, "--python-version", PYTHON[name],
            "--python-platform", "x86_64-manylinux_2_35", "--no-header", "--no-annotate",
            "--emit-index-url", "-o", str(root / "locks" / (name + ".txt"))]
    if name == "rosteals":
        args += ["--extra-index-url", "https://download.pytorch.org/whl/cu113", "--index-strategy", "unsafe-best-match"]
    with (root / 'locks' / (name + '.resolve.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    print(name, 'resolved' if result.returncode == 0 else 'FAILED', flush=True)
    if result.returncode:
        print((root/'locks'/(name+'.resolve.log')).read_text('utf-8')[-5000:])
        result.check_returncode()
