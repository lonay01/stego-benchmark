"""Kill an isolated worker group if its notebook/controller is abruptly killed.

Run only inside the private process group created by monitored_process. This
wrapper closes the orphan-worker gap when Jupyter kills its direct subprocess
without letting the controller's Python finally blocks run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import psutil


def parent_alive(pid: int, created: float) -> bool:
    try:
        parent = psutil.Process(pid)
        return parent.is_running() and parent.create_time() == created
    except psutil.NoSuchProcess:
        return False


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("Worker command required")
    parent = psutil.Process(os.getppid())
    parent_pid, created = parent.pid, parent.create_time()
    child = subprocess.Popen(command)
    while child.poll() is None:
        if not parent_alive(parent_pid, created):
            if os.name == "posix":
                if os.getpgrp() != os.getpid():
                    raise RuntimeError("Refusing to kill a non-private process group")
                os.killpg(os.getpgrp(), signal.SIGKILL)
            else:
                family = psutil.Process(child.pid)
                for process in family.children(recursive=True):
                    process.kill()
                family.kill()
            return 125
        time.sleep(1)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
