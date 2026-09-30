"""Single-model queue with persisted progress and fail-fast technical gates."""

import os
from pathlib import Path
import sys
import time

from authornight.environment import prepared_environment
from authornight.runner import local_lock, supervise
from authornight.storage import read
from .migration import make_plan
from .report import report


def run(settings, code_hash):
    if os.name != "posix":
        raise RuntimeError("Run GPU inference in Colab/Linux NVIDIA, not the Windows notebook kernel")
    root, scratch = Path(settings["output"]).resolve(), Path(settings["scratch"]).resolve()
    project = Path(__file__).resolve().parent.parent
    scratch.mkdir(parents=True, exist_ok=True)
    make_plan(settings, code_hash)
    deadline = time.time() + float(settings["max_hours"]) * 3600
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(project)
    template = {"output": str(root), "scratch": str(scratch), "project": str(project),
                "method": "rosteals", "worker_module": "rostealsfix.worker"}

    def stage(phase, python, env, minutes=None, **extra):
        if deadline - time.time() < 90:
            print("Session budget reached; Run All again to resume.", flush=True)
            return False
        end = min(deadline, time.time() + minutes * 60) if minutes else deadline
        outcome = supervise(python, {**template, "phase": phase, **extra}, env, end)
        report(root)
        if outcome["status"] != "returned":
            print("Queue stopped at", phase, "— see the saved log. No later GPU jobs launched.", flush=True)
            return False
        return True

    with local_lock(root):
        for phase, minutes in (("import", 60), ("prepare", 45)):
            if not stage(phase, sys.executable, base_env, minutes, settings=settings, code_hash=code_hash):
                return report(root)
        meta = read(root / "_meta/rosteals/prepared.json")
        env = prepared_environment("rosteals", project, root, scratch, Path(meta["source"]))
        for phase, minutes in (("imports", 15), ("preflight", 30), ("coco", None)):
            if not stage(phase, meta["python"], env, minutes, source=meta["source"]):
                return report(root)
        result = report(root)
        if not result["native_complete"] or not result["native_checks_passed"]:
            print("Native cases incomplete or a technical check failed. Attacks not started; inspect report.", flush=True)
            return result
        stage("corruptions", meta["python"], env, source=meta["source"])
        return report(root)
