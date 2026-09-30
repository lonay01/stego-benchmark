"""Two-model queue, no DistSeal. Runtime boundaries release GPU allocations."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from authornight.environment import prepared_environment
from authornight.runner import local_lock, supervise
from authornight.storage import read
from .migration import make_plan


def run(settings, code_hash):
    if os.name != "posix":
        raise RuntimeError("Use Colab or a Linux NVIDIA server for the GPU queue")
    root, scratch = Path(settings["output"]).resolve(), Path(settings["scratch"]).resolve()
    project = Path(__file__).resolve().parent.parent
    scratch.mkdir(parents=True, exist_ok=True)
    make_plan(settings, code_hash)
    deadline = time.time() + float(settings["max_hours"]) * 3600
    template = {"output": str(root), "scratch": str(scratch), "project": str(project),
                "worker_module": "authorrecovery.worker"}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project)
    from .report import report

    def stage(method, phase, python, environment, minutes, **extra):
        if deadline - time.time() < 90:
            return False
        outcome = supervise(python, {**template, "method": method, "phase": phase, **extra},
                            environment, min(deadline, time.time() + minutes * 60))
        report(root)
        return outcome["status"] == "returned"

    with local_lock(root):
        if not stage("all", "import", sys.executable, env, 60, settings=settings, code_hash=code_hash):
            raise RuntimeError("Parent import incomplete; see the log and rerun without deleting results")
        # A tiny independent CPU environment: no CUDA or SepWater model needed.
        if stage("ecc", "prepare_ecc", sys.executable, env, 10):
            cpu = read(root / "_meta/ecc/prepared.json")
            stage("ecc", "ecc", cpu["python"], env, 15)
        ready = {}
        for method in ("rosteals", "sepwater"):
            if not stage(method, "prepare", sys.executable, env, 35):
                continue
            meta = read(root / "_meta" / method / "prepared.json")
            method_env = prepared_environment(method, project, root, scratch, Path(meta["source"]))
            if stage(method, "preflight", meta["python"], method_env, 30, source=meta["source"]):
                ready[method] = (meta, method_env)
        if "rosteals" in ready:
            meta, method_env = ready["rosteals"]
            if not stage("rosteals", "coco", meta["python"], method_env, 180, source=meta["source"]):
                ready.pop("rosteals")
        # Finish the shorter, already-started SepWater sweep first.
        for method in ("sepwater", "rosteals"):
            if method not in ready:
                continue
            meta, method_env = ready[method]
            stage(method, "corruptions", meta["python"], method_env, 300, source=meta["source"])
        result = report(root)
    print("Recovery report:", root / "_reports/report.md", flush=True)
    return result
