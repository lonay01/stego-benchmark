"""Bounded unattended queue. Never remove historical inputs or results."""

from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from .environment import prepared_environment
from .spec import make_plan
from .storage import read, write

# Hard per-stage ceilings in minutes. Fast stages leave time for later stages.
# Long stages checkpoint and yield; another Run All continues the same queue.
STAGES = [
    ("rosteals", "coco", 65, {}),
    ("sepwater", "coco", 65, {}),
    ("distseal", "coco", 45, {}),
    ("distseal", "generation", 60, {"generation_rounds": 1}),
    ("rosteals", "corruptions", 45, {}),
    ("sepwater", "corruptions", 40, {}),
    ("distseal", "attacks", 30, {"cohort": "coco"}),
    ("distseal", "attacks", 45, {"cohort": "generated"}),
    ("distseal", "generation", 9999, {}),
    ("distseal", "attacks", 45, {"cohort": "generated"}),
    ("distseal", "inception", 30, {}),
]


@contextmanager
def local_lock(root):
    """OS lock prevents duplicates on this machine; fresh Drive lease blocks others."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    from .storage import digest
    import tempfile
    lock_path = Path(tempfile.gettempdir()) / ("authornight-" + digest(str(root))[:20] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another queue owns this output folder") from error
        status = root / "_logs/live.json"
        if status.exists():
            prior = read(status)
            if prior.get("running") and prior.get("host") != socket.gethostname() and time.time()-prior.get("heartbeat", 0) < 180:
                raise RuntimeError("Another runtime has a fresh Drive lease; do not run two copies")
        try:
            yield
        finally:
            write(status, {"running": False, "heartbeat": time.time(), "host": socket.gethostname()})
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def heartbeat(root, scratch, name, started):
    stats = {"running": True, "host": socket.gethostname(), "heartbeat": time.time(),
             "stage": name, "elapsed_minutes": (time.monotonic()-started)/60,
             "free_scratch_gib": shutil.disk_usage(scratch).free / 1024**3}
    try:
        stats["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True, timeout=5).strip()
    except Exception:
        stats["gpu"] = "unavailable"
    write(root / "_logs/live.json", stats)
    print(f"{name}: {stats['elapsed_minutes']:.1f} min; SSD {stats['free_scratch_gib']:.1f} GiB free; {stats['gpu']}", flush=True)


def supervise(python, job, env, deadline):
    root, scratch = Path(job["output"]), Path(job["scratch"])
    label = f"{job['method']}_{job['phase']}_{time.time_ns()}"
    job_path = root / "_logs" / (label + ".json")
    job["deadline"] = deadline - 15  # let the worker commit its last small record
    write(job_path, job)
    log_path = job_path.with_suffix(".log")
    started = time.monotonic()
    status = "failed"
    process = None
    print(f"START {label}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen([str(python), "-u", "-m", "authornight.worker", str(job_path)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                heartbeat(root, scratch, label, started)
                if time.time() >= deadline:
                    status = "time_budget_reached"
                    stop_group(process)
                    break
                if shutil.disk_usage(scratch).free < 3*1024**3:
                    status = "scratch_space_low"
                    stop_group(process)
                    break
                try:
                    process.wait(timeout=min(30, max(.1, deadline-time.time())))
                except subprocess.TimeoutExpired:
                    pass
            else:
                status = "returned" if process.returncode == 0 else "failed"
        finally:
            if process is not None:
                stop_group(process)
    outcome = {"stage": label, "method": job["method"], "phase": job["phase"], "status": status,
               "returncode": process.returncode if process else None, "seconds": time.monotonic()-started,
               "log": log_path.relative_to(root).as_posix()}
    write(job_path.with_suffix(".outcome.json"), outcome)
    print(f"END {label}: {status}. Completion is counted from saved records, not exit code.", flush=True)
    if status == "failed":
        print(log_path.read_text("utf-8", errors="replace")[-4000:], flush=True)
    return outcome


def run(settings, code_hash):
    if os.name != "posix":
        raise RuntimeError("Run this GPU queue on Colab or a Linux NVIDIA server")
    root, scratch = Path(settings["output"]).resolve(), Path(settings["scratch"]).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent.parent
    make_plan(settings, code_hash)
    deadline = time.time() + float(settings["max_hours"])*3600
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(project)
    template = {"output": str(root), "scratch": str(scratch), "project": str(project)}
    from .report import report
    with local_lock(root):
        inputs = supervise(sys.executable, {**template, "method": "all", "phase": "inputs"}, base_env, min(deadline, time.time()+1200))
        if inputs["status"] != "returned":
            report(root)
            raise RuntimeError("Inputs incomplete; inspect the persistent log and rerun")
        prepared, failed = {}, set()
        for method, phase, minutes, extra in STAGES:
            if deadline-time.time() < 120:
                break
            if method in failed:
                continue
            if method not in prepared:
                result = supervise(sys.executable, {**template, "method": method, "phase": "prepare"},
                                   base_env, min(deadline, time.time()+1800))
                if result["status"] != "returned":
                    failed.add(method)
                    report(root)
                    continue
                meta = read(root / "_meta" / method / "prepared.json")
                prepared[method] = meta
            meta = prepared[method]
            env = prepared_environment(method, project, root, scratch, Path(meta["source"]))
            stage_deadline = min(deadline, time.time()+minutes*60)
            if minutes == 9999:
                stage_deadline -= 75*60  # reserve attacks and optional Inception Score
            if stage_deadline-time.time() < 120:
                continue
            result = supervise(meta["python"], {**template, "method": method, "phase": phase,
                               "source": meta["source"], **extra}, env, stage_deadline)
            report(root)
        result = report(root)
    print(f"Report saved: {root / '_reports/report.md'}", flush=True)
    return result
