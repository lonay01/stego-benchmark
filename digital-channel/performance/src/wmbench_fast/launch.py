"""Detached local controller; a browser disconnect does not close its stdout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from wmbench.io import read_json, write_json
from wmbench.colab_canary import _boot_id, stop_process_tree


def alive(record):
    if record.get("boot") != _boot_id():
        return False
    try:
        proc = psutil.Process(record["pid"])
        return (proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                and proc.create_time() == record["created"] and "wmbench_fast.launch" in proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "supervise", "stop"])
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    if args.action == "start":
        from .lease_io import storage_preflight
        storage_preflight(settings)
    meta = Path(settings["output"])/"_reports"
    meta.mkdir(parents=True, exist_ok=True)
    pidfile = meta/"fast_launcher.json"
    old = read_json(pidfile) if pidfile.exists() else {}
    if args.action == "start":
        if old and alive(old):
            print(f"Already running: PID {old['pid']}; log {old['log']}. No duplicate launch.")
            return
        # Snapshot settings so editing the notebook cannot mutate an active run.
        stamp = str(time.time_ns())
        config = meta/"sessions"/(stamp+"_settings.json")
        write_json(config, settings)
        log = meta/"sessions"/(stamp+"_controller.log")
        with log.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen([sys.executable, "-m", "wmbench_fast.launch", "supervise", "--settings", str(config)],
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(pidfile, dict(pid=proc.pid, created=psutil.Process(proc.pid).create_time(), boot=_boot_id(),
                                 log=str(log), config=str(config)))
        print(f"Started detached controller, PID {proc.pid}. Log: {log}")
        print("Browser disconnect is OK; Colab runtime termination still stops the calculation.")
    elif args.action == "supervise":
        child = subprocess.Popen([sys.executable, "-m", "wmbench_fast.runner", "run", "--settings", str(args.settings)],
                                 start_new_session=True)
        import signal
        def interrupted(_sig, _frame):
            # Worker process guards reap descendants after their parent is gone.
            stop_process_tree(child)
            raise SystemExit(130)
        signal.signal(signal.SIGTERM, interrupted)
        code = child.wait()
        write_json(args.settings.with_name(args.settings.stem+"_exit.json"), dict(exit_code=code, ended=time.time()))
        raise SystemExit(code)
    else:
        if not old or not alive(old):
            print("No matching live fast controller; nothing stopped.")
            return
        # PID, creation time, boot and command line have all been verified.
        psutil.Process(old["pid"]).terminate()
        print("Stop requested; completed/periodic checkpoints remain on Drive. Wait for shutdown before restarting.")


if __name__ == "__main__":
    main()
