"""Sequential, bounded isolated environments with durable per-phase resume."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from wmbench.embedding_bank import _log_tail
from wmbench.environment import worker_environment
from wmbench.io import read_json, write_json
from wmbench_full_payload.runner import cleanup, prepare as prepare_frozen
from wmbench_full_payload.storage import EvidenceError, disjoint, freeze

from .core import complete, context, profiles as phase_names
from .report import report


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process
    bank, root, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(bank, root, scratch)
    disjoint(project, bank, root)
    budget = float(settings.get("max_hours", 6)) * 3600
    if not 0 < budget <= 86400:
        raise ValueError("max_hours must be in (0,24]")
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        bank, root, scratch, models, plan = context(project, settings)
        report(root)
        started = time.monotonic()
        for index, (name, model) in enumerate(models.items(), 1):
            owned()
            cases = plan["cases"][name]
            if all(complete(root, c, plan["identity"]) for c in cases):
                print(f"[{index}/{len(models)}] {name}: REUSE verified results", flush=True)
                continue
            if time.monotonic() - started >= budget:
                print("Session budget reached. Re-run to resume.", flush=True)
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 30 * 1024**3:
                print("STOP: less than 30 GiB scratch space. Results remain saved.", flush=True)
                break
            slot, meta = scratch / "slots" / name, root / name / "_meta"
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": plan["identity"]})
            session = str(time.time_ns())
            job = {"model": model, "project": str(project), "output": str(root), "slot": str(slot),
                   "bank": str(bank), "cases": cases, "settings": settings, "session": session,
                   "identity": plan["identity"]}
            path = meta / "jobs" / f"{session}.json"
            write_json(path, job)
            env = worker_environment(project, slot, model)
            env.update(UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
            failed = True  # Clean up only after every phase completed, including on interrupt.
            try:
                for phase in ("prepare", *phase_names(name)):
                    if phase != "prepare" and all(complete(root, c, plan["identity"], phase) for c in cases):
                        print(f"{name} {phase}: REUSE", flush=True)
                        continue
                    remaining = budget - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Session budget reached")
                    python = sys.executable if phase == "prepare" else read_json(meta / "prepared.json")["python"]
                    log = meta / "logs" / f"{session}_{phase}.log"
                    write_json(meta / "status.json", {"status": "running", "phase": phase, "session": session})
                    print(f"[{index}/{len(models)}] {name}: {phase.upper()}", flush=True)
                    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                        "-m", "wmbench_quality_review.runner", phase, "--settings", str(path)],
                        env, log, slot, min(5400, remaining),
                        {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 100, "telemetry": True},
                        assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{phase}.json", result)
                    if result["exit_code"] or result["reason"]:
                        print(_log_tail(log), flush=True)
                        raise RuntimeError(result["reason"] or f"{phase} failed; see {log}")
                    if phase == "prepare" and read_json(meta / "prepared.json")["status"] != "ready":
                        raise RuntimeError(read_json(meta / "prepared.json").get("reason", "Preparation blocked"))
                    report(root)
                if not all(complete(root, c, plan["identity"]) for c in cases):
                    raise EvidenceError("Worker exited with an incomplete set")
                write_json(meta / "status.json", {"status": "checks_complete", "session": session})
                failed = False
            except EvidenceError:
                failed = True
                raise
            except Exception as error:
                failed = True
                write_json(meta / "status.json", {"status": "error", "reason": str(error), "session": session})
                print(f"{name}: ERROR: {error}", flush=True)
            finally:
                locks = slot / "locks" / name
                if locks.exists():
                    shutil.copytree(locks, meta / "environment", dirs_exist_ok=True)
                report(root)
                if settings.get("cleanup", True) and not failed:
                    cleanup(slot, scratch, plan["identity"])
    report(root)



def prepare(job):
    prepare_frozen(job)
    # Keep degrade.pth as the baseline; clean.pth is a separate diagnostic profile.
    if job["model"]["id"] == "editguard":
        from bank_diagnostics.assets import fetch
        root = Path(job["output"]) / "_assets"
        target = root / "editguard/clean.pth"
        try:
            if not job["settings"].get("download_author_assets", True) and not target.is_file():
                raise FileNotFoundError(f"Supply the author's clean.pth at {target}")
            fetch(root, "editguard/clean.pth", "1w4e1gpdInAv7Lj_NQ7EGgmMuInyfUYgi")
        except Exception as error:
            # A missing new checkpoint must not suppress the historical baseline.
            write_json(root / "editguard/clean_download_error.json", {"reason": str(error)})
            print(f"Clean checkpoint unavailable; baseline can still run: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "historical", "resize256", "crop256", "clean_checkpoint", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    if args.action == "prepare":
        prepare(settings)
    elif args.action in ("historical", "resize256", "crop256", "clean_checkpoint"):
        from .worker import work
        work(settings, args.action)
    elif args.action == "report":
        report(Path(settings["output"]))
    else:
        run(Path(__file__).resolve().parents[2], settings)


if __name__ == "__main__":
    main()
