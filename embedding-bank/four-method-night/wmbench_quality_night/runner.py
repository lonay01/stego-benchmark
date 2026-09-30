"""Sequential GPU queue, bounded sessions, independent profile failures."""

import argparse
from pathlib import Path
import shutil
import sys
import time

from wmbench.embedding_bank import _log_tail
from wmbench.environment import worker_environment
from wmbench.io import read_json, write_json
from wmbench_full_payload.runner import cleanup, prepare as prepare_frozen
from wmbench_full_payload.storage import EvidenceError, freeze
from wmbench_native_diagnostics.core import copy_frozen

from .core import context, import_previous, complete
from .environment import constrain_build_dependencies, verify_dependencies
from .report import report


def prepare(job):
    meta = Path(job["output"]) / job["model"]["id"] / "_meta"
    old_runtime = Path(job["bank"]) / job["model"]["id"] / "_meta/environment/current_runtime.json"
    version = read_json(old_runtime)["python"] if old_runtime.is_file() else None
    with constrain_build_dependencies(meta / "historical_installed.freeze.txt", version):
        prepare_frozen(job)
    prepared = read_json(meta / "prepared.json")
    if prepared.get("status") != "ready":
        return
    verify_dependencies(meta, prepared)
    if job["model"]["id"] == "editguard":
        from bank_diagnostics.assets import fetch
        root = Path(job["output"]) / "_assets"
        target = root / "editguard/clean.pth"
        previous = Path(job["settings"]["previous"]) / "_assets/editguard/clean.pth"
        # The previous downloaded author asset is reused, not downloaded again.
        if previous.is_file() and previous.with_name("clean.pth.receipt.json").is_file():
            from wmbench.io import sha256
            receipt = previous.with_name("clean.pth.receipt.json")
            if read_json(receipt)["sha256"] != sha256(previous):
                raise EvidenceError("Prior clean checkpoint does not match its receipt")
            copy_frozen(previous, target)
            copy_frozen(receipt, target.with_name("clean.pth.receipt.json"))
        try:
            if not job["settings"].get("download_author_assets", True) and not target.is_file():
                raise FileNotFoundError(f"Supply author clean.pth: {target}")
            fetch(root, "editguard/clean.pth", "1w4e1gpdInAv7Lj_NQ7EGgmMuInyfUYgi")
        except Exception as error:
            write_json(meta / "clean_download_error.json", {"reason": str(error)})
            print(f"Clean checkpoint unavailable; degrade baseline can still run: {error}", flush=True)


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process
    root = Path(settings["output"]).resolve()
    budget = float(settings.get("max_hours", 12)) * 3600
    if not 0 < budget <= 24 * 3600:
        raise ValueError("MAX_HOURS must be in (0,24]")
    started = time.monotonic()
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        bank, root, scratch, models, plan = context(project, settings)
        print("Inputs/locks verified. Importing compatible completed v1 measurements...", flush=True)
        import_previous(project, root, Path(settings["previous"]), plan)
        report(root)
        for index, (name, model) in enumerate(models.items(), 1):
            owned()
            cases = plan["cases"][name]
            # One artifact audit per controller pass, not every report refresh.
            pending = [p for p in plan["profiles"][name]
                       if not all(complete(root, c, p, plan["identity"]) for c in cases)]
            if not pending:
                print(f"[{index}/4] {name}: REUSE verified measurements", flush=True)
                continue
            if time.monotonic() - started >= budget:
                print("Session budget reached. Run again to resume.", flush=True)
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 30 * 1024**3:
                print("STOP: <30 GiB local SSD; results are saved.", flush=True)
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
            failures, returned_normally = [], False
            try:
                for phase in ("prepare", *pending):
                    owned()
                    remaining = budget - (time.monotonic() - started)
                    if remaining <= 0:
                        failures.append("Session budget reached; remaining profiles resumable")
                        break
                    python = sys.executable if phase == "prepare" else read_json(meta / "prepared.json")["python"]
                    log = meta / "logs" / f"{session}_{phase}.log"
                    write_json(meta / "status.json", {"status": "running", "phase": phase, "session": session})
                    print(f"[{index}/4] {name}: {phase.upper()}", flush=True)
                    observed = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                        "-m", "wmbench_quality_night.runner", phase, "--settings", str(path)],
                        env, log, slot, min(7200 if phase == "prepare" else budget, remaining),
                        {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 100, "telemetry": True},
                        assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{phase}.json", observed)
                    failed = bool(observed["exit_code"] or observed["reason"])
                    if not failed and phase == "prepare":
                        failed = read_json(meta / "prepared.json").get("status") != "ready"
                    if failed:
                        reason = observed["reason"] or f"{phase} failed; see {log}"
                        failures.append(reason)
                        print(_log_tail(log), flush=True)
                        write_json(meta / f"phase_{phase}.json", {"status": "error", "reason": reason})
                        # Preparation/evidence failures block a method; an ordinary
                        # profile failure must not suppress its independent sibling.
                        if phase == "prepare" or observed["exit_code"] == 2 or observed["reason"]:
                            break
                    else:
                        write_json(meta / f"phase_{phase}.json", {"status": "finished"})
                    report(root)
                returned_normally = True
            finally:
                locks = slot / "locks" / name
                if locks.exists():
                    shutil.copytree(locks, meta / "environment", dirs_exist_ok=True)
                state = "interrupted" if not returned_normally else "partial" if failures else "queue_finished"
                write_json(meta / "status.json", {"status": state, "reason": "; ".join(failures), "session": session})
                report(root)
                if settings.get("cleanup", True) and not failures and returned_normally:
                    cleanup(slot, scratch, plan["identity"])
        summary = report(root, archive=True)
        print(f"DONE: measured {summary['measured']}/{summary['expected']}; "
              f"technical errors={summary['technical_errors']}; consistency failures={summary['check_failures']}. "
              f"Complete={summary['complete']}. Report: {root / 'Four_Method_NIGHT_report.zip'}", flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "historical", "resize256", "resize224", "clean_checkpoint", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    try:
        if args.action == "prepare":
            prepare(settings)
        elif args.action == "report":
            report(Path(settings["output"]), archive=True)
        elif args.action == "run":
            run(Path(__file__).resolve().parents[2], settings)
        else:
            from .worker import work
            work(settings, args.action)
    except EvidenceError as error:
        print(f"EVIDENCE STOP: {error}. Existing evidence was not replaced.", flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
