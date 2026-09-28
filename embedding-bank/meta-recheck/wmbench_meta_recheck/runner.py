"""One notebook, sequential isolated processes, durable results and bounded runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

from wmbench.colab_canary import controller_lease, monitored_process
from wmbench.embedding_bank import _log_tail
from wmbench.environment import worker_environment
from wmbench.io import read_json, write_json
from wmbench_full_payload.runner import cleanup
from wmbench_full_payload.storage import EvidenceError, disjoint, freeze
from wmbench_author_reference.core import context as reference_context
from wmbench_author_reference.report import report as reference_report
from . import meta
from .report import report, review_zip


def paths(project, settings):
    bank, root, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(bank, root, scratch)
    disjoint(project, bank, root)
    if settings.get("n_images") != 5:
        raise ValueError("This diagnostic is fixed at five images. Do not set 100 here.")
    if settings.get("models") != ["meta_hidden_revisited", "rosteals", "sepwater"]:
        raise ValueError("This release checks exactly Meta, RoSteALS and SepWater")
    if not (root / "recheck.json").exists() and root.exists():
        foreign = [p.name for p in root.iterdir() if p.name not in ("_release", "_reports")]
        if foreign:
            raise EvidenceError(f"Output is not a new diagnostic folder: {foreign[:5]}")
    freeze(root / "recheck.json", {"schema": "meta-sepwater-rosteals-recheck-v1", "bank": str(bank),
                                   "models": settings["models"], "n_images": 5})
    return bank, root, scratch


def meta_run(project, root, scratch, plan, settings, remaining, owned):
    mroot = root / "meta_hidden"
    if all(meta.complete(mroot, c, plan["identity"]) for c in plan["cases"]):
        print("Meta: REUSE verified 50 measurements", flush=True)
        return
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 30 * 1024**3:
        raise RuntimeError("Less than 30 GiB scratch free. Keep results; use a fresh runtime.")
    slot = scratch / "slots/meta_hidden_revisited"
    slot.mkdir(parents=True, exist_ok=True)
    freeze(slot / ".full-payload-owner.json", {"owner": plan["identity"]})
    session = str(time.time_ns())
    job = {"project": str(project), "root": str(mroot), "slot": str(slot), "device": settings["device"]}
    path = mroot / "_meta/jobs" / f"{session}.json"
    write_json(path, job)
    env = worker_environment(project, slot, meta.MODEL)
    env.update(UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
    failed = True
    try:
        for phase in ("prepare", "work"):
            owned()
            if remaining() <= 0:
                raise TimeoutError("Session budget reached; resume with the same settings")
            python = sys.executable if phase == "prepare" else read_json(mroot / "_meta/prepared.json")["python"]
            log = mroot / "_meta/logs" / f"{session}_{phase}.log"
            write_json(mroot / "_meta/status.json", {"status": "running", "phase": phase})
            print(f"[1/3] Meta: {phase.upper()}", flush=True)
            result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                "-m", "wmbench_meta_recheck.runner", f"meta-{phase}", "--settings", str(path)],
                env, log, slot, min(5400, remaining()),
                {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 100, "telemetry": True}, assert_owned=owned)
            write_json(mroot / "_meta/timings" / f"{session}_{phase}.json", result)
            if result["exit_code"] or result["reason"]:
                print(_log_tail(log), flush=True)
                raise RuntimeError(result["reason"] or f"Meta {phase} failed: {log}")
            report(root)
        if not all(meta.complete(mroot, c, plan["identity"]) for c in plan["cases"]):
            raise EvidenceError("Meta worker returned with missing measurements")
        write_json(mroot / "_meta/status.json", {"status": "checks_complete"})
        failed = False
    except EvidenceError:
        raise
    except Exception as error:
        write_json(mroot / "_meta/status.json", {"status": "error", "reason": str(error)})
        print(f"Meta: ERROR: {error}", flush=True)
    finally:
        if not failed:
            cleanup(slot, scratch, plan["identity"])
        report(root)


def run(project, settings):
    budget = float(settings.get("max_hours", 6)) * 3600
    if not 0 < budget <= 86400:
        raise ValueError("max_hours must be in (0,24]")
    bank, root, scratch = paths(project, settings)
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        plan = meta.context(project, bank, root / "meta_hidden")
        ref_settings = {**settings, "output": str(root / "references"), "models": ["rosteals", "sepwater"]}
        reference_context(project, ref_settings)
        reference_report(root / "references")
        report(root)
        start = time.monotonic()
        remaining = lambda: max(0, budget - (time.monotonic() - start))
        try:
            meta_run(project, root, scratch, plan, settings, remaining, owned)
            if remaining() > 0:
                from .reference_runner import run as references_run
                ref_settings["max_hours"] = remaining() / 3600
                print("[2–3/3] RoSteALS and SepWater: historical raw100 replay and independent cores", flush=True)
                references_run(project, ref_settings)
            else:
                print("Time budget reached. Re-run cells 1–3 to resume.", flush=True)
        finally:
            report(root)
            review_zip(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report", "meta-prepare", "meta-work"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    if args.action == "meta-prepare":
        meta.prepare(Path(settings["project"]), Path(settings["root"]), Path(settings["slot"]), settings["device"])
    elif args.action == "meta-work":
        root = Path(settings["root"])
        meta.work(Path(settings["project"]), root, read_json(root / "plan.json"))
    elif args.action == "report":
        root = Path(settings["output"])
        if (root / "references/plan.json").exists():
            reference_report(root / "references")
        report(root)
        review_zip(root)
    else:
        run(Path(__file__).resolve().parents[2], settings)


if __name__ == "__main__":
    main()
