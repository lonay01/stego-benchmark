"""Sequential five-model controller using the proven isolated-environment loader."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from wmbench.embedding_bank import _log_tail
from wmbench.environment import worker_environment
from wmbench.io import digest, read_json, sha256, write_json
from wmbench_full_payload.runner import prepare, cleanup
from wmbench_full_payload.storage import EvidenceError, disjoint, freeze
from wmbench_native_canary.contracts import models_for_trial

from .core import METHODS, POLICY, case_key, complete, copy_frozen, freeze_cases
from .report import report
from .recovery import freeze_run_identity, source_manifest


def context(project, settings):
    bank, output, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(bank, output, scratch)
    trial = read_json(bank / "trial.json")
    if trial.get("policy", {}).get("schema") != "native-resolution-canary-v1":
        raise ValueError("BANK_ROOT must be the completed native-resolution canary, not the older 512/full-payload bank")
    selected = settings.get("models", list(METHODS))
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - set(METHODS):
        raise ValueError("Select unique IDs from the five diagnostic methods")
    models = models_for_trial(project)
    for name in selected:
        if trial["models"][name] != models[name]:
            raise EvidenceError(f"Model definition changed since the canary: {name}")
    data = read_json(bank / "_inputs/manifest.json")
    images = data["images"][:5]
    if len(images) != 5:
        raise ValueError("The historical five-image prefix is required")
    identity = {"policy": POLICY, "historical_trial": digest(trial), "models": {n: models[n] for n in selected},
                "source_code": digest(source_manifest(project))}
    identity = freeze_run_identity(project, output, identity)
    cases = {}
    for name in selected:
        print(f"Preflight historical evidence: {name}", flush=True)
        cases[name] = freeze_cases(bank, output, models[name], images)
        old, new = bank / name / "_meta", output / name / "_meta"
        for filename in ("model_identity.json", "dependency_identity.json"):
            copy_frozen(old / filename, new / f"historical_{filename}")
        # Reuse the exact frozen packages; never resolve a newer dependency set.
        for filename in ("requirements.lock", "request.json"):
            copy_frozen(old / "environment" / filename, new / "environment" / filename)
        freeze(new / "historical_runtime.json", read_json(
            output / name / "cases" / cases[name][0]["stem"] / "historical_encode.json")["execution"])
    plan = {"identity": digest(identity), "cases": cases}
    freeze(output / "plan.json", plan)
    return bank, output, scratch, {n: models[n] for n in selected}, plan


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process
    bank, root, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(bank, root, scratch)
    disjoint(project, root, bank)  # Project may be inside scratch, never inside old results.
    # Take the lease before freezing inputs or writing model records.
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        bank, root, scratch, models, plan = context(project, settings)
        report(root)
        budget = float(settings.get("max_hours", 6)) * 3600
        if not 0 < budget <= 24 * 3600:
            raise ValueError("max_hours must be in (0,24]")
        started = time.monotonic()
        current_source = digest(source_manifest(project))
        for index, (name, model) in enumerate(models.items(), 1):
            owned()
            cases = plan["cases"][name]
            meta = root / name / "_meta"
            if all(complete(root, c, case_key(plan["identity"], c)) for c in cases):
                print(f"[{index}/{len(models)}] {name}: REUSE verified diagnostic records", flush=True)
                continue
            if time.monotonic() - started >= budget:
                print("Session budget reached; run again to resume.", flush=True)
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 30 * 1024**3:
                print("STOP: less than 30 GiB free scratch; saved stages are safe.", flush=True)
                break
            slot = scratch / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": plan["identity"]})
            session = str(time.time_ns())
            job = {"model": model, "project": str(project), "output": str(root), "slot": str(slot),
                   "bank": str(bank), "cases": cases, "settings": settings, "session": session,
                   "identity": plan["identity"], "source_code": current_source}
            path = meta / "jobs" / f"{session}.json"
            write_json(path, job)
            status_path = meta / "status.json"
            if status_path.exists():
                prior_status = read_json(status_path)
                freeze(meta / "status_history" / f"{digest(prior_status)}.json", prior_status)
            write_json(meta / "status.json", {"status": "running", "session": session})
            env = worker_environment(project, slot, model)
            env.update(UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
            failed = False
            try:
                for stage in ("prepare", "work"):
                    remaining = budget - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Session time budget reached")
                    python = sys.executable if stage == "prepare" else read_json(meta / "prepared.json")["python"]
                    log = meta / "logs" / f"{session}_{stage}.log"
                    print(f"[{index}/{len(models)}] {name}: {stage.upper()}", flush=True)
                    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                        "-m", "wmbench_native_diagnostics.runner", stage, "--settings", str(path)],
                        env, log, slot, min(5400, remaining),
                        {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 100, "telemetry": True},
                        assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{stage}.json", result)
                    if result["exit_code"] or result["reason"]:
                        print(_log_tail(log), flush=True)
                        raise RuntimeError(result["reason"] or f"{stage} failed; see {log}")
                    if stage == "prepare" and read_json(meta / "prepared.json")["status"] != "ready":
                        raise RuntimeError(read_json(meta / "prepared.json").get("reason", "Preparation blocked"))
                if not all(complete(root, c, case_key(plan["identity"], c)) for c in cases):
                    raise EvidenceError("Worker exited without a complete diagnostic set")
                write_json(meta / "status.json", {"status": "diagnostics_complete", "session": session})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "work", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    if args.action == "prepare":
        prepare(settings)
    elif args.action == "work":
        from .worker import work
        work(settings)
    elif args.action == "report":
        report(Path(settings["output"]))
    else:
        run(Path(__file__).resolve().parents[2], settings)


if __name__ == "__main__":
    main()
