"""One method at a time; persistent measurements, disposable attack packages."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

from wmbench.io import read_json, write_json
from wmbench_full_payload.storage import EvidenceError, freeze
from wmbench_bank.runner import prepare as production_prepare, stage_environment
from .bank import context, collect
from .storage import read_result, result_path, complete, remove_packages
from .report import method_report, final_report


def prepare(job):
    name = job["model"]["id"]
    original = Path(job["bank"]) / name / "_meta/environment"
    target = Path(job["output"]) / name / "_meta/environment"
    for filename in ("requirements.lock", "request.json"):
        if (original / filename).is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / filename, target / filename)
    production_prepare(job)


def invoke(job, stage, path, owned):
    from wmbench.colab_canary import monitored_process
    from wmbench.embedding_bank import _log_tail
    remaining = job["deadline"]-time.time()
    if remaining <= 0:
        raise TimeoutError("Session budget reached")
    meta = Path(job["output"]) / job["model"]["id"] / "_meta"
    if stage == "decode":
        prepared = read_json(meta / "prepared.json")
        if prepared["status"] != "ready":
            raise RuntimeError(f"Preparation blocked: {prepared}")
        python, env = prepared["python"], stage_environment(job, "work")
    else:
        python, env = sys.executable, os.environ.copy()
    digital_src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = digital_src+os.pathsep+str(Path(job["project"])/"src")+os.pathsep+env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["UV_PYTHON_INSTALL_DIR"] = str(Path(job["scratch"])/"python")
    if stage in {"quality", "prepare"}:
        env.update(TORCH_HOME=str(Path(job["scratch"])/"quality-weights"),
                   UV_CACHE_DIR=str(Path(job["slot"])/"cache/uv"))
    log = Path(job["results"]) / job["model"]["id"] / "logs" / f"{job['session']}_{stage}.log"
    print(f"{job['model']['id']} {stage.upper()} — log: {log}", flush=True)
    limits = {"min_free_disk_gib": 5, "min_free_ram_gib": 1, "max_slot_gib": 150, "telemetry": True}
    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
        "-m", "wmbench_digital.runner", stage, "--settings", str(path)],
        env, log, Path(job["slot"]), remaining, limits, assert_owned=owned)
    write_json(Path(job["results"]) / job["model"]["id"] / "timings" / f"{job['session']}_{stage}.json", result)
    if result["exit_code"] or result["reason"]:
        print(_log_tail(log), flush=True)
        raise RuntimeError(result["reason"] or f"{stage} exit={result['exit_code']}")


def run(project, settings):
    from wmbench.colab_canary import controller_lease
    from wmbench_full_payload.runner import cleanup
    if os.name != "posix":
        raise RuntimeError("All 29 decoders require Linux/NVIDIA. Use Colab or a Linux GPU machine.")
    if not 0 < settings["max_hours"] <= 168 or settings["ssd_reserve_gib"] < 35:
        raise ValueError("max_hours must be (0,168]; SSD reserve must be at least 35 GiB")
    root = Path(settings["output"]).resolve()
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        ctx = context(project, settings)
        deadline = time.time()+settings["max_hours"]*3600
        for index, name in enumerate(ctx["selected"], 1):
            if time.time() >= deadline:
                break
            owned()
            model = ctx["models"][name]
            print(f"METHOD {index}/{len(ctx['selected'])}: {name}; validating bank records", flush=True)
            cases = collect(ctx, model)
            status = root / name / "status.json"
            slot = ctx["scratch"] / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": ctx["identity"]})
            job = dict(model=model, project=str(project), bank=str(ctx["bank"]), results=str(root),
                       output=str(root / "_runtime"), slot=str(slot), scratch=str(ctx["scratch"]),
                       identity=ctx["identity"], session=str(time.time_ns()), deadline=deadline,
                       settings=settings, images=ctx["images"])
            path = root / name / "jobs" / (job["session"]+".json")
            write_json(path, job)
            write_json(status, {"status": "running", "session": job["session"]})
            try:
                done = all(complete(read_result(result_path(root, c), c)) for c in cases)
                if not done:
                    invoke(job, "attacks", path, owned)
                # Quality does not hold a GPU model while a decoder is installed/used.
                if any(not (root / name / "quality" / (c["id"]+".json")).exists() for c in cases):
                    invoke(job, "quality", path, owned)
                if not done:
                    if shutil.disk_usage(slot).free < 35*1024**3:
                        raise RuntimeError("Need 35 GiB free SSD for the isolated decoder; attack ZIPs are retained")
                    invoke(job, "prepare", path, owned)
                    invoke(job, "decode", path, owned)
                counts = method_report(root, name)
                if counts["complete_carriers"] != counts["carriers"]:
                    raise RuntimeError("Incomplete technical results; cache kept for resume")
                ready = root / name / "packages-ready.json"
                manifests = read_json(ready)["manifests"] if ready.exists() else []
                remove_packages(root, cases, manifests, [ctx["scratch"], root/"_temporary"], ctx["identity"], name)
                write_json(status, {"status": "complete", "session": job["session"],
                                    "temporary_attacks_removed": True, "bank_unchanged": True})
                print(f"COMPLETE {name}: tables committed; disposable attack ZIPs removed", flush=True)
            except Exception as error:
                write_json(status, {"status": "error", "reason": repr(error), "session": job["session"]})
                method_report(root, name)
                print(f"STOP {name}: {error}. Results and attack cache retained.", flush=True)
                # Do not accumulate many unfinished methods' attack caches on Drive.
                break
            finally:
                cleanup(slot, ctx["scratch"], ctx["identity"])
                final_report(root)
        final_report(root, pairwise=settings.get("pairwise", True))
        successful = all((root / name / "status.json").exists() and
                         read_json(root / name / "status.json")["status"] == "complete"
                         for name in ctx["selected"])
        print("ALL SELECTED METHODS COMPLETE" if successful else
              f"RUN INCOMPLETE: inspect {root / '_reports/REPORT.md'}; resume with the same settings", flush=True)
        return successful


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "run", "attacks", "quality", "prepare", "decode", "report"])
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    project = Path(__file__).resolve().parents[2] / "production"
    if args.action == "run":
        if not run(project, settings):
            raise SystemExit(2)
    elif args.action == "validate":
        ctx = context(project, settings)
        print(f"Validated production release, {len(ctx['selected'])} methods, {len(ctx['images'])} images; bank is read-only")
    elif args.action == "report":
        root = Path(settings["output"])
        for name in read_json(root / "selection.json")["models"]:
            if (root / name / "cases.json").exists():
                method_report(root, name)
        final_report(root, pairwise=settings.get("pairwise", True))
    elif args.action == "prepare":
        prepare(settings)
    elif args.action == "attacks":
        from .packages import prepare_packages
        prepare_packages(settings)
    elif args.action == "quality":
        from .quality import run as quality_run
        quality_run(settings)
    else:
        from .worker import decode
        decode(settings)


if __name__ == "__main__":
    main()
