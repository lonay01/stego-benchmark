"""Single-model scheduling, isolated environments, durable per-mode checkpoints."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

from wmbench.embedding_bank import freeze_inputs, _log_tail
from wmbench.environment import worker_environment
from wmbench.io import atomic_bytes, digest, read_json, sha256, write_json
from wmbench_full_payload.runner import prepare as registry_prepare, cleanup
from wmbench_full_payload.storage import EvidenceError, disjoint, freeze, inside

from .contracts import POLICY, capacity, models_for_run
from .report import report
from .worker import complete


def context(project, settings):
    source, output, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(source, output, scratch)
    n = settings.get("n_images", 5)
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("N_IMAGES must be an integer from 1 to 100")
    if (output / "bank.json").exists():
        raise ValueError("Use a new output folder, not a historical bank")
    if not (source / "_inputs/manifest.json").is_file():
        raise FileNotFoundError(f"Need frozen historical _inputs/manifest.json under {source}")
    models = models_for_run(project)
    selected = list(models) if settings.get("models", "all") == "all" else settings["models"]
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or set(selected)-models.keys():
        raise ValueError("Select unique registered method IDs or 'all'")
    data = freeze_inputs(Path("__unused__"), source)
    # The release file inventory includes Python, author locks and registry.
    release = read_json(project / "release.json")
    for name, expected in release["files"].items():
        if sha256(inside(project, name)) != expected:
            raise EvidenceError(f"Release file changed: {name}")
    identity = {"policy": POLICY, "models": models, "dataset": data["content_hash"],
                "release": digest(release)}
    freeze(output / "trial.json", identity)
    for image in data["images"]:
        origin, target = inside(source, image["path"]), inside(output, image["path"])
        if target.exists() and sha256(target) != image["sha256"]:
            raise EvidenceError(f"Stored source changed: {target}")
        if not target.exists():
            atomic_bytes(target, origin.read_bytes())
    freeze(output / "_inputs/manifest.json", data)
    csv = source / "_inputs/legacy_manifest.csv"
    target = output / "_inputs/legacy_manifest.csv"
    if target.exists() and sha256(target) != sha256(csv):
        raise EvidenceError("Stored source CSV changed")
    if not target.exists():
        atomic_bytes(target, csv.read_bytes())
    freeze(output / "_inputs/contracts.json", {k: capacity(v) for k, v in models.items()})
    return {"output": output, "scratch": scratch, "bank": source, "selected": selected,
            "models": models, "images": data["images"][:n], "identity": digest(identity)}


def prepare(job):
    model, project, root, slot = job["model"], Path(job["project"]), Path(job["output"]), Path(job["slot"])
    if model["engine"] != "author":
        if model["id"] == "editguard":
            from .build_constraints import constrain_build_dependencies, dependency_diff
            snapshot = project / "locks/editguard-installed.txt"
            with constrain_build_dependencies(snapshot, model["python"]):
                registry_prepare(job)
            prepared = read_json(root / model["id"] / "_meta/prepared.json")
            if prepared["status"] != "blocked":
                actual = Path(prepared["config"]).parent / "installed.freeze.txt"
                difference = dependency_diff(snapshot.read_text(), actual.read_text())
                if difference:
                    raise EvidenceError(f"EditGuard dependencies differ from reviewed run: {difference}")
        else:
            registry_prepare(job)
        return
    from authornight.environment import prepare as author_prepare
    python, source, _ = author_prepare(model["author_name"], project / "authors", root / "_author", slot)
    write_json(root / model["id"] / "_meta/prepared.json",
               {"status": "ready", "python": str(python), "source": str(source), "engine": "author"})


def stage_environment(job, stage):
    project, slot, root, model = (Path(job["project"]), Path(job["slot"]), Path(job["output"]), job["model"])
    env = worker_environment(project, slot, model)
    if model["engine"] == "author" and stage == "work":
        from authornight.environment import prepared_environment
        prepared = read_json(root / model["id"] / "_meta/prepared.json")
        env = prepared_environment(model["author_name"], project / "authors", root / "_author", slot,
                                   Path(prepared["source"]))
        env["PYTHONPATH"] = str(project / "src") + os.pathsep + env["PYTHONPATH"]
    env.update(UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(Path(job["scratch"]) / "python"))
    return env


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process
    if os.name == "nt":
        raise RuntimeError("The complete 29-method release requires Linux + NVIDIA. Use Colab, not native Windows.")
    budget = float(settings.get("max_hours", 16))*3600
    if not 0 < budget <= 168*3600:
        raise ValueError("max_hours must be in (0, 168]")
    root = Path(settings["output"]).resolve()
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        ctx = context(project, settings)
        root, scratch = ctx["output"], ctx["scratch"]
        deadline = time.time()+budget
        limits = {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 150, "telemetry": True}
        report(root, settings)
        for index, name in enumerate(ctx["selected"], 1):
            owned()
            model, meta = ctx["models"][name], root / name / "_meta"
            print(f"[{index}/{len(ctx['selected'])}] {name}: verify saved task records", flush=True)
            if complete(root, model, ctx["images"]):
                print("REUSE: no embedding or decoding required", flush=True)
                report(root, settings, refresh_methods=[name])
                continue
            if time.time() >= deadline:
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 35*1024**3:
                print("STOP: need 35 GiB free temporary SSD; durable results retained", flush=True)
                break
            slot = scratch / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": ctx["identity"]})
            session = str(time.time_ns())
            job = {"model": model, "project": str(project), "output": str(root), "slot": str(slot),
                "scratch": str(scratch), "bank": str(ctx["bank"]), "images": ctx["images"],
                "settings": settings, "session": session, "deadline": deadline}
            path = meta / "jobs" / f"{session}.json"
            write_json(path, job)
            state = meta / "status.json"
            write_json(state, {"status": "running", "session": session})
            try:
                for stage in ("prepare", "work"):
                    remaining = deadline-time.time()
                    if remaining <= 0:
                        raise TimeoutError("Session budget reached")
                    owned()
                    python = sys.executable if stage == "prepare" else read_json(meta / "prepared.json")["python"]
                    log = meta / "logs" / f"{session}_{stage}.log"
                    print(f"{name} {stage.upper()}: {log}", flush=True)
                    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                        "-m", "wmbench_bank.runner", stage, "--settings", str(path)],
                        stage_environment(job, stage), log, slot, remaining, limits, assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{stage}.json", result)
                    if result["exit_code"] or result["reason"]:
                        print(_log_tail(log), flush=True)
                        raise RuntimeError(result["reason"] or f"{stage} exit={result['exit_code']}")
                    if stage == "prepare" and read_json(meta / "prepared.json")["status"] == "blocked":
                        write_json(state, read_json(meta / "prepared.json"))
                        break
                else:
                    done = complete(root, model, ctx["images"])
                    write_json(state, {"status": "complete" if done else "incomplete", "session": session})
            except EvidenceError:
                raise
            except Exception as error:
                write_json(state, {"status": "error", "reason": str(error), "session": session})
                print(f"ERROR {name}: {error}. Other methods can continue.", flush=True)
            finally:
                locks = slot / "locks" / name
                if locks.exists():
                    shutil.copytree(locks, meta / "environment", dirs_exist_ok=True)
                report(root, settings, refresh_methods=[name])
                if settings.get("cleanup", True):
                    # Only owned installations/caches; logs, PNGs, keys, author
                    # download receipts and model identity remain on Drive.
                    cleanup(slot, scratch, ctx["identity"])
        report(root, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "work", "report", "reindex", "validate"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    project = Path(__file__).resolve().parents[2]
    if args.action == "prepare":
        prepare(settings)
    elif args.action == "work":
        from .worker import work
        work(settings)
    elif args.action == "report":
        report(settings["output"], settings)
    elif args.action == "reindex":
        # Explicit recovery after a hard runtime loss before index flush.
        # Read task records only: no GPU inference and no weight hashing.
        identity = read_json(Path(settings["output"]) / "trial.json")
        report(settings["output"], settings, refresh_methods=list(identity["models"]))
    elif args.action == "validate":
        ctx = context(project, settings)
        print(f"Validated {len(ctx['models'])} configurations; selected {len(ctx['images'])} of 100 frozen images")
    else:
        run(project, settings)


if __name__ == "__main__":
    main()
