"""Isolated generation and decoder stages; no changes to the digital run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sys
import time

from wmbench.io import read_json, write_json
from wmbench_full_payload.storage import freeze, inside
from wmbench_digital.runner import prepare
from wmbench_bank.runner import stage_environment
from .bank import context, collect, inputs
from .storage import read_result, result_path, complete


def invoke(job, action, owned):
    from wmbench.colab_canary import monitored_process
    from wmbench.embedding_bank import _log_tail
    name, root = job["model"]["id"], Path(job["results"])
    action_id = action + ("_"+job["engine"] if action in {"generate", "decode"} else "")
    path = root / name / "jobs" / f"{job['session']}_{action_id}.json"
    write_json(path, job)
    if action in {"controls", "decode"}:
        prepared = read_json(Path(job["output"]) / name / "_meta/prepared.json")
        if prepared["status"] != "ready":
            raise RuntimeError(f"Decoder preparation blocked: {prepared}")
        python, env = prepared["python"], stage_environment(job, "work")
        if name == "stegastamp":
            from stegastamp_cuda_hotfix import library_environment
            env = library_environment(env)
    else:
        python, env = sys.executable, os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    env.update(PYTHONPATH=os.pathsep.join([src, str(Path(job["project"])/"src"), env.get("PYTHONPATH", "")]),
        PYTHONUNBUFFERED="1", UV_PYTHON_INSTALL_DIR=str(Path(job["scratch"])/"python"),
        TORCH_HOME=str(Path(job["scratch"])/"quality-weights"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    log = root / name / "logs" / f"{job['session']}_{action_id}.log"
    print(f"{name} {action_id.upper()} — {log}", flush=True)
    remaining = job["deadline"]-time.time()
    if remaining <= 0:
        raise TimeoutError("Session budget reached")
    limits = dict(min_free_disk_gib=5, min_free_ram_gib=1, max_slot_gib=170, telemetry=True)
    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
        "-m", "wmbench_neural.runner", action, "--settings", str(path)], env, log, Path(job["slot"]),
        remaining, limits, assert_owned=owned)
    write_json(root / name / "timings" / f"{job['session']}_{action_id}.json", result)
    if result["exit_code"] or result["reason"]:
        print(_log_tail(log), flush=True)
        raise RuntimeError(result["reason"] or f"{action_id} exited {result['exit_code']}")


def run(project, settings):
    from wmbench.colab_canary import controller_lease
    from wmbench_full_payload.runner import cleanup
    from .report import method_report, final_report
    if os.name != "posix":
        raise RuntimeError("Use Linux/NVIDIA (Colab or a GPU server), not native Windows")
    if not 0 < settings["max_hours"] <= 168:
        raise ValueError("Session budget must be 0..168 hours")
    root = Path(settings["output"])
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        ctx = context(project, settings)
        inputs(ctx, settings.get("captions_file"))
        deadline = time.time()+settings["max_hours"]*3600
        for number, name in enumerate(ctx["selected"], 1):
            owned()
            cases = collect(ctx, ctx["models"][name])
            write_json(root / name / "attack-cases.json", [{k: c[k] for k in
                ("id", "key", "method", "image_id", "replicate", "mode", "output", "carrier_sha256", "size")} for c in cases])
            status = root / name / "status.json"
            if all(complete(read_result(result_path(root, c), c)) for c in cases):
                # Recover a crash after final result commit but before ZIP deletion.
                from collections import defaultdict
                from .spec import ATTACKS
                from .storage import manifest_path, remove_committed_package
                groups = defaultdict(list)
                for case in cases:
                    groups[case["image_id"]].append(case)
                cleanup_job = dict(results=str(root), identity=ctx["identity"], model=ctx["models"][name])
                for group in groups.values():
                    for attack in ATTACKS[1:]:
                        manifest = manifest_path(root, name, attack, group[0]["image_id"])
                        if manifest.exists() and inside(root, read_json(manifest)["archive"]).exists():
                            remove_committed_package(cleanup_job, group, attack)
                write_json(status, dict(status="complete", session="reuse"))
                method_report(root, name)
                print(f"REUSE {name}: all {len(cases)*21} results already committed", flush=True)
                continue
            if time.time() >= deadline:
                break
            ctx["scratch"].mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(ctx["scratch"]).free < settings["ssd_reserve_gib"]*1024**3:
                raise RuntimeError("Insufficient local SSD for a decoder + neural checkpoints; results preserved")
            slot = ctx["scratch"] / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": ctx["identity"]})
            job = dict(model=ctx["models"][name], project=str(project), bank=str(ctx["bank"]),
                results=str(root), output=str(root / "_runtime"), slot=str(slot), scratch=str(ctx["scratch"]),
                identity=ctx["identity"], session=str(time.time_ns()), deadline=deadline,
                settings=settings, images=ctx["images"])
            print(f"METHOD {number}/{len(ctx['selected'])}: {name}; {len(cases)} eligible carriers", flush=True)
            write_json(status, dict(status="running", session=job["session"]))
            try:
                # CPU dependency installation overlaps GPU quality measurement.
                # Model installations themselves remain separate from each other.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    preparation = pool.submit(invoke, job, "prepare", owned)
                    quality = pool.submit(invoke, job, "quality", owned) if any(
                        not (root / name / "quality" / (c["id"]+".json")).exists() for c in cases) else None
                    preparation.result()
                    if quality:
                        quality.result()
                invoke(job, "geometry", owned)
                invoke(job, "controls", owned)  # Fail fast BEFORE expensive attacks.
                for engine in ("vae", "sdxl", "ip2p", "inpaint"):
                    invoke({**job, "engine": engine}, "generate", owned)
                    invoke({**job, "engine": engine}, "decode", owned)
                counts = method_report(root, name)
                if counts["complete_carriers"] != counts["carriers"]:
                    raise RuntimeError("Incomplete technical results; all remaining packages retained")
                write_json(status, dict(status="complete", session=job["session"], bank_unchanged=True))
            except Exception as error:
                write_json(status, dict(status="error", reason=repr(error), session=job["session"]))
                method_report(root, name)
                print(f"STOP {name}: {error}. Rerun the same notebook to resume.", flush=True)
                break
            finally:
                cleanup(slot, ctx["scratch"], ctx["identity"])
                final_report(root)
        final_report(root, pairwise=settings.get("pairwise", True))
        return all((root / n / "status.json").exists() and read_json(root / n / "status.json")["status"] == "complete"
                   for n in ctx["selected"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate", "prepare", "quality", "geometry", "controls", "generate", "decode", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    project = Path(__file__).resolve().parents[2] / "production"
    if args.action == "run":
        if not run(project, settings):
            raise SystemExit(2)
    elif args.action == "validate":
        ctx = context(project, settings)
        inputs(ctx, settings.get("captions_file"))
        print(f"READY: {len(ctx['selected'])} methods, {len(ctx['images'])} images. Bank unchanged.")
    elif args.action == "prepare":
        prepare(settings)
    elif args.action == "quality":
        from wmbench_digital.quality import run as quality
        quality(settings)
    elif args.action == "generate":
        from .generate import run as generate
        generate(settings)
    elif args.action == "geometry":
        from .generate import prepare_geometry
        prepare_geometry(settings)
    elif args.action in {"controls", "decode"}:
        from .decode import controls, run as decode
        (controls if args.action == "controls" else decode)(settings)
    else:
        from .report import method_report, final_report
        root = Path(settings["output"])
        for name in read_json(root / "selection.json")["models"]:
            if (root / name / "cases.json").exists():
                method_report(root, name)
        final_report(root, pairwise=settings.get("pairwise", True))


if __name__ == "__main__":
    main()
