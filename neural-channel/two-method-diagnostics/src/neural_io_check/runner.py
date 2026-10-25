"""Separate output, explicit subprocess stages, no mutation of historical runs."""

import argparse
import os
from pathlib import Path
import sys
import time

from .common import (ATTACK_IDS, BASE_SHA, CONTROLS, METHODS, commit, digest,
                     read, separate_paths, sha, write)


def validate(project, settings):
    from wmbench_neural.bank import context, collect
    from wmbench_neural.storage import read_result, result_path

    bank, neural, root, scratch = separate_paths(*(settings[k] for k in ("bank", "neural", "output", "scratch")))
    if settings["models"] != list(METHODS) or not 1 <= settings["n_images"] <= 5:
        raise ValueError("This diagnostic covers InvisMark and ChunkySeal, 1..5 historical images")
    old_run = read(neural / "neural-run.json")
    if settings["gpu_batch_size"] != old_run["gpu_batch_size"]:
        raise ValueError("Replay batch width must equal the historical neural run")
    source_hashes = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    spec = dict(version="two-method-io-v1", base_zip_sha256=BASE_SHA,
                bank_trial_sha256=sha(bank / "trial.json"), old_neural_sha256=sha(neural / "neural-run.json"),
                n_images=settings["n_images"], methods=list(METHODS), conditions=list(CONTROLS) + list(ATTACK_IDS),
                source_hashes=source_hashes, batch_width=settings["gpu_batch_size"])
    commit(root / "diagnostic-run.json", spec)
    ctx = context(project, settings)
    if read(root / "neural-run.json") != old_run:
        raise RuntimeError("Historical neural release/protocol identity differs from the supplied capsule")
    commit(root / "_inputs/captions.json", read(neural / "_inputs/captions.json"))
    for method in METHODS:
        cases = collect(ctx, ctx["models"][method])
        historical_cases = {c["id"]: c for c in read(neural / method / "cases.json")}
        native = []
        for case in cases:
            if case["id"] not in historical_cases or case["key"] != historical_cases[case["id"]]["key"]:
                raise RuntimeError("Historical case differs; this diagnostic replays, it does not invent new cases")
            if case["mode"] == "native":
                prior = read_result(result_path(neural, case), case)
                if not prior or not all(a in prior["rows"] for a in ATTACK_IDS):
                    raise RuntimeError("The selected historical Native record is not complete")
                commit(root / method / "historical" / (case["id"] + ".json"), prior)
                native.append(case)
        if len(native) != settings["n_images"] * 5:
            raise RuntimeError("Expected five complete Native messages per image")
        commit(root / method / "native-cases.json", native)
        # Generator input is answer-blind, with exactly the old all-mode order.
        keys = ("id", "key", "method", "image_id", "replicate", "mode", "output", "carrier_sha256", "size")
        commit(root / method / "attack-cases.json", [{k: c[k] for k in keys} for c in cases])
    return ctx


def invoke(job, action, owned):
    from wmbench.colab_canary import monitored_process
    from wmbench_bank.runner import stage_environment
    from wmbench.embedding_bank import _log_tail

    root = Path(job["results"])
    stage = action + "_" + job["engine"]
    path = root / job["model"]["id"] / "jobs" / (job["session"] + "_" + stage + ".json")
    write(path, job)
    if action == "decode":
        prepared = read(Path(job["output"]) / job["model"]["id"] / "_meta/prepared.json")
        if prepared["status"] != "ready":
            raise RuntimeError(f"Decoder setup not ready: {prepared}")
        python, env = prepared["python"], stage_environment(job, "work")
    else:
        python, env = sys.executable, os.environ.copy()
    source = Path(__file__).resolve().parents[1]
    base_src = Path(job["project"]).parent / "src"
    env.update(PYTHONPATH=os.pathsep.join([str(source), str(base_src), str(Path(job["project"]) / "src")]),
               PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               UV_PYTHON_INSTALL_DIR=str(Path(job["scratch"]) / "python"))
    log = path.parent.parent / "logs" / (job["session"] + "_" + stage + ".log")
    print(f"STAGE {job['model']['id']} {stage}: {log}", flush=True)
    limits = dict(min_free_disk_gib=5, min_free_ram_gib=1, max_slot_gib=170, telemetry=True)
    remaining = job["deadline"] - time.time()
    if remaining <= 0:
        raise TimeoutError("Session budget reached")
    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
        "-m", "neural_io_check.runner", action, "--settings", str(path)], env, log, Path(job["slot"]),
        remaining, limits, assert_owned=owned)
    write(path.parent.parent / "timings" / (job["session"] + "_" + stage + ".json"), result)
    if result["exit_code"] or result["reason"]:
        print(_log_tail(log), flush=True)
        raise RuntimeError(result["reason"] or f"{stage}: exit={result['exit_code']}")


def method_complete(root, method, n):
    from .common import report_state, verify_artifacts
    rows = [read(p) for p in (root / method / "checks").glob("*/*.json")]
    for row in rows:
        verify_artifacts(root, row)
    return report_state(rows, n * 5 * (len(CONTROLS) + len(ATTACK_IDS))).startswith("IO_AGREEMENT")


def run(project, settings):
    import shutil
    from wmbench.colab_canary import controller_lease
    from wmbench_full_payload.runner import cleanup
    from wmbench_full_payload.storage import freeze
    from .report import report

    if os.name != "posix":
        raise RuntimeError("Run the GPU diagnostic in Colab or Linux/NVIDIA, not native Windows")
    if not 0 < settings["max_hours"] <= 24:
        raise ValueError("max_hours must be in (0,24]")
    # Validate disjoint destinations before creating even the controller lease.
    separate_paths(*(settings[k] for k in ("bank", "neural", "output", "scratch")))
    root = Path(settings["output"])
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root / "_reports") as owned:
        ctx = validate(project, settings)
        deadline = time.time() + settings["max_hours"] * 3600
        for method in METHODS:
            if method_complete(root, method, settings["n_images"]):
                print(f"REUSE completed {method}", flush=True)
                continue
            scratch = ctx["scratch"]
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 60 * 1024**3:
                raise RuntimeError("Need at least 60 GiB of local free SSD before setup")
            slot = scratch / "slots" / method
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": ctx["identity"]})
            job = dict(model=ctx["models"][method], project=str(project), bank=str(ctx["bank"]),
                neural=settings["neural"], results=str(root), output=str(root / "_runtime"),
                slot=str(slot), scratch=str(scratch), identity=ctx["identity"], settings=settings,
                images=ctx["images"], session=str(time.time_ns()), deadline=deadline, engine="controls")
            try:
                write(root / method / "status.json", dict(status="running", session=job["session"]))
                invoke(job, "prepare", owned)
                invoke(job, "controls", owned)
                invoke(job, "decode", owned)
                for engine in ("vae", "sdxl", "ip2p"):
                    current = {**job, "engine": engine}
                    invoke(current, "generate", owned)
                    invoke(current, "decode", owned)
                    report(root, settings)
                if not method_complete(root, method, settings["n_images"]):
                    raise RuntimeError("Diagnostics are incomplete or show an IO disagreement")
                write(root / method / "status.json", dict(status="complete", session=job["session"]))
            except Exception as error:
                write(root / method / "status.json", dict(status="error", reason=repr(error)))
                raise
            finally:
                report(root, settings)
                cleanup(slot, scratch, ctx["identity"])
        report(root, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "prepare", "controls", "generate", "decode", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.settings)
    if args.action in {"validate", "run"}:
        # The notebook supplies the exact extracted capsule path, not cwd.
        project = Path(settings["project"]) / "production"
        (validate if args.action == "validate" else run)(project, settings)
    elif args.action == "prepare":
        from wmbench_digital.runner import prepare
        prepare(settings)
    elif args.action in {"controls", "generate"}:
        from .pipeline import controls, generate
        (controls if args.action == "controls" else generate)(settings)
    elif args.action == "decode":
        from .decoder import decode
        decode(settings)
    else:
        from .report import report
        report(Path(settings["output"]), settings)


if __name__ == "__main__":
    main()
