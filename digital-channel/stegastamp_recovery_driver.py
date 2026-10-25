"""One-time, byte-exact repair of the four known empty StegaStamp carriers."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from wmbench.io import read_json, write_json
from wmbench_digital.runner import prepare
from wmbench_bank.runner import stage_environment
from stegastamp_cuda_hotfix import library_environment
from stegastamp_byte_recovery import TARGETS, collect_evidence, sha, commit


def run(project, bank, work, scratch, results):
    from wmbench.colab_canary import controller_lease
    from wmbench_fast.launch import alive
    launch = results / "_reports/fast_launcher.json"
    if launch.exists() and alive(read_json(launch)):
        raise RuntimeError("Stop the active digital runner before restoring bank files")
    for other in (bank, results):
        if work.is_relative_to(other) or other.is_relative_to(work):
            raise ValueError("Recovery evidence must be separate from bank and digital results")
    with controller_lease(results / "_reports") as owned:
        collect_evidence(bank)
        if all(sha(bank / f"stegastamp/native/{s}.png") == h for s, h in TARGETS.items()):
            print("StegaStamp: all four original PNG hashes already match; no repair needed", flush=True)
            return
        work.mkdir(parents=True, exist_ok=True)
        slot = scratch / "slots/stegastamp"
        slot.mkdir(parents=True, exist_ok=True)
        model = read_json(bank / "stegastamp/_meta/model_identity.json")["registry"]
        # Preserve registry and dependency lock; only select the original Python
        # patch release for the fresh repair environment before normal prepare().
        python = slot / "environments/stegastamp/bin/python"
        if not python.exists():
            uv = shutil.which("uv")
            if not uv:
                raise RuntimeError("Run the notebook controller setup first")
            subprocess.run([uv, "venv", "--python", "3.10.18", str(python.parent.parent)], check=True)
        stamp = str(time.time_ns())
        job = dict(model=model, bank=str(bank), project=str(project), output=str(work / "runtime"),
                   slot=str(slot), scratch=str(scratch), results=str(results), session=stamp,
                   deadline=time.time() + 3600, settings=dict(device="cuda", download_author_assets=True))
        path = work / (stamp + "_job.json")
        write_json(path, job)
        # CUDA is linked only in these StegaStamp subprocesses. No locked package
        # or unrelated model environment is changed.
        env = library_environment(stage_environment(job, "work"))
        src = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join([src, str(project / "src"), env.get("PYTHONPATH", "")])
        # prepare() itself spawns model workers; supply the same linked libraries.
        previous = {key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "WM_STEGA_CUDA_LIBRARIES")}
        try:
            os.environ.update({key: env[key] for key in previous})
            prepare(job)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        owned()
        prepared = read_json(work / "runtime/stegastamp/_meta/prepared.json")
        if prepared["status"] != "ready":
            raise RuntimeError(f"Recovery environment not ready: {prepared}")
        candidate_dir = work / ("candidates_" + stamp)
        subprocess.run([prepared["python"], "-u", "-m", "stegastamp_byte_recovery", "candidates",
                        "--job", str(path), "--work", str(candidate_dir)], env=env, check=True, timeout=1800)
        owned()
        # Commit validates the entire set first; mismatches leave bank unchanged.
        commit(candidate_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "bank", "work", "scratch", "results"):
        parser.add_argument("--" + key, required=True, type=lambda p: Path(p).resolve())
    run(**vars(parser.parse_args()))
