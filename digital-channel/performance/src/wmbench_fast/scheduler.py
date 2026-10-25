"""Bounded process concurrency inside ONE method's isolated environment."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import subprocess
import sys
import time

from wmbench.io import read_json, write_json
from wmbench_bank.runner import stage_environment


def shards(cases, workers):
    groups = defaultdict(list)
    for case in cases:
        groups[case["image_id"]].append(case)
    result = [[] for _ in range(min(workers, len(groups)))]
    sizes = [0] * len(result)
    for image, rows in sorted(groups.items(), key=lambda item: -len(item[1])):
        index = min(range(len(result)), key=lambda n: sizes[n])
        result[index].append(image)
        sizes[index] += len(rows)
    return result


def can_admit(sample, gpu_free_gib, ram_free_gib, active=1):
    # Warm-up is not a worst-case bound. Keep headroom and stop on any later OOM.
    used = sample.get("gpu_gib")
    return (used is not None and used > 0 and gpu_free_gib >= max(used*1.75, 2)+2
            and ram_free_gib >= max(sample["rss_gib"]*1.75, 2)+4)


def free_resources():
    import psutil
    gpu = None
    try:
        data = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                       text=True, timeout=15)
        # Single-GPU benchmark. Do not infer an aggregate multi-GPU memory pool.
        if len(data.splitlines()) == 1:
            gpu = float(data.strip()) / 1024
    except (ValueError, OSError, subprocess.SubprocessError):
        pass
    return gpu, psutil.virtual_memory().available / 1024**3


def invoke(job, stage, path, owned):
    from wmbench.colab_canary import monitored_process
    from wmbench.embedding_bank import _log_tail
    meta = Path(job["output"]) / job["model"]["id"] / "_meta"
    if stage == "decode":
        prepared = read_json(meta / "prepared.json")
        if prepared["status"] != "ready":
            raise RuntimeError(f"Environment not ready: {prepared}")
        python, env = prepared["python"], stage_environment(job, "work")
    else:
        python, env = sys.executable, os.environ.copy()
    src = Path(__file__).resolve().parents[1]
    env.update(PYTHONPATH=os.pathsep.join([str(src), str(Path(job["project"])/"src"), env.get("PYTHONPATH", "")]),
               PYTHONUNBUFFERED="1", UV_PYTHON_INSTALL_DIR=str(Path(job["scratch"])/"python"),
               WM_BENCH_TELEMETRY_PYTHON=sys.executable)
    # Keep the reviewed thread/numerical settings; do not enable TF32 or FP16.
    log = Path(job["results"])/job["model"]["id"]/"logs"/(job["session"]+"_"+stage+".log")
    print(f"{job['model']['id']} {stage} worker {job.get('worker_index', '-')}: {log}", flush=True)
    def guard():
        owned()
        if job.get("cancel_path") and Path(job["cancel_path"]).exists():
            raise RuntimeError("Sibling worker failed; stopping this subprocess")
    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
        "-m", "wmbench_fast.runner", stage, "--settings", str(path)], env, log, Path(job["slot"]),
        max(1, job["deadline"]-time.time()),
        {"min_free_disk_gib": 5, "min_free_ram_gib": 1, "max_slot_gib": 150, "telemetry": True}, assert_owned=guard)
    write_json(Path(job["results"])/job["model"]["id"]/"timings"/(job["session"]+"_"+stage+".json"), result)
    if result["exit_code"] or result["reason"]:
        print(_log_tail(log), flush=True)
        raise RuntimeError(result["reason"] or f"{stage} exit={result['exit_code']}")


def decode(job, pending, owned):
    from .worker import prepare_probes
    count = job["settings"].get("decode_workers", 2)
    groups = shards(pending, count)
    if not groups:
        return
    # Same source image is never processed by two cache writers.
    session_root = Path(job["slot"])/"tmp"/job["session"]
    reference = session_root/"parallel-probes/reference.json"
    cancel = session_root/"parallel-probes/cancel.json"
    probe_inputs = prepare_probes(job, pending[0])
    plans = []
    for index, images in enumerate(groups):
        worker = dict(job, session=job["session"]+f"_w{index}", worker_index=index, shard_images=images,
                      probe_case=pending[0], probe_inputs=probe_inputs, probe_reference=str(reference),
                      ready_path=str(session_root/f"ready-{index}.json"), cancel_path=str(cancel))
        path = Path(job["results"])/job["model"]["id"]/"jobs"/(worker["session"]+".json")
        write_json(path, worker)
        plans.append((worker, path))
    live, next_index, sample = {}, 0, None
    started, peak_concurrent, succeeded = time.monotonic(), 1, False
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        try:
            while live or next_index < len(plans):
                owned()
                if time.time() >= job["deadline"]:
                    raise TimeoutError("Session budget reached")
                if not live:
                    current, path = plans[next_index]
                    live[pool.submit(invoke, current, "decode", path, owned)] = next_index
                    next_index += 1
                first_ready = Path(plans[0][0]["ready_path"])
                if sample is None and first_ready.exists():
                    sample = read_json(first_ready)
                # Wait until every running worker has passed its warm-up before admitting another.
                if next_index < len(plans) and sample is not None and all(
                    Path(plans[i][0]["ready_path"]).exists() for i in live.values()):
                    gpu, ram = free_resources()
                    if gpu is not None and can_admit(sample, gpu, ram):
                        current, path = plans[next_index]
                        print(f"ADMIT worker {next_index}: free VRAM {gpu:.1f} GiB, RAM {ram:.1f} GiB", flush=True)
                        live[pool.submit(invoke, current, "decode", path, owned)] = next_index
                        peak_concurrent = max(peak_concurrent, len(live))
                        next_index += 1
                done, _ = wait(live, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    del live[future]
            succeeded = True
        except BaseException:
            write_json(cancel, {"time": time.time(), "reason": "controller/worker failure"})
            # monitored_process checks ownership/guard and kills child trees.
            raise
        finally:
            write_json(Path(job["results"])/job["model"]["id"]/"timings"/(job["session"]+"_parallel.json"),
                       dict(requested_workers=count, shards=len(groups), peak_concurrent=peak_concurrent,
                            pending_carriers_at_start=len(pending), wall_seconds=time.monotonic()-started,
                            completed=succeeded, warmup_sample=sample,
                            note="Includes safety probes; per-call latency under contention is not isolated latency"))
