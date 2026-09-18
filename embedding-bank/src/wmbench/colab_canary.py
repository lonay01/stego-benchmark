"""Bounded, sequential Colab canary. No heavy library is imported by the controller.

Completed artifacts live on Drive; disposable environments live on Colab SSD.
This is an installation/round-trip test, NOT a robustness benchmark or a leaderboard.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .channels import channel_models
from .environment import code_fingerprint, prepare_model, worker_environment
from .io import atomic_bytes, digest, read_json, sha256, write_csv, write_json
from .protocol import attack_manifest, load_registry, paired_seed
from .quality import quality_model
from .runner import Experiment
from .store import JobStore

GIB = 1024**3
SCHEMA = "colab-canary-v1"
DISPOSABLE = ("environments", "assets", "sources", "cache", "tmp", "locks")
BLOCKED_CAPABILITIES = {"integration_required", "out_of_scope", "candidate", "unsupported"}
REVIEWED_CODE = "7bf73509cef53bcffb281027331f152e3497713e76ea501a04735522d3cb8903"
REVIEWED_RUN = "404e5f88f100ecc6341f"
# Historical evidence, not a claim that the new environment was exercised.
REVIEWED_COMPLETED = {
    "maskwm",
    "robust_wide",
    "pixelseal",
    "rivagan_benchmark",
    "dwt_dct_svd",
    "mbrs",
    "cin",
    "pimog",
    "sepmark",
    "trustmark_q",
    "videoseal",
    "hidden_benchmark",
    "metr",
    "stegastamp",
    "ssrh",
    "channel_vae",
    "channel_sdxl",
    "channel_ip2p",
    "quality_metrics",
}


def read_lease(path: Path, attempts: int = 5, delay: float = 0.2) -> dict:
    """Bounded retry of transient Drive/FUSE reads, never bypass ownership checks."""
    for attempt in range(attempts):
        try:
            return read_json(path)
        except (OSError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def inherited_summary(
    persistent: Path, job: dict, dataset: dict, settings: dict, identity: str
) -> dict | None:
    """Import only the explicitly reviewed release, with original hashes/provenance.

    Old artifacts are neither moved nor rewritten. Changed adapters/inputs do not
    qualify; scientific claims cannot mix these historical and current runtimes.
    """
    if not settings.get("reuse_reviewed_results", True):
        return None
    model_id = job["model"]["id"]
    if model_id not in REVIEWED_COMPLETED:
        return None
    old_root = persistent / REVIEWED_RUN
    try:
        old_settings = read_json(old_root / "settings.json")
        keys = ("image_limit", "device", "operator_probes")
        if any(settings.get(k) != old_settings.get(k) for k in keys):
            return None
        count = int(settings.get("image_limit", 1))
        old_dataset = read_json(old_root / "dataset_manifest.json")
        if old_dataset["images"][:count] != dataset["images"][:count]:
            return None
        catalog = read_json(old_root / "catalog.json")
        old_job = next(j for j in catalog if j["model"]["id"] == model_id)
        if old_job != job:
            return None
        old_identity = digest(
            {
                "schema": SCHEMA,
                "code": REVIEWED_CODE,
                "jobs": catalog,
                "settings": {k: old_settings.get(k) for k in keys},
                "images": old_dataset["images"][:count],
            }
        )
        if old_identity[:20] != REVIEWED_RUN:
            return None
        path = old_root / "models" / model_id / "status.json"
        value = reusable_summary(path, old_identity)
        if not value or value["status"] not in ("passed", "completed_with_warnings"):
            return None
        if any(
            not Path(f["path"]).resolve().is_relative_to(old_root.resolve())
            for f in value["sealed_files"]
        ):
            return None
        value.update(
            identity=identity,
            reused_from={
                "identity": old_identity,
                "status_path": str(path),
                "controller_code": REVIEWED_CODE,
                "not_retested": True,
            },
        )
        for evidence in (path, old_root / "catalog.json", old_root / "dataset_manifest.json"):
            value["sealed_files"].append({"path": str(evidence), "sha256": sha256(evidence)})
        if model_id == "metr":
            value.update(
                status="review_required",
                reason=(
                    "Historical clean-control collision: preserved, NOT accepted as detector readiness. "
                    "Needs independent negative controls; do not rerun until a favourable outcome."
                ),
            )
        return value
    except (OSError, ValueError, KeyError, StopIteration):
        return None


def checked_remove(folder: Path, slot: Path) -> None:
    """Remove only named, regenerable children of a positively identified private slot."""
    slot = slot.resolve()
    if read_json(slot / ".canary-owner.json").get("schema") != SCHEMA:
        raise RuntimeError("Missing canary ownership marker")
    if folder.name not in DISPOSABLE or folder.parent.resolve() != slot:
        raise ValueError(f"Refusing cleanup outside an owned disposable directory: {folder}")
    if folder.is_symlink() or not folder.resolve().is_relative_to(slot):
        raise ValueError(f"Refusing cleanup through a symlink: {folder}")
    if folder.exists():
        shutil.rmtree(folder)


def reusable_summary(path: Path, identity: str) -> dict | None:
    """A green status alone is insufficient: check the files sealed at completion."""
    try:
        value = read_json(path)
        if value["identity"] != identity or not value.get("finished"):
            return None
        files = value.get("sealed_files", [])
        if not files:
            return None
        if any(not Path(r["path"]).is_file() or sha256(r["path"]) != r["sha256"] for r in files):
            return None
        return value
    except (OSError, ValueError, KeyError):
        return None


def blocked_reason(model: dict, manual: Path) -> str | None:
    capability = model.get("capability", {})
    if capability.get("status") in BLOCKED_CAPABILITIES:
        return "integration_required: " + model.get("options", {}).get(
            "unavailable_reason", model.get("notes", "Adapter is not integrated")
        )
    if model.get("backend") == "matlab" and not shutil.which("matlab"):
        return "requires_matlab: licensed MATLAB/toolboxes are not installed in this Colab"
    for asset in model.get("manual_assets", []):
        path = manual / model["id"] / asset["path"]
        if not path.exists():
            return f"missing_manual_asset: {path}; author URL: {asset.get('url', '')}"
    return None


def hardware_blocked_reason(model: dict, device: str, gpu_capability: float | None) -> str | None:
    """Avoid installing 50+ GiB before an explicitly BF16-only adapter rejects T4."""
    if model["id"] in ("indexmark", "bitmark"):
        if not device.startswith("cuda") or (gpu_capability is not None and gpu_capability < 8.0):
            return "requires_ampere_bf16: native adapter requires compute capability >=8.0 (e.g. A100); no precision substitution"
    return None


def detect_gpu_capability() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return float(output.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None  # Worker still performs its strict Torch capability check.


def seal_files(folder: Path) -> list[dict]:
    return [
        {"path": str(p.resolve()), "sha256": sha256(p)}
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and p.suffix != ".lock"
        and "logs" not in p.parts
        and not p.name.endswith((".tmp", ".part"))
    ]


def summarize_model(run: Path, image_count: int, track: str) -> dict:
    """Distinguish software execution from actual payload recovery; never invent BER."""
    store = JobStore(run)
    specs = read_json(run / "planned_jobs.json")
    records = [store.valid(s, include_terminal=False) for s in specs]
    good = [r for r in records if r]
    errors = []
    warnings = []
    for spec, record in zip(specs, records):
        if record:
            continue
        raw = read_json(store.path(spec)) if store.path(spec).exists() else {}
        if spec["stage"] == "match" and raw.get("status") not in (None, "failed"):
            warnings.append(f"matched_quality: {raw.get('status')}")
        else:
            errors.append(raw.get("data", {}).get("error", f"{spec['stage']}: missing result"))
    decodes = [r for r in good if r["spec"]["stage"] == "decode"]
    native = [
        r
        for r in decodes
        if r["spec"].get("role") == "positive" and r["spec"].get("mode") == "native"
    ]
    controls = [r for r in decodes if r["spec"].get("role") == "negative_test"]
    if len(native) != image_count or len(controls) != image_count:
        errors.append("Incomplete native round trip or clean-control extraction")
    matched = [r for r in decodes if r["spec"].get("mode") == "matched42"]
    if track == "posthoc" and len(matched) != image_count:
        warnings.append("Matched42 is not fully verified")
    for r in native + matched:
        if r["data"].get("native_ber", 0) > 0:
            warnings.append(f"{r['spec']['mode']}: nonzero native BER")
    control_collision = any(r["data"].get("native_ber") == 0 for r in controls)
    if control_collision:
        warnings.append("Clean control recovered the entire payload; investigate false positives")
    if any("native_ber" not in r["data"] for r in native):
        warnings.append("Score/key detector: threshold/FPR calibration is NOT checked")
    return {
        "status": (
            "failed"
            if errors
            else "review_required"
            if control_collision
            else "completed_with_warnings"
            if warnings
            else "passed"
        ),
        "finished": not errors,
        "reason": "; ".join(dict.fromkeys(errors + warnings)),
        "native_decode_ok": len(native),
        "matched_decode_ok": len(matched),
        "clean_decode_ok": len(controls),
        "native_ber": [r["data"].get("native_ber") for r in native],
        "matched_ber": [r["data"].get("native_ber") for r in matched],
        "native_scores": [r["data"].get("score") for r in native],
        "clean_scores": [r["data"].get("score") for r in controls],
        "psnr": [r["data"].get("psnr") for r in good if r["spec"]["stage"] in ("embed", "match")],
        "job_seconds": sum(r["seconds"] for r in good),
        "peak_cuda_gib": max(
            (r["data"].get("peak_cuda_reserved_bytes", 0) for r in good), default=0
        )
        / GIB,
        "run_root": str(run),
    }


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.exists() else socket.gethostname()


@contextmanager
def controller_lease(root: Path, heartbeat_interval: float = 20):
    """Immutable ownership file + separate heartbeat; never replace a file being read.

    Drive is not a distributed locking service. This is fail-closed best-effort
    protection, not authorization to run two Colab runtimes concurrently.
    """
    import psutil

    path = root / "active_controller.json"
    token = uuid.uuid4().hex
    host, boot = socket.gethostname(), _boot_id()
    if path.exists():
        old = read_lease(path)
        heartbeat_path = root / ("heartbeat_" + old["token"] + ".json")
        same_host = old.get("host") == host and old.get("boot") == boot
        alive = same_host and psutil.pid_exists(old.get("pid", -1))
        last_seen = old.get("heartbeat", time.time())
        if heartbeat_path.exists():
            last_seen = read_lease(heartbeat_path)["heartbeat"]
        fresh_remote = not same_host and time.time() - last_seen < 180
        if alive or fresh_remote:
            raise RuntimeError(
                "Another controller is active. Stop it first; after a lost runtime wait 3 minutes."
            )
        archive = root / "old_leases"
        archive.mkdir(exist_ok=True)
        path.rename(archive / f"{time.time_ns()}.json")
    state = {
        "token": token,
        "pid": os.getpid(),
        "host": host,
        "boot": boot,
        "heartbeat": time.time(),
    }
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(state, handle)
    stopped = threading.Event()
    lease_error = []
    io_lock = threading.RLock()
    heartbeat_path = root / ("heartbeat_" + token + ".json")

    def assert_owned():
        with io_lock:
            if lease_error or read_lease(path).get("token") != token:
                raise RuntimeError("Controller lease lost; refusing simultaneous writes")

    def heartbeat():
        while not stopped.wait(heartbeat_interval):
            try:
                with io_lock:
                    assert_owned()
                    write_json(heartbeat_path, {"token": token, "heartbeat": time.time()})
            except Exception as error:
                lease_error.append(str(error))
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield assert_owned
    finally:
        stopped.set()
        thread.join(timeout=5)
        if not thread.is_alive():
            with io_lock:
                try:
                    if read_lease(path).get("token") == token:
                        path.unlink()
                        heartbeat_path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    # Leave uncertain ownership in place; never steal another lease.
                    pass


def stop_process_tree(process: subprocess.Popen) -> None:
    import psutil

    # POSIX workers are started in a private process group, including uv/gdown children.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=15)


def monitored_process(
    command: list[str],
    env: dict,
    log: Path,
    slot: Path,
    timeout: float,
    limits: dict,
    assert_owned=lambda: None,
) -> dict:
    """Bound wall time, total free SSD, available RAM, and terminate the whole process group."""
    import psutil

    started = time.monotonic()
    peak_rss = 0
    peak_slot_bytes = 0
    next_status = 0.0
    next_size = 0.0
    reason = ""
    log.parent.mkdir(parents=True, exist_ok=True)
    recorder = None
    if limits.get("telemetry", False):
        from .resource_probe import ResourceRecorder

        recorder = ResourceRecorder(log)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            while process.poll() is None:
                assert_owned()
                elapsed = time.monotonic() - started
                free_disk = shutil.disk_usage(slot).free / GIB
                if recorder:
                    from .resource_probe import memory_snapshot

                    memory = memory_snapshot()
                    free_ram = memory["effective_available_gib"]
                else:
                    free_ram = psutil.virtual_memory().available / GIB
                rss = 0
                try:
                    parent = psutil.Process(process.pid)
                    family = [parent, *parent.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in family if p.is_running())
                    peak_rss = max(peak_rss, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if recorder:
                    recorder.sample(elapsed, rss / GIB, free_disk, memory)
                if elapsed >= next_size:
                    # Only disposable dirs, never recursively scan Drive artifacts here.
                    size = 0
                    for name in DISPOSABLE:
                        for base, dirs, files in os.walk(slot / name, followlinks=False):
                            for filename in files:
                                p = Path(base) / filename
                                try:
                                    if not p.is_symlink():
                                        size += p.stat().st_size
                                except FileNotFoundError:
                                    pass
                    peak_slot_bytes = max(size, peak_slot_bytes)
                    next_size = elapsed + 30
                if free_disk < limits["min_free_disk_gib"]:
                    reason = "disk_limit: global free disk reserve reached"
                elif free_ram < limits["min_free_ram_gib"]:
                    reason = "ram_limit: available host RAM reserve reached"
                elif peak_slot_bytes / GIB > limits["max_slot_gib"]:
                    reason = "disk_limit: per-model disposable storage limit reached"
                elif elapsed > timeout:
                    reason = "timeout: stage time budget exhausted"
                if reason:
                    stop_process_tree(process)
                    break
                if elapsed >= next_status:
                    detail = ""
                    if recorder:
                        devices = recorder.latest_gpu["devices"]
                        detail = "; " + "; ".join(
                            f"{d['name']}: VRAM {d['used_gib']:.1f}/{d['total_gib']:.1f} GiB, GPU {d['utilization_percent']}%"
                            for d in devices
                        )
                    print(
                        f"  {log.stem}: {elapsed / 60:.1f} min; free SSD {free_disk:.1f} GiB; "
                        f"free RAM {free_ram:.1f} GiB; log on Drive{detail}",
                        flush=True,
                    )
                    if recorder and recorder.progress_line():
                        print("  " + recorder.progress_line(), flush=True)
                    next_status = elapsed + 45
                time.sleep(2)
        except BaseException:
            stop_process_tree(process)
            raise
        # Also reap orphaned children if the controller subprocess was killed by the OOM killer.
        if os.name == "posix":
            stop_process_tree(process)
    diagnostics = recorder.finish() if recorder else {}
    if not reason and diagnostics.get("cgroup_oom_kill_increased"):
        reason = "host_oom: cgroup oom_kill counter increased during this stage"
    return {
        **diagnostics,
        "exit_code": process.returncode,
        "reason": reason,
        "wall_seconds": time.monotonic() - started,
        "peak_process_tree_rss_gib": peak_rss / GIB,
        "peak_slot_gib": peak_slot_bytes / GIB,
    }


def _copy_lock_evidence(slot: Path, destination: Path) -> None:
    if (slot / "locks").exists():
        shutil.copytree(slot / "locks", destination / "environment", dirs_exist_ok=True)
    # PRC creates its private key during initialization. Preserve it across partial-run eviction.
    for path in (slot / "assets").rglob("prc_key_*"):
        if path.is_file():
            target = destination / "private_state" / path.relative_to(slot / "assets")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _copy_manual_assets(model: dict, source: Path, slot: Path) -> None:
    for asset in model.get("manual_assets", []):
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manual asset path escapes its model directory")
        old = source / model["id"] / relative
        new = slot / "assets" / model["id"] / relative
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.is_dir():
            shutil.copytree(old, new, dirs_exist_ok=True)
        else:
            shutil.copy2(old, new)


def child_main(job: dict, action: str) -> None:
    """Runs under the lightweight controller's Python, launching an isolated model worker."""
    project, slot, out = map(Path, (job["project"], job["slot"], job["output"]))
    model, config = job["model"], job["config"]
    try:
        if action == "prepare":
            prepared = prepare_model(project, slot, model, config["device"])
            write_json(out / "prepared_result.json", prepared)
        elif job["kind"] == "model":
            experiment = Experiment(project, config)
            run = experiment.run()
            result = summarize_model(run, config["image_limit"], model["track"])
            write_json(out / "roundtrip_result.json", result)
        else:
            prepared = read_json(out / "prepared_result.json")
            dataset = read_json(config["dataset_manifest"])
            image = dataset["images"][0]
            root = out / "operator_run" / prepared["fingerprint"][:20]
            store = JobStore(root)
            if job["kind"] == "quality":
                spec = {
                    "stage": "quality",
                    "operator": model["id"],
                    "model_id": model["id"],
                    "first_path": image["path"],
                    "second_path": image["path"],
                    "first_path_sha256": image["sha256"],
                    "second_path_sha256": image["sha256"],
                    "prompt": image["prompt"],
                }
            else:
                attack = next(a for a in attack_manifest(1) if a["id"] == job["attack_id"])
                spec = {
                    "stage": "attack",
                    "operator": model["id"],
                    "model_id": model["id"],
                    "input_path": image["path"],
                    "input_path_sha256": image["sha256"],
                    "attack": attack,
                    "prompt": image["prompt"],
                    "role": "clean_operator_probe",
                    "seed": paired_seed(image["image_id"], attack["id"], 0),
                }
            spec.update(
                {"schema": 1, "environment": prepared["fingerprint"], "image_id": image["image_id"]}
            )
            plan, worker_config = root / "plans/probe.json", root / "plans/worker.json"
            write_json(plan, [spec])
            worker = read_json(prepared["config"])
            worker["run_root"] = str(root)
            write_json(worker_config, worker)
            subprocess.run(
                [prepared["python"], "-m", "wmbench.worker", "run", str(worker_config), str(plan)],
                env=worker_environment(project, slot, model),
                check=True,
            )
            record = store.valid(spec, include_terminal=False)
            if not record:
                raw = read_json(store.path(spec)) if store.path(spec).exists() else {}
                raise RuntimeError(str(raw.get("data", "Operator did not publish a valid result")))
            write_json(
                out / "roundtrip_result.json",
                {
                    "status": "passed",
                    "finished": True,
                    "reason": "Operator-only probe; NOT watermark robustness",
                    "run_root": str(root),
                    "job_seconds": record["seconds"],
                    "peak_cuda_gib": record["data"].get("peak_cuda_reserved_bytes", 0) / GIB,
                },
            )
    except Exception as error:
        write_json(
            out / f"{action}_error.json",
            {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()},
        )
        raise


def build_canary_report(root: Path, jobs: list[dict], identity: str) -> list[dict]:
    rows = []
    for job in jobs:
        path = root / "models" / job["model"]["id"] / "status.json"
        row = read_json(path) if path.exists() else {"status": "not_attempted", "reason": ""}
        if row.get("identity") not in (None, identity):
            row = {"status": "not_attempted", "reason": "Different configuration"}
        row = {k: v for k, v in row.items() if k not in ("sealed_files", "identity")}
        rows.append(
            {
                "model_id": job["model"]["id"],
                "family": job["model"].get("family", "operator"),
                "origin_group": job["model"].get("origin_group", "operator"),
                "track": job["model"]["track"],
                "test_kind": job["kind"],
                **row,
            }
        )
    write_csv(root / "canary_report.csv", rows)
    counts = dict(Counter(r["status"] for r in rows))
    write_json(
        root / "canary_report.json",
        {
            "identity": identity,
            "counts": counts,
            "rows": rows,
            "timing_summary": {
                "historical_not_retested_seconds": sum(
                    r.get("total_seconds", 0) for r in rows if r.get("reused_from")
                ),
                "current_release_attempt_seconds": sum(
                    r.get("total_seconds", 0) for r in rows if not r.get("reused_from")
                ),
            },
            "full_experiment_completed": False,
            "quality_or_FPR_claims_allowed": False,
        },
    )
    lines = [
        "# Colab canary: technical readiness, not a scientific ranking",
        "",
        "One COCO image by default. Native and Matched42 where applicable; clean controls.",
        "AI operators are separate clean-image probes, NOT attacks on every watermark.",
        "Passed means software ran; inspect BER, matched warnings, scores and GPU runtime before scaling.",
        "One-image cold timings include initialization. They do not predict full-run steady throughput.",
        "Peak CUDA is PyTorch allocator memory; it does not cover ONNX or all driver allocations.",
        "Results are reused only with verified files; interrupted/failed models are retried next launch.",
        "reused_from identifies historical evidence and its original runtime; it was NOT retested on this GPU/code.",
        "review_required is a completed but scientifically unresolved check, never a successful detector.",
        "",
        f"Statuses: {counts}",
        "",
        "| Model | Test | Status | Detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        detail = str(row.get("reason", "")).replace("|", "/").replace("\n", " ")[:400]
        lines.append(f"| {row['model_id']} | {row['test_kind']} | {row['status']} | {detail} |")
    atomic_bytes(root / "CANARY_REPORT.md", ("\n".join(lines) + "\n").encode())
    return rows


def run_canary(project: Path, settings: dict) -> Path:
    """At most ONE environment/weight bundle and ONE heavy process at any time."""
    project = project.resolve()
    persistent = Path(settings["save_root"]).resolve()
    local = Path(settings["local_root"]).resolve()
    if local == persistent or persistent.is_relative_to(local) or local.is_relative_to(persistent):
        raise ValueError("Temporary and persistent storage must be disjoint")
    registry = load_registry(project)
    selected = list(registry) if settings.get("models", "all") == "all" else settings["models"]
    if len(selected) != len(set(selected)) or any(m not in registry for m in selected):
        raise ValueError("Unknown or duplicate model IDs")
    dataset = read_json(settings["dataset_manifest"])
    n = int(settings.get("image_limit", 1))
    if not 1 <= n <= 5 or len(dataset["images"]) < n:
        raise ValueError("Canary is limited to 1..5 images; no implicit Core100 execution")
    for image in dataset["images"][:n]:
        if sha256(image["path"]) != image["sha256"]:
            raise ValueError("Canary input PNG changed")
        if not image.get("prompt", "").strip():
            raise ValueError("Fixed COCO caption is required for generation-origin methods")
    jobs = [{"model": registry[m], "kind": "model"} for m in selected]
    if settings.get("operator_probes", True):
        for name, attack in (
            ("vae", "vae_rt_1"),
            ("sdxl", "sdxl_i2i_s030"),
            ("ip2p", "ip2p_golden_hour"),
            ("inpaint", "sdxl_inpaint_center25"),
        ):
            jobs.append(
                {
                    "model": channel_models()["channel_" + name],
                    "kind": "operator",
                    "attack_id": attack,
                }
            )
        jobs.append({"model": quality_model(), "kind": "quality"})
    scientific_settings = {k: settings.get(k) for k in ("image_limit", "device", "operator_probes")}
    identity = digest(
        {
            "schema": SCHEMA,
            "code": code_fingerprint(project),
            "jobs": jobs,
            "settings": scientific_settings,
            "images": dataset["images"][:n],
        }
    )
    root = persistent / identity[:20]
    root.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    print(
        f"Drive results: {root}\nCatalogue: {len(selected)} entries + {len(jobs) - len(selected)} operators",
        flush=True,
    )
    limits = settings["limits"]
    started = time.monotonic()
    with controller_lease(persistent) as assert_owned:
        write_json(root / "settings.json", settings)
        write_json(root / "dataset_manifest.json", dataset)
        write_json(root / "catalog.json", jobs)
        write_json(persistent / "latest_run.json", {"run_root": str(root), "identity": identity})
        gpu_capability = detect_gpu_capability()
        try:
            for index, job in enumerate(jobs, 1):
                assert_owned()
                model = job["model"]
                model_id = model["id"]
                if not model_id.replace("_", "").isalnum():
                    raise ValueError("Invalid model ID for an owned slot")
                out = root / "models" / model_id
                out.mkdir(parents=True, exist_ok=True)
                cached = reusable_summary(out / "status.json", identity)
                if cached and (
                    not cached.get("reused_from") or settings.get("reuse_reviewed_results", True)
                ):
                    print(f"[{index}/{len(jobs)}] SKIP verified: {model_id}", flush=True)
                    continue
                if cached and cached.get("reused_from"):
                    # User explicitly disabled historical reuse. Preserve the
                    # reference before replacing the per-model status with a new attempt.
                    write_json(out / "history" / f"{time.time_ns()}_historical.json", cached)
                inherited = inherited_summary(persistent, job, dataset, settings, identity)
                if inherited:
                    write_json(out / "status.json", inherited)
                    print(
                        f"[{index}/{len(jobs)}] REUSE verified historical: {model_id}", flush=True
                    )
                    build_canary_report(root, jobs, identity)
                    continue
                if time.monotonic() - started >= limits["total_hours"] * 3600:
                    print(
                        "Session time budget reached. Run the same cell again to continue.",
                        flush=True,
                    )
                    break
                manual = persistent / "manual_assets"
                reason = blocked_reason(model, manual) or hardware_blocked_reason(
                    model, settings.get("device", "cuda"), gpu_capability
                )
                requirements = settings.get("resource_requirements", {}).get(model_id)
                if requirements and not reason:
                    from .resource_probe import resource_preflight

                    admission = resource_preflight(local, requirements)
                    write_json(out / "resource_preflight.json", admission)
                    if not admission["ready"]:
                        reason = "resource_preflight: " + "; ".join(admission["reasons"])
                if reason:
                    write_json(
                        out / "status.json",
                        {
                            "identity": identity,
                            "status": "blocked",
                            "reason": reason,
                            "finished": False,
                        },
                    )
                    print(f"[{index}/{len(jobs)}] BLOCKED: {model_id}: {reason[:180]}", flush=True)
                    build_canary_report(root, jobs, identity)
                    continue
                free = shutil.disk_usage(local).free / GIB
                if free < limits["start_free_disk_gib"]:
                    print(
                        f"STOP: only {free:.1f} GiB free. Start a fresh Colab runtime; Drive results remain.",
                        flush=True,
                    )
                    break
                # Stable paths allow exact partial-job resume on the same runtime fingerprint.
                slot = local / identity[:20] / model_id
                slot.mkdir(parents=True, exist_ok=True)
                marker = slot / ".canary-owner.json"
                if marker.exists() and read_json(marker) != {
                    "schema": SCHEMA,
                    "model_id": model_id,
                }:
                    raise RuntimeError("Slot belongs to another workflow")
                write_json(marker, {"schema": SCHEMA, "model_id": model_id})
                (out / "runs").mkdir(exist_ok=True)
                # No worker is running under our exclusive controller lease. Archive interrupted claims.
                for claim in (out / "runs").glob("*/locks/*.lock"):
                    archive = out / "stale_claims"
                    archive.mkdir(exist_ok=True)
                    claim.rename(archive / f"{time.time_ns()}_{claim.name}")
                for claim in (out / "operator_run").glob("*/locks/*.lock"):
                    archive = out / "stale_claims"
                    archive.mkdir(exist_ok=True)
                    claim.rename(archive / f"{time.time_ns()}_{claim.name}")
                # Retain pinned resolution across environment eviction, but never installed markers.
                evidence = out / "environment" / model_id
                for name in ("requirements.lock", "request.json"):
                    old = evidence / name
                    target = slot / "locks" / model_id / name
                    if old.exists() and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(old, target)
                config = {
                    "profile": SCHEMA,
                    "work_root": str(slot),
                    "run_parent": str(out / "runs"),
                    "dataset_manifest": settings["dataset_manifest"],
                    "models": [model_id],
                    "image_limit": n,
                    "repeats": 1,
                    "identity_only": True,
                    "modes": ["native", "matched42"],
                    "matched_targets": [42],
                    "quality_metrics": False,
                    "device": settings.get("device", "cuda"),
                    "require_legacy_dataset": False,
                    "bootstrap_repeats": 100,
                }
                job.update(
                    {
                        "project": str(project),
                        "slot": str(slot),
                        "output": str(out),
                        "config": config,
                    }
                )
                job_path = out / "job.json"
                write_json(job_path, job)
                env = os.environ.copy()
                temp = slot / "tmp"
                temp.mkdir(exist_ok=True)
                env.update(
                    {
                        "PYTHONPATH": str(project / "src"),
                        "PYTHONUNBUFFERED": "1",
                        "UV_CACHE_DIR": str(slot / "cache/uv"),
                        "UV_PYTHON_INSTALL_DIR": str(slot / "cache/python"),
                        "PIP_CACHE_DIR": str(slot / "cache/pip"),
                        "TMPDIR": str(temp),
                        "TEMP": str(temp),
                        "TMP": str(temp),
                        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                        "HF_HUB_DISABLE_XET": "1",
                        "UV_HTTP_TIMEOUT": "120",
                        "UV_HTTP_RETRIES": "5",
                        "MPLBACKEND": "Agg",
                    }
                )
                row = {
                    "identity": identity,
                    "status": "running",
                    "finished": False,
                    "reason": "",
                    "started_unix": time.time(),
                }
                write_json(out / "status.json", row)
                print(f"[{index}/{len(jobs)}] {model_id} — one isolated model", flush=True)
                try:
                    _copy_manual_assets(model, manual, slot)
                    if (out / "private_state").exists():
                        shutil.copytree(out / "private_state", slot / "assets", dirs_exist_ok=True)
                    for action, time_key in (
                        ("prepare", "prepare_minutes"),
                        ("run", "run_minutes"),
                    ):
                        error_path = out / f"{action}_error.json"
                        error_path.unlink(missing_ok=True)
                        if action == "prepare":
                            (out / "prepared_result.json").unlink(missing_ok=True)
                        if action == "run":
                            (out / "roundtrip_result.json").unlink(missing_ok=True)
                        remaining = limits["total_hours"] * 3600 - (time.monotonic() - started)
                        stats = monitored_process(
                            [
                                sys.executable,
                                "-m",
                                "wmbench.colab_canary",
                                "child",
                                action,
                                str(job_path),
                            ],
                            env,
                            out / "logs" / f"{action}.log",
                            slot,
                            min(
                                limits.get("model_overrides", {})
                                .get(model_id, {})
                                .get(time_key, limits[time_key])
                                * 60,
                                max(0, remaining),
                            ),
                            limits,
                            assert_owned,
                        )
                        row[action + "_seconds"] = stats["wall_seconds"]
                        row["peak_process_tree_rss_gib"] = max(
                            row.get("peak_process_tree_rss_gib", 0),
                            stats["peak_process_tree_rss_gib"],
                        )
                        row["peak_slot_gib"] = max(
                            row.get("peak_slot_gib", 0), stats["peak_slot_gib"]
                        )
                        write_json(out / f"{action}_resources.json", stats)
                        if stats["exit_code"] or stats["reason"]:
                            detail = (
                                read_json(error_path).get("error", "")
                                if error_path.exists()
                                else ""
                            )
                            raise RuntimeError(
                                stats["reason"]
                                or detail
                                or f"{action} failed; inspect logs/{action}.log"
                            )
                    row.update(read_json(out / "roundtrip_result.json"))
                except KeyboardInterrupt:
                    row.update(
                        {"status": "interrupted", "reason": "Stopped by user; rerun to resume"}
                    )
                    raise
                except Exception as error:
                    row.update({"status": "failed", "reason": str(error), "finished": False})
                finally:
                    _copy_lock_evidence(slot, out)
                    if (out / "prepared_result.json").exists():
                        row["runtime"] = read_json(out / "prepared_result.json")["identity"][
                            "runtime"
                        ]
                    row["finished_unix"] = time.time()
                    row["total_seconds"] = row["finished_unix"] - row["started_unix"]
                    if row.get("finished"):
                        # Seal both scientific artifacts and environment/weight provenance before cleanup.
                        row["sealed_files"] = seal_files(Path(row["run_root"])) + seal_files(
                            out / "environment"
                        )
                    write_json(out / "status.json", row)
                    build_canary_report(root, jobs, identity)
                    # All output artifacts, extraction keys and logs are outside these directories.
                    retain = settings.get("keep_failed_slot", False) and not row.get("finished")
                    if not retain:
                        for name in DISPOSABLE:
                            checked_remove(slot / name, slot)
                    print(
                        f"  {row['status']}: {model_id}; GPU worker released; "
                        + (
                            "failed slot retained for retry on this runtime"
                            if retain
                            else "disposable SSD released"
                        ),
                        flush=True,
                    )
        finally:
            build_canary_report(root, jobs, identity)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["child"])
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    child_main(read_json(args.job), args.action)


if __name__ == "__main__":
    main()
