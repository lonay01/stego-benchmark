"""Lightweight host/GPU diagnostics; never imports Torch in the controller.

All GPU numbers are whole-device, sampled values (not exact allocator peaks).
RAM budgets honour readable cgroup limits in addition to host MemAvailable.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import time

GIB = 1024**3


def cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> list[dict]:
    candidates = {root}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" or "memory" in controllers.split(","):
                base = root if hierarchy == "0" else root / "memory"
                target = (base / relative.lstrip("/")).resolve()
                if target.is_relative_to(base.resolve()):
                    candidates.update(
                        p for p in (target, *target.parents) if p.is_relative_to(base.resolve())
                    )
    except (OSError, ValueError):
        pass
    candidates.add(root / "memory")
    rows = []
    for path in sorted(candidates):
        for current, limit, events in (
            ("memory.current", "memory.max", "memory.events"),
            ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt"),
        ):
            try:
                usage = int((path / current).read_text().strip())
                raw = (path / limit).read_text().strip()
                ceiling = None if raw == "max" else int(raw)
                if ceiling is not None and ceiling >= 2**60:
                    ceiling = None
            except (OSError, ValueError):
                continue
            counters = {}
            try:
                for line in (path / events).read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        counters[parts[0]] = int(parts[1])
                    elif len(parts) == 1:
                        counters["failcnt"] = int(parts[0])
            except (OSError, ValueError):
                pass
            rows.append({"path": str(path), "usage": usage, "limit": ceiling, "events": counters})
            break
    return rows


def memory_snapshot() -> dict:
    import psutil

    ram = psutil.virtual_memory()
    groups = cgroup_memory()
    totals = [ram.total] + [g["limit"] for g in groups if g["limit"] is not None]
    available = [ram.available] + [
        max(0, g["limit"] - g["usage"]) for g in groups if g["limit"] is not None
    ]
    return {
        "host_total_gib": ram.total / GIB,
        "host_available_gib": ram.available / GIB,
        "effective_total_gib": min(totals) / GIB,
        "effective_available_gib": min(available) / GIB,
        "cgroups": groups,
    }


def gpu_snapshot() -> dict:
    fields = "index,name,memory.total,memory.used,utilization.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        devices = []
        for values in csv.reader(io.StringIO(result.stdout)):
            if len(values) != 5:
                continue
            index, name, total, used, util = (v.strip() for v in values)
            devices.append(
                {
                    "index": int(index),
                    "name": name,
                    "total_gib": float(total) / 1024,
                    "used_gib": float(used) / 1024,
                    "utilization_percent": float(util)
                    if util.replace(".", "", 1).isdigit()
                    else None,
                }
            )
        return {"devices": devices}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"devices": [], "error": str(error)}


def resource_preflight(local: Path, requirements: dict) -> dict:
    """Conservative admission policy, NOT a measured minimum for these models."""
    memory, gpu = memory_snapshot(), gpu_snapshot()
    disk = shutil.disk_usage(local).free / GIB
    reasons = []
    if memory["effective_total_gib"] < requirements.get("ram_total_gib", 0):
        reasons.append(
            f"system RAM {memory['effective_total_gib']:.1f} GiB < {requirements['ram_total_gib']} GiB"
        )
    if memory["effective_available_gib"] < requirements.get("ram_available_gib", 0):
        reasons.append(
            f"available RAM {memory['effective_available_gib']:.1f} GiB < {requirements['ram_available_gib']} GiB"
        )
    if disk < requirements.get("disk_free_gib", 0):
        reasons.append(f"free SSD {disk:.1f} GiB < {requirements['disk_free_gib']} GiB")
    # The diagnostic notebook admits exactly one visible device: no GPU pooling.
    if len(gpu["devices"]) != 1:
        reasons.append("diagnostic requires exactly one NVIDIA GPU visible to nvidia-smi")
    elif gpu["devices"][0]["total_gib"] - gpu["devices"][0]["used_gib"] < requirements.get(
        "vram_free_gib", 0
    ):
        reasons.append(f"free GPU VRAM below {requirements['vram_free_gib']} GiB")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "requirements": requirements,
        "memory": memory,
        "gpu": gpu,
        "free_disk_gib": disk,
        "unix": time.time(),
    }


def oom_events_increased(before: list[dict], after: list[dict]) -> bool:
    old = {g["path"]: g["events"] for g in before}
    return any(
        g["path"] in old and g["events"].get("oom_kill", 0) > old[g["path"]].get("oom_kill", 0)
        for g in after
    )


class ResourceRecorder:
    """Append bounded telemetry every ten seconds, preserving previous attempts."""

    def __init__(self, log: Path):
        self.path = log.with_name(log.stem + "_telemetry.jsonl")
        self.attempt = time.time_ns()
        self.before = memory_snapshot()["cgroups"]
        self.peak_vram_gib = 0.0
        self.next_sample = 0.0
        self.latest_gpu = {"devices": []}
        self.log_offset = log.stat().st_size if log.exists() else 0
        self.log = log

    def sample(self, elapsed: float, rss: float, free_disk: float, memory: dict) -> None:
        if elapsed < self.next_sample:
            return
        self.latest_gpu = gpu_snapshot()
        self.peak_vram_gib = max(
            [self.peak_vram_gib] + [d["used_gib"] for d in self.latest_gpu["devices"]]
        )
        row = {
            "attempt": self.attempt,
            "unix": time.time(),
            "elapsed_seconds": elapsed,
            "process_tree_rss_gib": rss,
            "free_disk_gib": free_disk,
            "memory": memory,
            "gpu": self.latest_gpu,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.next_sample = elapsed + 10

    def progress_line(self) -> str:
        """Bounded tail, never read a multi-GB log into memory."""
        try:
            with self.log.open("rb") as stream:
                stream.seek(max(self.log_offset, self.log.stat().st_size - 8192))
                lines = stream.read(8192).decode("utf-8", errors="replace").splitlines()
            useful = [
                line for line in lines if line.startswith(("WM_DIAG ", "ZODIAC ", "Download "))
            ]
            return useful[-1][:400] if useful else ""
        except OSError:
            return ""

    def finish(self) -> dict:
        after = memory_snapshot()["cgroups"]
        return {
            "sampled_device_vram_peak_gib": self.peak_vram_gib,
            "cgroup_oom_kill_increased": oom_events_increased(self.before, after),
            "cgroup_before": self.before,
            "cgroup_after": after,
            "telemetry": str(self.path),
            "attempt": self.attempt,
        }
