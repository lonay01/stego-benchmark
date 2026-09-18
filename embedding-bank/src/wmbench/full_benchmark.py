"""Full neural-channel protocol with bounded lazy workers and durable provenance.

The notebook is a UI; this module is also the terminal entry point. No GPU
library is imported by the controller. Scientific settings do not depend on N.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from .colab_canary import (
    DISPOSABLE,
    SCHEMA,
    GIB,
    _copy_lock_evidence,
    _copy_manual_assets,
    blocked_reason,
    checked_remove,
    controller_lease,
    detect_gpu_capability,
    hardware_blocked_reason,
    monitored_process,
)
from .environment import code_fingerprint, prepare_model, worker_environment
from .io import digest, read_json, sha256, write_json
from .protocol import LEGACY_DATASET_HASH, attack_manifest, load_registry, payload_bank
from .runner import Experiment
from .store import JobStore
from .maintenance import compatible_preparation, resume_identity

PROFILE = "neural-full-v1"
FIXED = {
    "profile": PROFILE,
    "models": "all",
    "repeats": 3,
    "identity_only": False,
    "modes": ["native", "matched42"],
    "matched_targets": [42],
    "quality_metrics": True,
    "calibration_count": 1000,
    "require_legacy_dataset": True,
    "bootstrap_repeats": 2000,
}


class SessionPaused(RuntimeError):
    """An infrastructure budget was reached; committed jobs remain reusable."""


def checked_settings(settings: dict) -> dict:
    result = {**settings, **FIXED}
    n = settings.get("image_limit", 5)
    if isinstance(n, bool) or not isinstance(n, int) or not 5 <= n <= 100:
        raise ValueError("image_limit must be an integer in 5..100")
    for key, value in FIXED.items():
        if key in settings and settings[key] != value:
            raise ValueError(f"This release fixes {key}={value!r}; do not silently change protocol")
    for key in ("save_root", "local_root"):
        result[key] = str(Path(settings[key]).expanduser().resolve())
    save, local = Path(result["save_root"]), Path(result["local_root"])
    if save == local or save.is_relative_to(local) or local.is_relative_to(save):
        raise ValueError("Persistent output and disposable storage must be disjoint")
    result["image_limit"] = n
    result["work_root"] = str(local / "controller")
    result.setdefault("device", "cuda")
    result["limits"] = {
        "total_hours": 6,
        "start_free_disk_gib": 35,
        "min_free_disk_gib": 12,
        "max_slot_gib": 100,
        "min_free_ram_gib": 2,
        "telemetry": True,
        **settings.get("limits", {}),
    }
    if result["limits"]["total_hours"] <= 0:
        raise ValueError("Positive session time budget required")
    return result


def initialize_inputs(settings: dict) -> dict:
    from .portable_inputs import prepare_inputs

    config = checked_settings(settings)
    source = Path(config["source_root"]) if config.get("source_root") else None
    captions = Path(config["captions_path"]) if config.get("captions_path") else None
    save = Path(config["save_root"])
    save.mkdir(parents=True, exist_ok=True)
    with controller_lease(save):
        inputs = prepare_inputs(source, save, config["image_limit"], 1000, captions_path=captions)
        reserved = read_json(inputs["reserved_manifest"])
        config.update(
            dataset_manifest=inputs["test_manifest"],
            reserved_manifest=inputs["reserved_manifest"],
            negative_calibration_manifest=inputs["calibration_manifest"],
            calibration_reference_ids=[r["image_id"] for r in reserved["images"][:5]],
            input_provenance_path=inputs["provenance_path"],
        )
        write_json(save / "config.completed.json", config)
    return config


def execution_context() -> dict:
    """No host UUID: a reconnect on equivalent hardware can reuse exact jobs."""
    gpu = "unavailable"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    packages = {}
    for name in ("numpy", "Pillow", "psutil", "uv"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "absent"
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gpu": gpu,
        "controller_packages": packages,
    }


def workload(project: Path, config: dict) -> dict:
    registry = load_registry(project)
    posthoc = sum(m["track"] == "posthoc" for m in registry.values())
    origin = sum(m["track"] == "origin" for m in registry.values())
    n = config["image_limit"]
    modes = 2 * posthoc + origin
    return {
        "image_count": n,
        "catalog_entries": len(registry),
        "posthoc": posthoc,
        "origin": origin,
        "conditional_other": len(registry) - posthoc - origin,
        "conditions_per_image_mode": len(attack_manifest(3)),
        "positive_decodes_upper_bound": modes * n * 21,
        "calibration_decodes_upper_bound": (posthoc + origin) * 1000,
        "origin_clean_calibration_generations_upper_bound": origin * 1000,
        "ai_image_outputs_upper_bound": (modes + origin + 1) * n * 20,
        "notes": "Upper bounds for applicable tracks, not time estimates. Missing assets remain explicit; calibration is NOT multiplied by N.",
    }


class PortableExperiment(Experiment):
    """Reuse the scientific DAG; install/evict only its current operator."""

    def __init__(self, project: Path, configuration: dict):
        super().__init__(project, checked_settings(configuration))
        self.save = Path(self.config["save_root"])
        self.local = Path(self.config["local_root"])
        self.active_operator = None
        self.dependency_status = {}
        self.failed_preparation = set()
        self.assert_owned = lambda: None
        self.started = time.monotonic()
        self.gpu_capability = detect_gpu_capability()
        self.attempt = str(time.time_ns())
        self.file_hash_cache = {}
        self.calibration_cache = None
        self.amendment = None

    def _hash(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.file_hash_cache:
            self.file_hash_cache[key] = sha256(path)
        return self.file_hash_cache[key]

    def _calibration_inputs(self):
        if self.calibration_cache is None:
            self.calibration_cache = super()._calibration_inputs()
        return self.calibration_cache

    def _validate_independence(self, reserved):
        current = self.images
        self.images = reserved["images"]
        try:
            self.calibration_cache = super()._calibration_inputs()
        finally:
            self.images = current

    def _decode(self, received, units):
        # Finish calibration generation and extraction while the same origin
        # model is resident. Do not install it a third time just for extraction.
        by_model_units, by_model_received = defaultdict(list), defaultdict(list)
        for unit in units:
            by_model_units[unit["model_id"]].append(unit)
        for item in received:
            by_model_received[item["unit"]["model_id"]].append(item)
        for model_id in self.models:
            super()._decode(by_model_received[model_id], by_model_units[model_id])

    def _start(self):
        dataset = read_json(self.config["dataset_manifest"])
        reserved = read_json(self.config["reserved_manifest"])
        calibration = read_json(self.config["negative_calibration_manifest"])
        if reserved.get("legacy_semantic_hash") != LEGACY_DATASET_HASH:
            raise ValueError("Exact historical COCO100 provenance required")
        self.images = dataset["images"][: self.config["image_limit"]]
        if len(self.images) != self.config["image_limit"] or len(reserved["images"]) != 100:
            raise ValueError("Incomplete test/reserved input pool")
        if self.images != reserved["images"][: len(self.images)]:
            raise ValueError("Test set must be the stable prefix of reserved COCO100")
        if len(calibration["images"]) != 1000:
            raise ValueError("Exactly 1000 fixed independent calibration inputs required")
        for row in [*reserved["images"], *calibration["images"]]:
            if not row.get("prompt", "").strip() or sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Missing prompt or changed input: {row['path']}")
        expected_refs = [r["image_id"] for r in reserved["images"][:5]]
        if self.config["calibration_reference_ids"] != expected_refs:
            raise ValueError("Calibration references must be the first five reserved IDs")
        self._validate_independence(reserved)
        identity = {
            "settings": FIXED,
            "models": self.registry,
            "code": code_fingerprint(self.project),
            "project_root": str(self.project),
            "reserved": reserved,
            "calibration": calibration,
            "reference_ids": expected_refs,
            "execution": execution_context(),
            "device": self.config["device"],
        }
        identity, self.amendment = resume_identity(
            self.project, self.save, identity, self.config.get("resume_run_id"),
            self.config.get("runtime_migration"),
        )
        self.store = JobStore(self.save / "runs" / digest(identity)[:20])
        if self.amendment:
            write_json(
                self.store.root / "amendments" / (self.amendment["amendment"] + ".json"),
                self.amendment,
            )
            print(
                f"AUDITED RESUME: {self.store.root.name}; saved jobs remain hash-validated",
                flush=True,
            )
        self.slot_parent = self.local / self.store.root.name
        self.slot_parent.mkdir(parents=True, exist_ok=True)
        self.evidence = self.store.root / "dependencies"
        self.evidence.mkdir(exist_ok=True)
        status_path = self.store.root / "dependency_status.json"
        if status_path.exists():
            self.dependency_status = read_json(status_path)
        write_json(self.save / "latest_run.json", {"run_root": str(self.store.root)})
        if not (self.store.root / "protocol.json").exists():
            write_json(self.store.root / "protocol.json", identity)
        write_json(self.store.root / "configuration.json", self.config)
        write_json(self.store.root / "dataset_manifest.json", dataset)
        write_json(self.store.root / "attack_manifest.json", self.attacks)
        write_json(self.store.root / "payload_manifest.json", payload_bank())
        write_json(self.store.root / "model_registry.json", list(self.registry.values()))
        for operator in self.all_models:
            path = self.evidence / operator / "prepared_result.json"
            if path.exists():
                value = read_json(path)
                if value["fingerprint"] != digest(value["identity"]):
                    raise ValueError(f"Corrupted dependency provenance: {path}")
                self.prepared[operator] = value
        self.prepared["controller"] = {
            "fingerprint": digest(identity["execution"]),
            "execution_fingerprint": digest(execution_context()),
        }
        previous_plan = self.store.root / "planned_jobs.json"
        if previous_plan.exists():
            test_ids = {str(row["image_id"]) for row in self.images}
            calibration_ids = {str(row["image_id"]) for row in calibration["images"]}
            self.planned = [
                spec
                for spec in read_json(previous_plan)
                if str(spec.get("image_id"))
                in (calibration_ids if spec.get("role") == "calibration" else test_ids)
            ]
        for path in (self.store.root / "locks").glob("*.lock"):
            archive = self.store.root / "stale_claims"
            archive.mkdir(exist_ok=True)
            path.rename(archive / f"{time.time_ns()}_{path.name}")
        print(f"RESULTS: {self.store.root}", flush=True)

    def remaining(self) -> float:
        seconds = self.config["limits"]["total_hours"] * 3600 - (time.monotonic() - self.started)
        if seconds <= 0:
            raise SessionPaused("Session time budget reached; rerun to resume")
        return seconds

    def _slot(self, operator):
        if not operator.replace("_", "").isalnum():
            raise ValueError("Invalid operator ID")
        slot = self.slot_parent / operator
        slot.mkdir(exist_ok=True)
        marker = slot / ".canary-owner.json"
        value = {"schema": SCHEMA, "model_id": operator}
        if marker.exists() and read_json(marker) != value:
            raise ValueError("Disposable slot ownership mismatch")
        write_json(marker, value)
        return slot

    def _evict(self, operator):
        slot = self._slot(operator)
        out = self.evidence / operator
        out.mkdir(exist_ok=True)
        prepared_path = slot / "locks" / operator / "prepared.json"
        approved = self.prepared.get(operator)
        evidence_target = out
        if approved and prepared_path.exists():
            if read_json(prepared_path)["fingerprint"] != approved["fingerprint"]:
                evidence_target = out / "rejected_environment" / str(time.time_ns())
        _copy_lock_evidence(slot, evidence_target)
        for name in DISPOSABLE:
            checked_remove(slot / name, slot)

    def _env(self, operator, slot):
        model = self.all_models.get(operator, {"id": operator})
        env = worker_environment(self.project, slot, model)
        temp = slot / "tmp"
        temp.mkdir(exist_ok=True)
        env.update(
            UV_CACHE_DIR=str(slot / "cache/uv"),
            UV_PYTHON_INSTALL_DIR=str(slot / "cache/python"),
            PIP_CACHE_DIR=str(slot / "cache/pip"),
            TMPDIR=str(temp),
            TEMP=str(temp),
            TMP=str(temp),
            UV_HTTP_TIMEOUT="120",
            UV_HTTP_RETRIES="5",
            MAX_JOBS="2",
        )
        return env

    def _status(self, operator, status, reason=""):
        self.dependency_status[operator] = {"status": status, "reason": reason}
        write_json(self.store.root / "dependency_status.json", self.dependency_status)

    def _monitored(self, command, *args):
        guarded = [sys.executable, "-m", "wmbench.process_guard", "--", *command]
        return monitored_process(guarded, *args)

    def _ensure(self, operator) -> bool:
        if operator == "controller":
            return True
        if operator in self.failed_preparation:
            return False
        model = self.all_models[operator]
        manual = self.save / "manual_assets"
        reason = blocked_reason(model, manual) or hardware_blocked_reason(
            model, self.config["device"], self.gpu_capability
        )
        if reason:
            self._status(operator, "blocked", reason)
            self.failed_preparation.add(operator)
            print(f"BLOCKED {operator}: {reason}", flush=True)
            return False
        if self.active_operator == operator:
            return True
        # Recover owned leftovers from an interrupted controller, never user files.
        for other in self.slot_parent.iterdir():
            if (
                other.is_dir()
                and other.name != operator
                and (other / ".canary-owner.json").exists()
            ):
                self._evict(other.name)
        self.active_operator = None
        slot = self._slot(operator)
        limits = self.config["limits"]
        if shutil.disk_usage(slot).free / GIB < limits["start_free_disk_gib"]:
            raise SessionPaused("Insufficient local free disk before preparation")
        if self.config["device"].startswith("cuda"):
            from .resource_probe import resource_preflight

            requirements = {
                "chunkyseal": {
                    "ram_total_gib": 48,
                    "ram_available_gib": 32,
                    "vram_free_gib": 35,
                    "disk_free_gib": 70,
                },
                "zodiac": {
                    "ram_total_gib": 10,
                    "ram_available_gib": 8,
                    "vram_free_gib": 14,
                    "disk_free_gib": 35,
                },
                "indexmark": {
                    "ram_total_gib": 48,
                    "ram_available_gib": 24,
                    "vram_free_gib": 35,
                    "disk_free_gib": 70,
                },
                "bitmark": {
                    "ram_total_gib": 48,
                    "ram_available_gib": 24,
                    "vram_free_gib": 35,
                    "disk_free_gib": 70,
                },
            }.get(operator)
            if requirements:
                admission = resource_preflight(slot, requirements)
                write_json(self.store.root / "preflight" / f"{operator}.json", admission)
                if not admission["ready"]:
                    self._status(operator, "hardware_blocked", "; ".join(admission["reasons"]))
                    self.failed_preparation.add(operator)
                    return False
        out = self.evidence / operator
        out.mkdir(exist_ok=True)
        for name in ("requirements.lock", "request.json"):
            old = out / "environment" / operator / name
            target = slot / "locks" / operator / name
            if old.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, target)
        _copy_manual_assets(model, manual, slot)
        if (out / "private_state").exists():
            shutil.copytree(out / "private_state", slot / "assets", dirs_exist_ok=True)
        candidate = out / "attempts" / str(time.time_ns())
        candidate.mkdir(parents=True)
        job = candidate / "prepare_job.json"
        write_json(
            job,
            {
                "project": str(self.project),
                "slot": str(slot),
                "model": model,
                "device": self.config["device"],
                "out": str(candidate),
            },
        )
        print(f"PREPARE {operator} (one isolated environment)", flush=True)
        stats = self._monitored(
            [sys.executable, "-m", "wmbench.full_benchmark", "prepare-child", "--config", str(job)],
            self._env(operator, slot),
            out / "logs" / f"prepare_{self.attempt}.log",
            slot,
            min(self.remaining(), 120 * 60),
            limits,
            self.assert_owned,
        )
        write_json(out / f"prepare_resources_{self.attempt}.json", stats)
        _copy_lock_evidence(slot, candidate)
        if stats["exit_code"] or stats["reason"]:
            if operator not in self.prepared:
                # Preserve the first generated PRC key even if preparation was
                # interrupted, without replacing an already approved identity.
                _copy_lock_evidence(slot, out)
            self._status(
                operator, "failed", stats["reason"] or "Preparation failed; inspect dependency logs"
            )
            self.failed_preparation.add(operator)
            self.remaining()
            return False
        value = read_json(candidate / "prepared_result.json")
        previous = self.prepared.get(operator)
        if previous and previous["fingerprint"] != value["fingerprint"]:
            value = compatible_preparation(
                previous,
                value,
                operator,
                self.amendment,
                candidate / "compatibility_audit.json",
            )
        _copy_lock_evidence(slot, out)
        write_json(out / "prepared_result.json", value)
        self.prepared[operator] = value
        self.active_operator = operator
        self._status(operator, "ready")
        return True

    def _dispatch(self, specs):
        groups = defaultdict(list)
        for spec in specs:
            groups[spec["operator"]].append(spec)
        for operator, group in groups.items():
            self.assert_owned()
            self.remaining()
            fingerprint = self.prepared.get(operator, {}).get("fingerprint", "not_prepared")
            for spec in group:
                spec["environment"] = fingerprint
            pending = self._pending_jobs(group, operator)
            if not pending:
                if operator != "controller":
                    self._status(operator, "ready", "Verified saved jobs; no reinstall required")
                print(f"SKIP verified {operator}: {len(group)} jobs", flush=True)
                continue
            if not self._ensure(operator):
                continue
            for spec in group:
                spec["environment"] = self.prepared[operator]["fingerprint"]
            pending = list({digest(s): s for s in self._pending_jobs(group, operator)}.values())
            if not pending:
                continue
            self._save_plan()
            slot = self._slot(operator)
            group_path = (
                self.store.root / "plans" / f"{pending[0]['stage']}_{operator}_{self.attempt}.json"
            )
            write_json(group_path, pending)
            if operator == "controller":
                python, config = sys.executable, {"model": {}, "options": {}}
            else:
                python = self.prepared[operator]["python"]
                config = read_json(self.prepared[operator]["config"])
            config.update(run_root=str(self.store.root), fail_fast=True)
            config["execution_provenance"] = {
                "controller_execution": execution_context(),
                "runtime_migration": (self.amendment or {}).get("runtime_migration"),
                "controller_code": code_fingerprint(self.project),
                "amendment": self.amendment["amendment"] if self.amendment else None,
                "dependency_execution_fingerprint": self.prepared[operator].get(
                    "execution_fingerprint", self.prepared[operator]["fingerprint"]
                ),
                "dependency_compatibility_audit": self.prepared[operator].get(
                    "compatibility_audit"
                ),
            }
            config_path = group_path.with_suffix(".worker.json")
            write_json(config_path, config)
            log = (
                self.store.root / "logs" / f"{pending[0]['stage']}_{operator}_{time.time_ns()}.log"
            )
            print(f"RUN {operator}: {len(pending)} jobs; {log.name}", flush=True)
            stats = self._monitored(
                [str(python), "-m", "wmbench.worker", "run", str(config_path), str(group_path)],
                self._env(operator, slot),
                log,
                slot,
                self.remaining(),
                self.config["limits"],
                self.assert_owned,
            )
            write_json(log.with_suffix(".resources.json"), stats)
            if stats["exit_code"] or stats["reason"]:
                self._status(operator, "failed", stats["reason"] or f"Worker failure: {log.name}")
                print(f"FAILED {operator}: see {log}", flush=True)
                self.remaining()

    def _save_plan(self):
        # Retain previously materialized descendants when a resumed session ends
        # before reaching them again. Resolve placeholders by logical spec key.
        unique = {}
        for spec in self.planned:
            key = digest({k: v for k, v in spec.items() if k != "environment"})
            if key not in unique or spec.get("environment") != "not_prepared":
                unique[key] = spec
        self.planned = list(unique.values())
        write_json(
            self.store.root / "planned_jobs.json",
            self.planned,
        )

    def _pending_jobs(self, group, operator):
        pending = []
        last = time.monotonic()
        print(f"VERIFY {operator}: 0/{len(group)} saved jobs", flush=True)
        for index, spec in enumerate(group, 1):
            self.remaining()
            if not self.store.valid(spec):
                pending.append(spec)
            if index == len(group) or time.monotonic() - last >= 15:
                print(f"VERIFY {operator}: {index}/{len(group)}; pending {len(pending)}", flush=True)
                last = time.monotonic()
        return pending

    def run(self):
        self.save.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        with controller_lease(self.save) as self.assert_owned:
            self._start()
            for name in ("pause.json", "fatal_error.json"):
                old = self.store.root / name
                if old.exists():
                    history = self.store.root / "session_history" / self.attempt
                    history.mkdir(parents=True, exist_ok=True)
                    old.rename(history / name)
            completed = False
            try:
                units = self._units()
                inputs = self._marked_inputs(units)
                received = self._received(inputs)
                self._decode(received, units)
                self._quality(received, inputs)
                completed = True
            except SessionPaused as error:
                print(f"PAUSED: {error}", flush=True)
                write_json(self.store.root / "pause.json", {"reason": str(error)})
            except Exception as error:
                print(f"FATAL: {type(error).__name__}: {error}", flush=True)
                # The notebook/terminal receives the exception even if Drive
                # cannot accept the diagnostic file. Do not mask the cause.
                try:
                    write_json(
                    self.store.root / "fatal_error.json",
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "attempt": self.attempt,
                    },
                    )
                except OSError as log_error:
                    print(f"Could not save fatal_error.json: {log_error}", flush=True)
                raise
            finally:
                if completed:
                    self._save_plan()
                    from .full_report import build_full_report
                    print("REPORT: validating saved results", flush=True)
                    build_full_report(
                        self.store, self.planned, self.config, self.inventory(), self.dependency_status
                    )
                    print(f"REPORT ZIP: {export_report(self.store.root)}", flush=True)
                else:
                    print("REPORT SKIPPED: run interrupted. Saved jobs retained; no automatic Drive rescan.", flush=True)
            return self.store.root


def export_report(root: Path) -> Path:
    """Shareable tables only; never bundle cryptographic sidecars or checkpoints."""
    target = root / "final_report.zip"
    temporary = target.with_suffix(".zip.part")
    files = [root / "RUN_REPORT.md"]
    files += sorted(
        path
        for path in (root / "tables").iterdir()
        if path.is_file() and path.suffix in (".csv", ".json", ".md")
    )
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError("Cannot export report through a symlink")
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Report archive verification failed")
    os.replace(temporary, target)
    return target


def export_inputs(config: dict) -> Path:
    root = Path(config["save_root"])
    target = root / "portable_inputs.zip"
    source = root / "inputs"
    if not source.is_dir():
        raise FileNotFoundError(source)
    temporary = target.with_suffix(".zip.part")
    with controller_lease(root):
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = {}
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Input bundle cannot contain symlinks")
                if path.is_file() and not path.name.endswith((".part", ".tmp")):
                    name = path.relative_to(root).as_posix()
                    archive.write(path, name)
                    manifest[name] = {"sha256": sha256(path), "size": path.stat().st_size}
            import json

            archive.writestr(
                "INPUT_BUNDLE_MANIFEST.json",
                json.dumps({"schema": "wmbench.input_bundle.v1", "files": manifest}, indent=2),
            )
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Input bundle ZIP verification failed")
        os.replace(temporary, target)
        write_json(
            root / "portable_inputs.sha256.json", {"file": target.name, "sha256": sha256(target)}
        )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inputs", "plan", "run", "report", "export-inputs", "prepare-child")
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare-child":
        prepared = prepare_model(
            Path(config["project"]), Path(config["slot"]), config["model"], config["device"]
        )
        write_json(Path(config["out"]) / "prepared_result.json", prepared)
    elif args.command == "inputs":
        print(initialize_inputs(config)["dataset_manifest"])
    elif args.command == "plan":
        import json

        print(
            json.dumps(
                workload(args.project, checked_settings(config)), indent=2, ensure_ascii=False
            )
        )
    elif args.command == "run":
        print(PortableExperiment(args.project, config).run())
    elif args.command == "export-inputs":
        print(export_inputs(config))
    else:
        from .full_report import build_full_report

        root = Path(read_json(Path(config["save_root"]) / "latest_run.json")["run_root"])
        experiment = PortableExperiment(args.project, config)
        with controller_lease(Path(config["save_root"])):
            build_full_report(
                JobStore(root),
                read_json(root / "planned_jobs.json"),
                read_json(root / "configuration.json"),
                experiment.inventory(),
                read_json(root / "dependency_status.json"),
            )
            print(f"REPORT ZIP: {export_report(root)}")
        print(root / "RUN_REPORT.md")


if __name__ == "__main__":
    main()
