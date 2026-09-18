"""One Python environment per adapter; immutable resolved locks after first preparation."""

from __future__ import annotations

import os
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .io import atomic_bytes, digest, download, read_json, safe_extract, sha256, write_json


def command(args: list, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def env_python(folder: Path) -> Path:
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def code_fingerprint(project: Path) -> str:
    return digest(
        {str(p.relative_to(project)): sha256(p) for p in sorted((project / "src").rglob("*.py"))}
    )


def source_checkout(source: dict, root: Path) -> Path:
    revision = source.get("revision", source.get("commit"))
    if not revision or len(revision) != 40:
        raise ValueError(f"A full pinned Git revision is required: {source}")
    target = root / "sources" / digest({"url": source["url"], "revision": revision})[:20]
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--filter=blob:none", "--no-checkout", source["url"], target])
        command(["git", "checkout", "--detach", revision], cwd=target)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if actual != revision:
        raise RuntimeError(f"Wrong source revision in {target}: {actual}")
    if source.get("submodules"):
        paths = source["submodules"] if isinstance(source["submodules"], list) else []
        command(["git", "submodule", "update", "--init", "--recursive", "--", *paths], cwd=target)
    return target


def worker_environment(project: Path, work: Path, model: dict) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src")
    asset_root = work / "assets" / model["id"]
    env.update(
        {
            "HF_HOME": str(asset_root / "huggingface"),
            "TORCH_HOME": str(asset_root / "torch"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(model.get("cpu_threads", 4)),
            "MKL_NUM_THREADS": str(model.get("cpu_threads", 4)),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            # Colab's inline backend is not installed in isolated model venvs.
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(work / "cache" / "matplotlib"),
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    return env


def resolve_options(model: dict, source: Path, asset_root: Path, device: str) -> dict:
    options = dict(model.get("options", {}))
    options.update(
        {
            "model_id": model["id"],
            "repo_root": str(source),
            "asset_root": str(asset_root),
            "device": device,
        }
    )

    def local(value):
        if isinstance(value, dict):
            return {key: local(path) for key, path in value.items()}
        if isinstance(value, str) and not Path(value).is_absolute():
            return str(asset_root / value)
        return value

    if "weights" in options:
        options["weights"] = local(options["weights"])
    return options


def prepare_model(project: Path, work: Path, model: dict, device: str = "cuda") -> dict:
    """Do not run experiments until the source, complete env lock and assets are recorded."""
    work = work.resolve()
    model_id = model["id"]
    capability = model.get("capability", {})
    status = capability.get("status", model.get("status", "ready_unverified"))
    if status in ("integration_required", "out_of_scope", "candidate", "unsupported"):
        raise RuntimeError(
            f"{model_id}: {status}: {capability.get('reason', model.get('notes', ''))}"
        )
    folder = work / "environments" / model_id
    locks = work / "locks" / model_id
    locks.mkdir(parents=True, exist_ok=True)
    asset_root = work / "assets" / model_id
    asset_root.mkdir(parents=True, exist_ok=True)
    for asset in model.get("manual_assets", []):
        if not (asset_root / asset["path"]).exists():
            raise FileNotFoundError(
                f"Required author asset: {asset_root / asset['path']}; download: {asset.get('url', '')}"
            )
    # uv is controller tooling, never installed into an arbitrary model environment.
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install controller tooling: python -m pip install uv==0.8.15")
    source = source_checkout(model["source"], work) if model.get("source") else project
    python = env_python(folder)
    import re

    requirements = list(model.get("requirements", []))
    names = {re.split(r"[<>=!~\[ @]", r, 1)[0].lower().replace("_", "-") for r in requirements}
    defaults = ["numpy==1.26.4", "Pillow==10.4.0"]
    if any(a.get("kind", a.get("type")) == "hf_snapshot" for a in model.get("assets", [])):
        defaults.append("huggingface-hub==0.34.4")
    for requirement in defaults:
        if requirement.split("==")[0].lower() not in names:
            requirements.append(requirement)
    request_hash = digest(
        {"python": model.get("python", "3.10"), "requirements": requirements, "registry": model}
    )
    request_path = locks / "request.json"
    if request_path.exists() and read_json(request_path)["sha256"] != request_hash:
        raise RuntimeError(
            f"Environment configuration changed for {model_id}. Use a new WORK_ROOT."
        )
    write_json(request_path, {"sha256": request_hash, "requirements": requirements})
    input_path, lock_path = locks / "requirements.in", locks / "requirements.lock"
    atomic_bytes(input_path, ("\n".join(requirements) + "\n").encode())
    if not python.exists():
        command([uv, "venv", "--python", model.get("python", "3.10"), folder])
    if not lock_path.exists():
        validated = project / model.get("validated_lock", "__no_validated_lock__")
        if (
            validated.is_file()
            and sys.platform == "linux"
            and platform.machine().lower() in ("x86_64", "amd64")
        ):
            shutil.copyfile(validated, lock_path)
        else:
            command([uv, "pip", "compile", "--python", python, input_path, "-o", lock_path])
    installed = locks / "installed.json"
    if not installed.exists() or read_json(installed).get("lock") != sha256(lock_path):
        command([uv, "pip", "sync", "--python", python, lock_path])
        for requirement in model.get("build_requirements", []):
            command([uv, "pip", "install", "--python", python, "--no-build-isolation", requirement])
        write_json(installed, {"lock": sha256(lock_path)})
    env = worker_environment(project, work, model)
    for asset in model.get("assets", []):
        kind = asset.get("kind", asset.get("type", "file"))
        target = asset_root / asset["path"]
        if kind == "hf_snapshot":
            if not asset.get("revision") or len(asset["revision"]) != 40:
                raise ValueError(f"Pin the Hugging Face snapshot: {asset}")
            if not (target / ".wmbench_snapshot.json").exists():
                command(
                    [
                        python,
                        "-m",
                        "wmbench.worker",
                        "download-hf",
                        asset["repo_id"],
                        asset["revision"],
                        target,
                        json.dumps(
                            asset.get(
                                "ignore_patterns",
                                [
                                    "*.ckpt",
                                    "*.onnx",
                                    "*.msgpack",
                                    "*.h5",
                                    "*.ot",
                                ],
                            )
                        ),
                    ],
                    env=env,
                )
                write_json(target / ".wmbench_snapshot.json", asset)
        elif kind in ("manual", "local"):
            if not target.exists():
                raise FileNotFoundError(f"Manual asset needed: {target}; {asset.get('url', '')}")
        elif kind in ("gdrive", "google_drive"):
            if not target.exists():
                command([python, "-m", "gdown", "--fuzzy", asset["url"], "-O", target], env=env)
        else:
            download(asset["url"], target, asset.get("sha256"))
            if kind == "archive":
                marker = target.with_suffix(target.suffix + ".extracted.json")
                if not marker.exists():
                    safe_extract(target, asset_root / asset.get("extract_to", "."))
                    write_json(marker, {"sha256": sha256(target)})
    options = resolve_options(model, source, asset_root, device)
    config = {"model": model, "options": options, "work": str(work), "project": str(project)}
    config_path = locks / "worker_config.json"
    write_json(config_path, config)
    init_marker = locks / "initialized.json"
    if not init_marker.exists():
        command([python, "-m", "wmbench.worker", "initialize", config_path, init_marker], env=env)
    runtime_path = locks / "current_runtime.json"
    command([python, "-m", "wmbench.worker", "runtime", runtime_path], env=env)
    # Full weight hashes are computed once, then reused only while size/mtime are unchanged.
    inventory_path = locks / "asset_inventory.json"
    old = read_json(inventory_path) if inventory_path.exists() else {}
    inventory = {}
    for path in sorted(asset_root.rglob("*")):
        if (
            not path.is_file()
            or path.name.endswith((".lock", ".part"))
            or ".cache" in path.parts
            or "quality_features" in path.parts
        ):
            continue
        name = str(path.relative_to(asset_root))
        stat = path.stat()
        prior = old.get(name, {})
        file_hash = (
            prior.get("sha256")
            if prior.get("size") == stat.st_size and prior.get("mtime_ns") == stat.st_mtime_ns
            else sha256(path)
        )
        inventory[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_hash}
    write_json(inventory_path, inventory)
    freeze = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], text=True)
    atomic_bytes(locks / "installed.freeze.txt", freeze.encode())
    identity = {
        "registry": model,
        "lock_sha256": sha256(lock_path),
        "installed_sha256": sha256(locks / "installed.freeze.txt"),
        "weights": {k: v["sha256"] for k, v in inventory.items()},
        "controller_code": code_fingerprint(project),
        "runtime": read_json(runtime_path),
    }
    prepared = {
        "identity": identity,
        "fingerprint": digest(identity),
        "python": str(python),
        "config": str(config_path),
        "model_id": model_id,
    }
    prepared["worker_config_sha256"] = sha256(config_path)
    prepared["source_path"] = str(source)
    write_json(locks / "prepared.json", prepared)
    return prepared


def validate_prepared(prepared: dict, project: Path, work: Path, model: dict) -> None:
    """Validate a prepared environment once per run, not once per image."""
    if prepared["identity"]["controller_code"] != code_fingerprint(project):
        raise RuntimeError("Controller changed; rerun preparation for a new execution fingerprint")
    if prepared["identity"]["registry"] != model:
        raise RuntimeError(f"Registry changed for {model['id']}; use a new work directory")
    if sha256(prepared["config"]) != prepared["worker_config_sha256"]:
        raise RuntimeError("Worker configuration changed; rerun preparation")
    locks = work / "locks" / model["id"]
    if sha256(locks / "requirements.lock") != prepared["identity"]["lock_sha256"]:
        raise RuntimeError("Resolved environment lock was modified")
    asset_root = work / "assets" / model["id"]
    for name, expected in prepared["identity"]["weights"].items():
        path = asset_root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Prepared asset changed or disappeared: {path}")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv must remain available to validate the installed environment")
    frozen = subprocess.check_output(
        [uv, "pip", "freeze", "--python", prepared["python"]], text=True
    )
    import hashlib

    if hashlib.sha256(frozen.encode()).hexdigest() != prepared["identity"]["installed_sha256"]:
        raise RuntimeError(f"Installed packages changed for {model['id']}; rerun preparation")
    if model.get("source"):
        source = Path(prepared["source_path"])
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if actual != model["source"].get("revision", model["source"].get("commit")):
            raise RuntimeError("Source checkout HEAD changed")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=source).returncode:
            raise RuntimeError(f"Tracked source files changed: {source}")
    runtime_path = locks / "runtime_check.json"
    command(
        [prepared["python"], "-m", "wmbench.worker", "runtime", runtime_path],
        env=worker_environment(project, work, model),
    )
    if read_json(runtime_path) != prepared["identity"]["runtime"]:
        raise RuntimeError(
            f"Runtime/GPU changed for {model['id']}; rerun prepare, without mixing old execution IDs"
        )
