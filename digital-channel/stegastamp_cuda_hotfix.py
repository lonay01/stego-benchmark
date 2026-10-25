"""Restore StegaStamp's CUDA runtime without changing the frozen benchmark.

The previous digital run silently used CPU because ONNX could not find CUDA
libraries. The embedding bank used CUDA. This overlay links the controller's
existing CUDA dependencies into StegaStamp children, verifies actual session
providers, and retains the original strict Clean gate. No weights, bits, images,
attacks, dependency locks or other methods are modified.
"""

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

METHOD = "stegastamp"
VERSION = "stegastamp-cuda-v1"
RUNNER_SHA = "1357f717a1d46fbc3f726ecc097455241ac6d13e628b0f54b2741860febedf92"
WORKER_SHA = "57f68d4e9c6134c1f5264703e6e00ac1a6ef55a0040265f5b0c6dfd46ef6cf2e"
FIELDS = ("status", "decoded_bits", "ber", "exact")
REQUIRED_LIBRARIES = ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9", "libcudart.so.12")


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def library_environment(env):
    """Use existing PyTorch CUDA12/cuDNN9 wheels, outside the locked ORT venv."""
    names = ("cublas", "cuda_runtime", "cuda_nvrtc", "cudnn", "cufft", "curand",
             "cusolver", "cusparse", "nvjitlink", "nvtx")
    folders, packages = [], {}
    for name in names:
        package = "nvidia-" + name.replace("_", "-") + "-cu12"
        try:
            dist = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        directory = Path(dist.locate_file("nvidia")) / name / "lib"
        if directory.is_dir():
            folders.append(directory.resolve())
            packages[package] = dist.version
    libraries = {}
    for filename in REQUIRED_LIBRARIES:
        matches = [folder / filename for folder in folders if (folder / filename).is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"StegaStamp needs exactly one {filename} in the controller's NVIDIA wheels; "
                               "rerun the original environment setup cell. No results were removed.")
        path = matches[0].resolve()
        libraries[filename] = {"path": str(path), "sha256": checksum(path)}
    patched = dict(env)
    patched["LD_LIBRARY_PATH"] = os.pathsep.join([*map(str, folders), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    patched["WM_STEGA_CUDA_LIBRARIES"] = json.dumps({"packages": packages, "libraries": libraries}, sort_keys=True)
    return patched


def require_cuda(session):
    """Available providers are not evidence that this session loaded CUDA."""
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"StegaStamp CUDA did not load (actual providers: {providers}). "
                           "CPU-only fallback is forbidden; no attack scores accepted.")
    session.disable_fallback()  # Do not retry a failed CUDA session on CPU.
    return {"fix": VERSION, "actual_providers": providers,
            "provider_options": session.get_provider_options(), "automatic_fallback": False}


def install_worker_guard():
    """Wrap only StegaStamp's loader; retain the original decoding and Clean gate."""
    from wmbench_digital import worker
    from wmbench.adapters.physical import StegaStampONNX

    if checksum(worker.__file__) != WORKER_SHA:
        raise RuntimeError("Unexpected digital worker; refusing CUDA overlay")
    # Never compose this repair with the earlier one-bit acceptance exception.
    tree = ast.parse(Path(worker.__file__).read_text("utf-8"))
    decode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
    exec(compile(ast.Module(body=[decode], type_ignores=[]), str(worker.__file__), "exec"), worker.__dict__)
    if getattr(StegaStampONNX._load, "_cuda_guard", False):
        return
    original_load, original_runtime = StegaStampONNX._load, worker.runtime_info
    evidence = {}

    def guarded_load(self):
        session = original_load(self)
        evidence.update(require_cuda(session))
        return session

    def runtime():
        return {**original_runtime(), "stegastamp_cuda": {
            **evidence, "linked_dependencies": json.loads(os.environ.get("WM_STEGA_CUDA_LIBRARIES", "{}"))}}

    guarded_load._cuda_guard = True
    StegaStampONNX._load = guarded_load
    worker.runtime_info = runtime


def receipt_path(job):
    return Path(job["results"]) / METHOD / "cuda-preflight" / (job["session"] + ".json")


def preflight(job):
    """All selected Clean carriers must replay exactly before replacing CPU rows."""
    from wmbench.io import read_json, write_json, sha256
    from wmbench_full_payload.storage import EvidenceError, inside
    from wmbench_digital import worker

    if job["model"]["id"] != METHOD:
        raise ValueError("CUDA preflight is StegaStamp-only")
    root, bank = Path(job["results"]), Path(job["bank"])
    cases = read_json(root / METHOD / "cases.json")
    install_worker_guard()
    adapter = worker.load_adapter(job)
    evidence = worker.runtime_info()
    print("STEGASTAMP ACTUAL PROVIDERS: " + str(evidence["stegastamp_cuda"]["actual_providers"]), flush=True)
    report = dict(fix=VERSION, session=job["session"], identity=job["identity"],
                  case_keys=[c["key"] for c in cases], execution=evidence, checks=[], status="running")
    write_json(receipt_path(job), report)
    for index, case in enumerate(cases, 1):
        if time.time() >= job["deadline"]:
            raise TimeoutError("Session budget reached during CUDA Clean preflight; CPU rows retained")
        carrier, metadata = inside(bank, case["output"]), inside(bank, case["decoder_metadata"])
        blind = read_json(metadata)
        if blind.get("runtime_providers", [None])[0] != "CUDAExecutionProvider":
            raise EvidenceError("Bank carrier was not recorded with CUDA; cannot silently change its backend")
        if any(any(blind.get(field, [])) for field in ("expected_bits", "payload", "full_payload")
               if isinstance(blind.get(field), list)):
            raise EvidenceError("Reference payload unexpectedly present in decoder metadata")
        if sha256(carrier) != case["carrier_sha256"]:
            raise EvidenceError("Bank carrier hash changed")
        fresh = worker.measured(adapter, carrier, metadata, case, int(case["signature"][:8], 16))
        passed = all(fresh.get(k) == case["clean"].get(k) for k in FIELDS)
        report["checks"].append({"case_key": case["key"], "case_id": case["id"],
                                  "fresh": fresh, "bank": case["clean"], "passed": passed})
        report["status"] = "running" if passed else "failed"
        write_json(receipt_path(job), report)
        print(f"STEGASTAMP CUDA CLEAN {index}/{len(cases)} {case['id']}: {'exact replay' if passed else 'FAILED'}", flush=True)
        if not passed:
            raise EvidenceError("StegaStamp still differs on actual CUDA; old results retained. "
                                "Inspect cuda-preflight; no tolerance or bit correction was applied.")
    report["status"] = "passed"
    write_json(receipt_path(job), report)


def cuda_record(document):
    """Only rows produced with the actual-provider guard are resumable as CUDA."""
    return bool(document and document["rows"] and all(
        row.get("execution", {}).get("runtime", {}).get("stegastamp_cuda", {}).get("fix") == VERSION
        and row["execution"]["runtime"]["stegastamp_cuda"].get("actual_providers", [None])[0] == "CUDAExecutionProvider"
        for row in document["rows"].values()))


def quarantine_legacy(job, cases):
    """Backup, verify and withdraw ONLY selected legacy StegaStamp record files.

    Content-addressed backups and append-only per-file receipts make a restart
    safe after any copy/write/unlink boundary. Bank PNGs and other methods never
    enter this function's target set. New CUDA rows are never withdrawn.
    """
    from wmbench.io import read_json, write_json, atomic_bytes
    from wmbench_digital.storage import result_path, read_result

    report = read_json(receipt_path(job))
    checked_keys = [row["case_key"] for row in report.get("checks", []) if row.get("passed") is True]
    runtime = report.get("execution", {}).get("stegastamp_cuda", {})
    if (job["model"]["id"] != METHOD or report["status"] != "passed"
            or report["fix"] != VERSION or report["session"] != job["session"]
            or report["identity"] != job["identity"]
            or runtime.get("fix") != VERSION
            or runtime.get("actual_providers", [None])[0] != "CUDAExecutionProvider"
            or checked_keys != [c["key"] for c in cases]
            or report["case_keys"] != [c["key"] for c in cases]):
        raise RuntimeError("Strict CUDA preflight must pass before legacy records can be archived")
    root = Path(job["results"]).resolve()
    active = root / METHOD / "records"
    backup_root = root / METHOD / "_preserved_cpu_records" / VERSION
    if any(p.is_symlink() for p in (root / METHOD, active, backup_root.parent, backup_root)):
        raise RuntimeError("Refusing linked result/backup directories")
    withdrawn = 0
    for case in cases:
        path = result_path(root, case)
        if case["method"] != METHOD or path.parent.resolve() != active.resolve() or path.is_symlink():
            raise RuntimeError("Unsafe legacy record target")
        document = read_result(path, case)
        if document is None or cuda_record(document):
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        backup = backup_root / digest / path.name
        if backup.parent.is_symlink() or backup.is_symlink():
            raise RuntimeError("Refusing linked backup file")
        if backup.exists() and backup.read_bytes() != data:
            raise RuntimeError("Legacy backup conflict; active result retained")
        if not backup.exists():
            atomic_bytes(backup, data)
        if checksum(backup) != digest or read_result(backup, case) != document:
            raise RuntimeError("Legacy backup readback failed; active result retained")
        receipt = {"fix": VERSION, "reason": "legacy StegaStamp results without verified CUDA provider",
                   "case_key": case["key"], "source": str(path), "backup": str(backup),
                   "sha256": digest, "preflight": str(receipt_path(job)), "state": "backed_up"}
        receipt_file = backup.with_suffix(".receipt.json")
        write_json(receipt_file, receipt)
        if read_json(receipt_file) != receipt or checksum(path) != digest:
            raise RuntimeError("Record changed before withdrawal; active result retained")
        path.unlink()  # Exact checked record only; verified byte-identical backup persists on Drive.
        receipt["state"] = "withdrawn_from_active_tables"
        write_json(receipt_file, receipt)
        withdrawn += 1
    print(f"STEGASTAMP CUDA: {withdrawn} legacy records preserved under {backup_root}; "
          "only these records need recomputation. Other methods unchanged.", flush=True)


def redirect(command, env, linked_env):
    """Modify only StegaStamp prepare/decode children, never global LD paths."""
    from wmbench.io import read_json

    command = list(command)
    for index in range(len(command) - 4):
        if command[index:index+2] != ["-m", "wmbench_digital.runner"]:
            continue
        stage = command[index+2]
        if stage not in {"prepare", "decode"} or command[index+3] != "--settings":
            continue
        job = read_json(command[index+4])
        if job["model"]["id"] != METHOD:
            return command, env
        patched = dict(env)
        for key in ("LD_LIBRARY_PATH", "WM_STEGA_CUDA_LIBRARIES"):
            patched[key] = linked_env[key]
        if stage == "decode":
            command[index+1] = "stegastamp_cuda_hotfix"
            if job.get("stegastamp_cuda_preflight"):
                command[index+2] = "preflight"
            patched["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
        return command, patched
    return command, env


def install_controller_patch():
    """Insert one guarded StegaStamp preparation hook before original stages."""
    from wmbench import colab_canary
    from wmbench.io import read_json, write_json
    from wmbench_digital import runner
    from wmbench_digital.storage import read_result, result_path, complete
    from digital_resume_hotfix import install_cleanup_patch

    install_cleanup_patch()
    if checksum(runner.__file__) != RUNNER_SHA:
        raise RuntimeError("Unexpected controller version; refusing CUDA overlay")
    original_monitor, original_invoke = colab_canary.monitored_process, runner.invoke
    prepared_sessions, linked = set(), {}

    def monitored(command, env, *args, **kwargs):
        # Lazy: completed methods need no CUDA preparation or linkage changes.
        if linked:
            command, env = redirect(command, env, linked)
        return original_monitor(command, env, *args, **kwargs)

    def invoke(job, stage, path, owned):
        if job["model"]["id"] == METHOD and stage == "prepare" and job["session"] in prepared_sessions:
            return
        return original_invoke(job, stage, path, owned)

    def before(job, path, owned, cases):
        if job["model"]["id"] != METHOD:
            return
        documents = [read_result(result_path(job["results"], c), c) for c in cases]
        if all(complete(d) and cuda_record(d) for d in documents):
            return
        linked.update(library_environment(os.environ))
        owned()
        original_invoke(job, "prepare", path, owned)
        prepared_sessions.add(job["session"])
        probe = {**job, "session": job["session"] + "_cuda-preflight", "stegastamp_cuda_preflight": True}
        probe_path = Path(path).with_name(probe["session"] + ".json")
        write_json(probe_path, probe)
        original_invoke(probe, "decode", probe_path, owned)
        owned()
        quarantine_legacy(probe, cases)
        # Original attacks stage now sees archived cases as pending and recreates
        # missing ZIPs with its existing deterministic SHA checks.

    tree = ast.parse(Path(runner.__file__).read_text("utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    count = 0

    class InsertBeforeDone(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "done":
                count += 1
                return [ast.copy_location(ast.parse("_stega_cuda_before(job, path, owned, cases)").body[0], node), node]
            return self.generic_visit(node)

    function = InsertBeforeDone().visit(function)
    if count != 1:
        raise RuntimeError("Expected exactly one method-completion boundary")
    runner.__dict__["_stega_cuda_before"] = before
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(runner.__file__) + " [StegaStamp CUDA linkage]", "exec"), runner.__dict__)
    colab_canary.monitored_process = monitored
    runner.invoke = invoke


def main():
    from wmbench.io import read_json
    from wmbench_digital import runner, worker

    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "run":
        install_controller_patch()
        runner.main()
    elif action in {"decode", "preflight"}:
        job = read_json(Path(sys.argv[sys.argv.index("--settings")+1]))
        if job["model"]["id"] != METHOD:
            raise ValueError("CUDA repair decoder is StegaStamp-only")
        if action == "preflight":
            preflight(job)
        else:
            install_worker_guard()
            worker.decode(job)
    else:
        raise ValueError("Use original runner for validation/report; this overlay supports run/decode/preflight")


if __name__ == "__main__":
    main()
