"""StegaStamp-only reproducible CUDA decoding with an audited paired Clean.

The bank's historical extraction is a reference, not a bitwise oracle across
inference environments. Use a declared, fixed numerical profile for both Clean
and attacks; preserve and report the historical reference separately. No weight,
payload, carrier, preprocessing, threshold or attack changes. No voting or
selection by BER. Original strict behavior remains for every other method.
"""

import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import stegastamp_cuda_hotfix as cuda

METHOD = "stegastamp"
VERSION = "stegastamp-paired-cuda-v1"
FIELDS = ("status", "decoded_bits", "ber", "exact")
POLICY = {
    "version": VERSION, "backend": "CUDAExecutionProvider", "use_tf32": "0",
    "cudnn_conv_algo_search": "HEURISTIC", "use_deterministic_compute": True,
    "execution_mode": "sequential", "intra_op_num_threads": 1, "inter_op_num_threads": 1,
    "cublas_workspace_config": ":4096:8", "same_session_extractions": 3,
    "fresh_session_extractions": 1, "paired_control": "first_fresh_extraction",
    "historical_bank": "preserved_and_reported_not_used_as_a_bitwise_oracle",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def same(left, right):
    return all(left.get(key) == right.get(key) for key in FIELDS)


def session_options(ort):
    options = ort.SessionOptions()
    options.use_deterministic_compute = True
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def install_numerical_profile():
    """Only the isolated StegaStamp decoder process calls this function."""
    from wmbench.adapters.physical import StegaStampONNX
    from wmbench_digital import worker

    if getattr(StegaStampONNX._load, "_paired_profile", False):
        return
    original_runtime = worker.runtime_info
    evidence = {}
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = POLICY["cublas_workspace_config"]

    def load(self):
        import onnxruntime as ort
        if not hasattr(self, "session"):
            if not self.device.startswith("cuda"):
                raise RuntimeError("This StegaStamp numerical profile requires CUDA, not CPU")
            weight = self._weight("model_files/stega_stamp/stega_stamp.onnx")
            device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            provider = {"device_id": device_id, "use_tf32": POLICY["use_tf32"],
                        "cudnn_conv_algo_search": POLICY["cudnn_conv_algo_search"]}
            session = ort.InferenceSession(str(weight), sess_options=session_options(ort),
                                          providers=[("CUDAExecutionProvider", provider), "CPUExecutionProvider"])
            actual = cuda.require_cuda(session)
            options = actual["provider_options"]["CUDAExecutionProvider"]
            if options.get("use_tf32") != "0" or options.get("cudnn_conv_algo_search") not in {"HEURISTIC", "1"}:
                raise RuntimeError("ORT did not apply the declared StegaStamp numerical profile")
            if not {"image", "secret"}.issubset({v.name for v in session.get_inputs()}):
                raise ValueError("Unexpected StegaStamp ONNX input names")
            if not {"stegastamp", "residual", "decoded"}.issubset({v.name for v in session.get_outputs()}):
                raise ValueError("Unexpected StegaStamp ONNX output names")
            # Version/hash evidence has no ephemeral /content paths in its identity.
            linked = json.loads(os.environ["WM_STEGA_CUDA_LIBRARIES"])
            gpu = subprocess.run(["nvidia-smi", f"--id={device_id}", "--query-gpu=name,driver_version",
                                  "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
            identity = {"policy": POLICY, "onnxruntime": ort.__version__, "onnx_sha256": cuda.checksum(weight),
                        "gpu_and_driver": gpu, "python": sys.version.split()[0],
                        "actual_providers": actual["actual_providers"], "provider_options": actual["provider_options"],
                        "packages": linked["packages"],
                        "library_sha256": {name: v["sha256"] for name, v in linked["libraries"].items()}}
            evidence.update(version=VERSION, fingerprint=digest(identity), identity=identity)
            self.session = session
        cuda.require_cuda(self.session)
        return self.session

    def runtime():
        return {**original_runtime(), "stegastamp_paired": dict(evidence)}

    load._paired_profile = True
    StegaStampONNX._load = load
    worker.runtime_info = runtime


def audit_root(job):
    return Path(job["results"]) / METHOD / VERSION


def receipt_path(job):
    return audit_root(job) / "preflight" / (job["session"] + ".json")


def baseline_path(job, case):
    from wmbench_full_payload.storage import inside
    return inside(audit_root(job) / "controls", case["id"] + ".json")


def execution_profile(runtime):
    profile = runtime.get("stegastamp_paired", {})
    identity = profile.get("identity", {})
    if (profile.get("version") != VERSION or identity.get("policy") != POLICY
            or identity.get("actual_providers", [])[:1] != ["CUDAExecutionProvider"]
            or profile.get("fingerprint") != digest(identity)):
        raise RuntimeError("Missing/invalid actual CUDA numerical-profile evidence")
    return profile


def current_record(document):
    if not document or not document.get("rows"):
        return False
    try:
        profiles = [execution_profile(row["execution"]["runtime"])["fingerprint"]
                    for row in document["rows"].values()]
        return len(set(profiles)) == 1 and document["rows"]["clean"]["paired_control"]["version"] == VERSION
    except (KeyError, TypeError, RuntimeError):
        return False


def validated_input(job, case):
    from wmbench.io import read_json, sha256
    from wmbench_full_payload.storage import inside, EvidenceError
    if case["method"] != METHOD:
        raise EvidenceError("StegaStamp repair must not inspect another method")
    carrier = inside(Path(job["bank"]), case["output"])
    metadata = inside(Path(job["bank"]), case["decoder_metadata"])
    blind = read_json(metadata)
    for field in ("expected_bits", "payload", "full_payload"):
        if isinstance(blind.get(field), list) and any(blind[field]):
            raise EvidenceError("Reference payload unexpectedly present in decoder metadata")
    if sha256(carrier) != case["carrier_sha256"]:
        raise EvidenceError("Bank carrier hash changed")
    return carrier, metadata


def check_stable(rows):
    if (not rows or any(row.get("status") != "measured" for row in rows)
            or any(not same(rows[0], row) for row in rows[1:])):
        raise RuntimeError("StegaStamp is not repeatable under the fixed CUDA profile; "
                           "all extractions retained for diagnosis; no best result selected")


def historical_difference(old, fresh):
    a, b = old.get("decoded_bits"), fresh.get("decoded_bits")
    return {"bank_ber": old.get("ber"), "paired_ber": fresh.get("ber"),
            "ber_difference_pp": 100 * (fresh["ber"] - old["ber"]) if old.get("ber") is not None else None,
            "different_bit_positions": [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
                if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) else None}


def preflight(job):
    """Three passes in one session and a fourth in a newly created session.

    First extraction is the control regardless of its BER. Historical drift is
    disclosed, never called an attack effect. Existing paired controls cannot
    change on resume, even when no attack has been decoded yet.
    """
    from wmbench.io import read_json, write_json
    from wmbench_full_payload.storage import freeze
    from wmbench_digital import worker
    if job["model"]["id"] != METHOD:
        raise ValueError("StegaStamp only")
    cases = read_json(Path(job["results"]) / METHOD / "cases.json")
    install_numerical_profile()
    adapter = worker.load_adapter(job)  # Original weights/dependency/geometry checks.
    profile = execution_profile(worker.runtime_info())
    report = {"version": VERSION, "identity": job["identity"], "session": job["session"],
              "case_keys": [c["key"] for c in cases], "profile": profile, "checks": [], "status": "running"}
    write_json(receipt_path(job), report)
    try:
        for index, case in enumerate(cases, 1):
            carrier, metadata = validated_input(job, case)
            rows = []
            for _ in range(POLICY["same_session_extractions"]):
                if time.time() >= job["deadline"]:
                    raise TimeoutError("Budget reached in StegaStamp repeatability preflight")
                rows.append(worker.measured(adapter, carrier, metadata, case, int(case["signature"][:8], 16)))
            check = {"case_key": case["key"], "case_id": case["id"], "extractions": rows,
                     "historical": case["clean"], "passed": False}
            report["checks"].append(check)
            write_json(receipt_path(job), report)
            check_stable(rows)
            print(f"STEGASTAMP PAIRED CLEAN {index}/{len(cases)} {case['id']}: 3/3 stable", flush=True)
        del adapter
        gc.collect()
        adapter = worker.load_adapter(job)
        if execution_profile(worker.runtime_info()) != profile:
            raise RuntimeError("StegaStamp runtime identity changed during session recreation")
        for index, (case, check) in enumerate(zip(cases, report["checks"]), 1):
            if time.time() >= job["deadline"]:
                raise TimeoutError("Budget reached during fresh-session replay")
            carrier, metadata = validated_input(job, case)
            row = worker.measured(adapter, carrier, metadata, case, int(case["signature"][:8], 16))
            check["extractions"].append(row)
            write_json(receipt_path(job), report)
            check_stable(check["extractions"])
            first = check["extractions"][0]
            target = baseline_path(job, case)
            if target.exists():
                previous = read_json(target)
                if (previous["case_key"] != case["key"] or previous["profile"] != profile
                        or previous["carrier_sha256"] != case["carrier_sha256"] or not same(previous["first"], first)):
                    raise RuntimeError("Paired Clean/environment changed on resume; old controls/results preserved")
            check.update(passed=True, difference=historical_difference(case["clean"], first))
            write_json(receipt_path(job), report)
            print(f"STEGASTAMP FRESH SESSION {index}/{len(cases)}: stable; "
                  f"bank BER={case['clean'].get('ber')}, paired BER={first.get('ber')}", flush=True)
        # Commit controls only after all cases pass. Freeze preserves the first
        # observation, including its timing; later runs do not overwrite it.
        for case, check in zip(cases, report["checks"]):
            target = baseline_path(job, case)
            if not target.exists():
                freeze(target, {"version": VERSION, "case_key": case["key"], "profile": profile,
                                "carrier_sha256": case["carrier_sha256"], "first": check["extractions"][0],
                                "historical": case["clean"], "source_preflight": str(receipt_path(job))})
        report["status"] = "passed"
        write_json(receipt_path(job), report)
        write_audit(job, cases)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        write_json(receipt_path(job), report)
        raise


def write_audit(job, cases):
    from wmbench.io import read_json, write_csv, atomic_bytes
    rows = []
    for case in cases:
        control = read_json(baseline_path(job, case))
        diff = historical_difference(case["clean"], control["first"])
        rows.append({"case_id": case["id"], "mode": case["mode"], "image_id": case["image_id"],
                     "replicate": case["replicate"], "bank_ber_percent": 100 * diff["bank_ber"] if diff["bank_ber"] is not None else None,
                     "paired_clean_ber_percent": 100 * diff["paired_ber"], "difference_pp": diff["ber_difference_pp"],
                     "bank_exact": case["clean"].get("exact"), "paired_exact": control["first"].get("exact"),
                     "different_bit_positions": json.dumps(diff["different_bit_positions"]),
                     "runtime_fingerprint": control["profile"]["fingerprint"], "control_rule": POLICY["paired_control"]})
    directory = Path(job["results"]) / METHOD / "tables"
    write_csv(directory / "stegastamp_bank_vs_paired_clean.csv", rows)
    atomic_bytes(directory / "STEGASTAMP_NUMERICAL_PROFILE.md", (
        "# StegaStamp numerical execution profile\n\n"
        "The embedding bank and its historical Clean are unchanged. Digital-channel scores use a first-fresh paired Clean "
        "extraction, measured with the same declared CUDA numerical profile as the attacked PNGs. Historical extraction "
        "differences are reported in `stegastamp_bank_vs_paired_clean.csv`, not attributed to attacks. "
        "This is an explicit numerical-profile change, not an exact reproduction of the old execution environment. "
        "No payload, weights, thresholds, messages or images changed; no voting or best-of-N.\n\n"
        + json.dumps(POLICY, indent=2) + "\n").encode())


def archive_legacy(job, cases):
    from wmbench.io import read_json, write_json, atomic_bytes
    from wmbench_digital.storage import result_path, read_result
    report = read_json(receipt_path(job))
    if (report["version"] != VERSION or report["status"] != "passed" or report["identity"] != job["identity"]
            or report["case_keys"] != [c["key"] for c in cases]
            or [r["case_key"] for r in report["checks"] if r["passed"]] != report["case_keys"]):
        raise RuntimeError("All repeatability checks must pass before legacy withdrawal")
    root = Path(job["results"]).resolve()
    active, backup_root = root / METHOD / "records", root / METHOD / "_preserved_previous_records" / VERSION
    if any(p.is_symlink() for p in (root / METHOD, active, backup_root.parent, backup_root)):
        raise RuntimeError("Refusing linked record/backup directories")
    count = 0
    for case in cases:
        path = result_path(root, case)
        if case["method"] != METHOD or path.is_symlink() or path.parent.resolve() != active.resolve():
            raise RuntimeError("Unsafe legacy record target")
        record = read_result(path, case)
        if record is None:
            continue
        if current_record(record):
            control = read_json(baseline_path(job, case))
            if (not same(record["rows"]["clean"], control["first"])
                    or execution_profile(record["rows"]["clean"]["execution"]["runtime"]) != report["profile"]):
                raise RuntimeError("Existing paired records conflict with current controls; retained, not silently replaced")
            continue
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        backup = backup_root / checksum / path.name
        if backup.parent.is_symlink() or backup.is_symlink():
            raise RuntimeError("Refusing linked backup")
        if backup.exists() and backup.read_bytes() != data:
            raise RuntimeError("Legacy backup conflict; active file retained")
        if not backup.exists():
            atomic_bytes(backup, data)
        if cuda.checksum(backup) != checksum or read_result(backup, case) != record:
            raise RuntimeError("Backup verification failed; active file retained")
        receipt = {"version": VERSION, "case_key": case["key"], "source": str(path), "backup": str(backup),
                   "sha256": checksum, "preflight": str(receipt_path(job)), "reason": "superseded numerical execution profile"}
        receipt_file = backup.with_suffix(".receipt.json")
        write_json(receipt_file, receipt)
        if read_json(receipt_file) != receipt or cuda.checksum(path) != checksum:
            raise RuntimeError("Readback conflict before withdrawal")
        path.unlink()  # This exact verified record only; byte-identical backup is persistent.
        count += 1
    print(f"STEGASTAMP: {count} legacy records preserved in {backup_root}; recompute with paired Clean", flush=True)


def check_clean(clean, case, adapter, metadata, rng_seed, root, execution, prior, job):
    from wmbench.io import read_json
    from wmbench_full_payload.storage import EvidenceError
    control = read_json(baseline_path(job, case))
    profile = execution_profile(execution["runtime"])
    if (case["method"] != METHOD or control["version"] != VERSION or control["case_key"] != case["key"]
            or control["carrier_sha256"] != case["carrier_sha256"] or control["profile"] != profile
            or not same(clean, control["first"])):
        raise EvidenceError("Fresh extraction does not match the committed paired Clean/profile; rows not mixed")
    if prior and (not current_record({"rows": prior}) or not same(prior["clean"], clean)
                  or execution_profile(prior["clean"]["execution"]["runtime"]) != profile):
        raise EvidenceError("Partial case uses a different Clean/profile; existing results retained")
    clean["paired_control"] = {"version": VERSION, "path": str(baseline_path(job, case)),
                               "historical_bank_ber": case["clean"].get("ber"),
                               "rule": POLICY["paired_control"], "profile": profile["fingerprint"]}


def install_worker_patch():
    from wmbench_digital import worker
    if cuda.checksum(worker.__file__) != cuda.WORKER_SHA:
        raise RuntimeError("Unexpected worker source")
    tree = ast.parse(Path(worker.__file__).read_text("utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
    count = 0
    class ReplaceGate(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal count
            if (isinstance(node.target, ast.Name) and node.target.id == "field" and isinstance(node.iter, ast.Tuple)
                    and [getattr(n, "value", None) for n in node.iter.elts] == list(FIELDS)):
                count += 1
                return ast.copy_location(ast.parse("_paired_clean(clean, case, adapter, metadata, rng_seed, root, execution, prior, job)").body[0], node)
            return self.generic_visit(node)
    function = ReplaceGate().visit(function)
    if count != 1:
        raise RuntimeError("Expected exactly one Clean comparison gate")
    worker.__dict__["_paired_clean"] = check_clean
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(worker.__file__) + " [StegaStamp paired Clean]", "exec"), worker.__dict__)
    install_numerical_profile()


def install_controller_patch():
    # Reuse the already tested CUDA-library and missing-cache fixes, without
    # editing their files or changing the frozen source archive.
    old_redirect = cuda.redirect
    def redirect(command, env, linked):
        command, updated = old_redirect(command, env, linked)
        for i in range(len(command) - 1):
            if command[i:i+2] == ["-m", "stegastamp_cuda_hotfix"]:
                command[i+1] = "stegastamp_paired_hotfix"
                updated = {**updated, "CUBLAS_WORKSPACE_CONFIG": POLICY["cublas_workspace_config"]}
                updated["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + updated.get("PYTHONPATH", "")
        return command, updated
    cuda.redirect = redirect
    cuda.cuda_record = current_record
    cuda.quarantine_legacy = archive_legacy
    cuda.install_controller_patch()


def main():
    from wmbench.io import read_json
    from wmbench_digital import runner, worker
    action = sys.argv[1]
    if action == "run":
        install_controller_patch()
        runner.main()
    elif action in {"preflight", "decode"}:
        job = read_json(Path(sys.argv[sys.argv.index("--settings") + 1]))
        if job["model"]["id"] != METHOD:
            raise ValueError("StegaStamp only")
        if action == "preflight":
            preflight(job)
        else:
            install_worker_patch()
            worker.decode(job)
    else:
        raise ValueError("Expected run, preflight or decode")


if __name__ == "__main__":
    main()
