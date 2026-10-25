"""Run the frozen decoder on a disjoint whole-image shard.

The original scoring, seeds, errors and Clean gate are unchanged. Only case
selection, temporary paths and checkpoint I/O are intercepted in this process.
"""

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import time

from wmbench.io import read_json, write_json, atomic_bytes
from wmbench_full_payload.storage import EvidenceError, inside
from wmbench_digital import worker as original
from wmbench_digital.spec import ATTACKS
from .cache import ResultCache, verify_inputs

WORKER_SHA = "57f68d4e9c6134c1f5264703e6e00ac1a6ef55a0040265f5b0c6dfd46ef6cf2e"
FIELDS = ("status", "decoded_bits", "ber", "exact")


def resources():
    """Process memory is a lower-bound sample, not a peak-memory guarantee."""
    try:
        import psutil
    except ModuleNotFoundError as error:
        if error.name != "psutil":
            raise
        # Keep the model's locked environment unchanged. The controller already
        # has psutil and can inspect this worker and its live descendants.
        monitor = os.environ.get("WM_BENCH_TELEMETRY_PYTHON")
        if not monitor:
            raise RuntimeError("Missing controller Python for psutil telemetry") from error
        script = (
            "import psutil, sys\n"
            "me = psutil.Process(int(sys.argv[1]))\n"
            "print(sum(p.memory_info().rss for p in "
            "[me, *me.children(recursive=True)] if p.is_running()) / 1024**3)\n"
        )
        rss = float(subprocess.check_output(
            [monitor, "-I", "-c", script, str(os.getpid())], text=True, timeout=15,
        ).strip())
        if not 0 < rss < float("inf"):
            raise RuntimeError("Invalid worker RAM measurement from controller")
        print("Telemetry: decoder has no psutil; worker RAM sampled by controller.", flush=True)
    else:
        me = psutil.Process()
        rss = sum(p.memory_info().rss for p in [me, *me.children(recursive=True)] if p.is_running()) / 1024**3
    gpu_gib = None
    try:
        data = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"
        ], text=True, timeout=15)
        gpu_gib = sum(float(line.split(",")[1]) for line in data.splitlines()
                      if int(line.split(",")[0]) == os.getpid()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return dict(rss_gib=rss, gpu_gib=gpu_gib)


def probe(adapter, job):
    """Fixed serial reference, then bitwise worker comparison; never best-of-N."""
    case = job["probe_case"]
    verify_inputs(job, case)
    metadata = inside(job["bank"], case["decoder_metadata"])
    blind = read_json(metadata)
    if any(isinstance(blind.get(k), list) and any(blind[k])
           for k in ("expected_bits", "payload", "full_payload")):
        raise EvidenceError("Non-blind probe metadata")
    folder = Path(job["slot"]) / "tmp" / ("probe-" + str(job["worker_index"]))
    folder.mkdir(parents=True, exist_ok=True)
    # Four fixed conditions; these are diagnostics, not extra scores.
    # Attacks are generated in the PINNED controller, not a model's numpy/Pillow.
    rows = []
    for item in job["probe_inputs"]:
        from wmbench.io import sha256
        if sha256(item["path"]) != item["sha256"]:
            raise EvidenceError("Probe input changed")
        row = original.measured(adapter, item["path"], metadata, case, int(case["signature"][:8], 16))
        if row["status"] not in {"measured", "rejected"}:
            raise EvidenceError(f"Parallel safety probe failed: {row}")
        rows.append({field: row.get(field) for field in FIELDS})
    if rows[0] != {k: case["clean"].get(k) for k in FIELDS}:
        raise EvidenceError("Probe Clean differs from bank; no results replaced")
    ref = Path(job["probe_reference"])
    if job["worker_index"] == 0:
        write_json(ref, dict(case_key=case["key"], rows=rows))
    elif read_json(ref) != dict(case_key=case["key"], rows=rows):
        raise EvidenceError("Parallel worker probe differs from serial reference")
    write_json(Path(job["ready_path"]), dict(**resources(), pid=os.getpid(), time=time.time(),
                                           probe_case_key=case["key"], probe_equal=True))


def decode(job):
    if job["model"]["id"] == "stegastamp":
        raise ValueError("StegaStamp must use its reviewed serial paired-Clean entrypoint")
    data = Path(original.__file__).read_bytes()
    if hashlib.sha256(data).hexdigest() != WORKER_SHA:
        raise EvidenceError("Unreviewed decoder source; performance wrapper refused")
    root, name = Path(job["results"]), job["model"]["id"]
    all_cases = read_json(root / name / "cases.json")
    cases = [c for c in all_cases if c["image_id"] in job["shard_images"]]
    cache = ResultCache(root, name, cases, job["settings"].get("full_audit", False))
    real_read, real_load = original.read_json, original.load_adapter

    def filtered_read(path):
        path = Path(path)
        if path == root / name / "cases.json":
            return cases
        value = real_read(path)
        if path == root / name / "packages-ready.json":
            if value["case_keys"] != [c["key"] for c in all_cases]:
                raise EvidenceError("Global attack stage boundary changed")
            return {**value, "case_keys": [c["key"] for c in cases]}
        return value

    def load_adapter(current):
        adapter = real_load(current)
        probe(adapter, current)
        return adapter

    original.read_json, original.load_adapter = filtered_read, load_adapter
    original.read_result, original.save_result = cache.read, cache.save
    original.__dict__["_verify_pending"] = verify_inputs
    # Fail fast in sibling workers after a technical error; next run resumes rows.
    def verify_pending(current, case):
        if Path(current["cancel_path"]).exists():
            raise RuntimeError("Sibling worker failed; partial checkpoints retained")
        verify_inputs(current, case)
    original.__dict__["_verify_pending"] = verify_pending
    tree = ast.parse(data)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and node.value == "tmp/digital-decode":
            node.value = "tmp/digital-decode-" + str(job["worker_index"])
            count += 1
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                and [getattr(x, "id", "") for x in node.target.elts] == ["index", "case"]):
            node.body.insert(0, ast.parse("_verify_pending(job, case)").body[0])
            count += 1
    if count != 2:
        raise EvidenceError("Unexpected frozen worker structure")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(original.__file__) + " [performance wrapper]", "exec"), original.__dict__)
    try:
        original.decode(job)
    finally:
        cache.flush()


def prepare_probes(job, case):
    """Called by the pinned CPU controller before any isolated decoder starts."""
    import numpy as np
    from PIL import Image
    from wmbench.io import sha256
    from wmbench_digital.attacks import apply, png_bytes
    verify_inputs(job, case)
    folder = Path(job["slot"]) / "tmp" / job["session"] / "parallel-probes"
    folder.mkdir(parents=True, exist_ok=True)
    carrier = inside(job["bank"], case["output"])
    with Image.open(carrier) as image:
        array = np.array(image.convert("RGB"))
    result = [dict(path=str(carrier), sha256=case["carrier_sha256"])]
    probes = [next(a for a in reversed(ATTACKS) if a["family"] == family)
              for family in ("jpeg", "noise", "rotation")]
    for attack in probes:
        pixels, _ = apply(array, attack, case["image_id"], case["replicate"])
        path = folder / (attack["id"] + ".png")
        atomic_bytes(path, png_bytes(pixels))
        result.append(dict(path=str(path), sha256=sha256(path)))
    return result
