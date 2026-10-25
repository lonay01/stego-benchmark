"""Parallelize independent case ZIPs without changing any serialized attack."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil

from wmbench.io import read_json, write_json
from wmbench_digital import packages as original
from wmbench_digital.storage import cache_root, result_path, complete
from .cache import ResultCache


def one_case(task):
    job, case, location = task
    real_read, real_write = original.read_json, original.write_json
    root, name = Path(job["results"]), job["model"]["id"]
    original.read_json = lambda p: [case] if Path(p) == root/name/"cases.json" else real_read(p)
    original.write_json = lambda p, v: None if Path(p) == root/name/"packages-ready.json" else real_write(p, v)
    # The parent checked the whole pending-method bound, not a per-worker bound.
    real_cache_root = original.cache_root
    original.cache_root = lambda _base, identity, method: real_cache_root(Path(location), identity, method)
    try:
        original.prepare_packages(job)
        return real_read(root / name / "packages" / (case["id"] + ".json"))
    finally:
        original.read_json, original.write_json = real_read, real_write
        original.cache_root = real_cache_root


def prepare(job):
    root, name = Path(job["results"]), job["model"]["id"]
    cases = read_json(root / name / "cases.json")
    cache = ResultCache(root, name, cases, job["settings"].get("full_audit", False))
    pending = [c for c in cases if not complete(cache.read(result_path(root, c), c))]
    bound = sum(c["size"][0]*c["size"][1]*3*78*1.05 + 2*1024**2 for c in pending)
    scratch = Path(job["scratch"])
    reserve = job["settings"]["ssd_reserve_gib"] * 1024**3
    location = scratch if shutil.disk_usage(scratch).free > reserve+bound else root/"_temporary"
    cache_root(location, job["identity"], name)  # Single writer of shared ownership marker.
    print(f"PARALLEL ATTACK STORAGE: {location}; total upper bound {bound/1024**3:.1f} GiB", flush=True)
    # Freeze the runtime once in the parent to avoid a concurrent first write.
    import importlib.metadata
    import zlib
    original.freeze(root / "attack-runtime.json", {
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "scipy")},
        "zlib": zlib.ZLIB_RUNTIME_VERSION})
    workers = job["settings"].get("attack_workers", 2)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, _ in enumerate(pool.map(one_case, ((job, c, str(location)) for c in pending)), 1):
            print(f"ATTACK CASES COMMITTED: {index}/{len(pending)}", flush=True)
    manifests = []
    for case in cases:
        path = root / name / "packages" / (case["id"] + ".json")
        if path.exists():
            manifests.append(read_json(path))
    write_json(root / name / "packages-ready.json", dict(case_keys=[c["key"] for c in cases], manifests=manifests))
    cache.flush()
