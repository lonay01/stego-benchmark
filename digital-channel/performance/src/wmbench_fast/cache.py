"""Explicit fast-resume policy; full source audit remains available.

The embedding bank is an immutable experiment input. Fast mode reuses its
previously validated case catalogue, not a new claim that every PNG was reread.
Pending carriers and decoder sidecars are still content-checked before use.
Result caches use directory size/mtime fingerprints and checksummed snapshots.
FULL_AUDIT ignores these caches; use it after external edits or storage repair.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import time

from wmbench.io import atomic_bytes, read_json, sha256, write_json
from wmbench_full_payload.storage import EvidenceError, inside
from wmbench_bank.contracts import MODES, message
from wmbench_bank.worker import signature
from wmbench_full_payload.worker import task_id
from wmbench_digital import bank as original_bank, storage
from wmbench_digital.spec import digest

RAW_READ = storage.read_result
RAW_SAVE = storage.save_result
VERSION = "fast-resume-v1"


def pack(path, value):
    data = gzip.compress(json.dumps({"digest": digest(value), "value": value},
                                    sort_keys=True, allow_nan=False).encode(), mtime=0)
    atomic_bytes(Path(path), data)
    if Path(path).read_bytes() != data:
        raise EvidenceError("Fast checkpoint readback failed")


def unpack(path):
    doc = json.loads(gzip.decompress(Path(path).read_bytes()))
    if digest(doc["value"]) != doc["digest"]:
        raise EvidenceError(f"Fast checkpoint checksum mismatch: {path}")
    return doc["value"]


def fingerprint(path):
    p = Path(path)
    if p.is_symlink():
        raise EvidenceError(f"Linked evidence is not supported: {path}")
    try:
        s = p.stat()
    except FileNotFoundError:
        return None
    return [s.st_size, s.st_mtime_ns]


def directory_inventory(folder):
    folder = Path(folder)
    if folder.is_symlink():
        raise EvidenceError("Linked results directory")
    if not folder.exists():
        return {}
    rows = {}
    for entry in os.scandir(folder):
        if not entry.name.endswith(".json.gz"):
            continue
        if entry.is_symlink() or not entry.is_file():
            raise EvidenceError(f"Unexpected result entry: {entry.name}")
        s = entry.stat()
        rows[entry.name] = [s.st_size, s.st_mtime_ns]
    return rows


def validate_catalogue(cases, coverage, ctx, model):
    """Check stored case keys/contracts, including five messages and all modes."""
    image_map = {r["image_id"]: r for r in ctx["images"]}
    cases = [c for c in cases if c["image_id"] in image_map]
    coverage = [c for c in coverage if c["image_id"] in image_map]
    rows = {(c["image_id"], c["replicate"], c["mode"]): c for c in coverage}
    if len(rows) != len(coverage):
        raise EvidenceError("Duplicate coverage rows")
    expected = {(i, r, m) for i in image_map for r in range(1, 6) for m in MODES}
    if set(rows) != expected:
        return None  # New image prefix needs ordinary bank validation.
    if any(r["status"] not in {"saved", "unmatched"} for r in rows.values()):
        raise EvidenceError("Incomplete bank catalogue")
    if any(rows[(i, r, "native")]["status"] != "saved" for i in image_map for r in range(1, 6)):
        raise EvidenceError("Missing native carrier")
    seen = set()
    for case in cases:
        item = (case["image_id"], case["replicate"], case["mode"])
        if item in seen or item not in rows or rows[item]["status"] != "saved":
            raise EvidenceError("Duplicate/unexpected saved case")
        seen.add(item)
        image = image_map[case["image_id"]]
        body = {k: v for k, v in case.items() if k != "key"}
        if case["key"] != digest({"run": ctx["identity"], "case": body}):
            raise EvidenceError("Case catalogue digest changed")
        if (case["method"] != model["id"]
                or case["id"] != f"{task_id(image, case['replicate'])}_{case['mode']}"
                or case["signature"] != signature(model, image, case["replicate"])
                or case["assignment"] != message(model, case["image_id"], case["replicate"])
                or case["size"] != [model["native_geometry"]["width"], model["native_geometry"]["height"]]
                or case["clean"]["status"] not in {"measured", "rejected"}):
            raise EvidenceError("Saved catalogue contract differs from the frozen bank")
        for key in ("output", "source_crop", "decoder_metadata"):
            inside(ctx["bank"], case[key])
    if seen != {k for k, r in rows.items() if r["status"] == "saved"}:
        raise EvidenceError("Saved carrier coverage differs from catalogue")
    return cases, coverage


def collect(ctx, model, full_audit=False):
    folder = ctx["output"] / model["id"]
    cached = folder / "_fast/catalogue.json.gz"
    pair = None
    value = None
    if not full_audit:
        if cached.exists():
            value = unpack(cached)
            if value["identity"] != ctx["identity"]:
                raise EvidenceError("Cached bank identity changed")
            pair = validate_catalogue(value["cases"], value["coverage"], ctx, model)
        elif (folder / "cases.json").exists() and (folder / "coverage.json").exists():
            pair = validate_catalogue(read_json(folder / "cases.json"), read_json(folder / "coverage.json"), ctx, model)
    if pair is None:
        from .incremental import collect_images
        cases, coverage = collect_images(ctx, model, full_audit)
        pair = validate_catalogue(cases, coverage, ctx, model)
        if pair is None:
            raise EvidenceError("Fresh bank validation did not cover selected images")
    else:
        print(f"CATALOGUE REUSED {model['id']}: {len(pair[0])} verified assignments; immutable-bank fast resume", flush=True)
    cases, coverage = pair
    current = dict(identity=ctx["identity"], cases=cases, coverage=coverage)
    if value != current:
        pack(cached, current)
    # Preserve canonical paths used by the frozen reports and StegaStamp helper.
    for filename, data in (("cases.json", cases), ("coverage.json", coverage)):
        target = folder / filename
        if not target.exists() or read_json(target) != data:
            write_json(target, data)
    return cases


def verify_inputs(job, case):
    """No cached verification for files that are about to enter a decoder."""
    from PIL import Image
    bank = Path(job["bank"])
    artifacts = {a["path"]: a["sha256"] for a in case["clean"].get("artifacts", [])}
    artifacts[case["output"]] = case["carrier_sha256"]
    if any(k not in artifacts for k in (case["decoder_metadata"], case["source_crop"])):
        # Old controls may not inventory sidecars; consult the original native record.
        from wmbench_full_payload.worker import record_path
        stem = case["id"].rsplit("_" + case["mode"], 1)[0]
        native = read_json(record_path(bank, job["model"], stem, "native"))
        artifacts.update({a["path"]: a["sha256"] for a in native.get("artifacts", [])})
        artifacts[case["output"]] = case["carrier_sha256"]
    required = (case["output"], case["decoder_metadata"], case["source_crop"])
    for name in required:
        if name not in artifacts or sha256(inside(bank, name)) != artifacts[name]:
            raise EvidenceError(f"Pending input changed or missing inventory: {name}")
    with Image.open(inside(bank, case["output"])) as png:
        if png.mode != "RGB" or list(png.size) != case["size"]:
            raise EvidenceError("Pending carrier geometry changed")


class ResultCache:
    """One checksummed blob per source image; no concurrent writers per image."""

    def __init__(self, root, method, cases, full_audit=False):
        self.folder = Path(root) / method
        self.cases = {c["id"] + ".json.gz": c for c in cases}
        self.current = directory_inventory(self.folder / "records")
        self.documents, self.stamps, self.dirty = {}, {}, set()
        self.groups = defaultdict(list)
        self.last_flush = time.monotonic()
        for filename, case in self.cases.items():
            self.groups[str(case["image_id"])].append(filename)
        if not full_audit:
            for image in self.groups:
                path = self.folder / "_fast/results" / (image + ".json.gz")
                if path.exists():
                    for filename, entry in unpack(path).items():
                        if filename in self.cases and self.current.get(filename) == entry["stat"]:
                            self._validate(entry["record"], self.cases[filename])
                            self.documents[filename] = entry["record"]
                            self.stamps[filename] = entry["stat"]

    @staticmethod
    def _validate(record, case):
        if record is not None and record["case_key"] != case["key"]:
            raise EvidenceError("Cached result belongs to another case")

    def read(self, path, case=None):
        filename = Path(path).name
        if filename not in self.cases:
            return RAW_READ(path, case)
        if filename not in self.documents:
            if filename not in self.current:
                # This process owns its shard; the initial listing is authoritative
                # until this process saves. No per-file Drive calls for absent rows.
                stamp, result = None, None
            else:
                stamp = fingerprint(path)
                result = RAW_READ(path, case or self.cases[filename])
                if fingerprint(path) != stamp:
                    raise EvidenceError("Result changed during validation; another writer may be active")
            self.documents[filename], self.stamps[filename] = result, stamp
            self.dirty.add(str(self.cases[filename]["image_id"]))
        result = self.documents[filename]
        self._validate(result, case or self.cases[filename])
        return result

    def save(self, path, value):
        filename = Path(path).name
        self._validate(value, self.cases[filename])
        RAW_SAVE(path, value)  # Mandatory original atomic write + content readback.
        self.documents[filename], self.stamps[filename] = value, fingerprint(path)
        self.dirty.add(str(self.cases[filename]["image_id"]))
        if time.monotonic() - self.last_flush > 60:
            self.flush()

    def warm(self, workers=4):
        """Migrate old records once, with bounded Drive I/O and visible progress."""
        pending = [n for n in self.cases if n not in self.documents]
        def read_one(n):
            if n not in self.current:
                return n, None, None
            p = self.folder / "records" / n
            before = fingerprint(p)
            value = RAW_READ(p, self.cases[n])
            if before != fingerprint(p):
                raise EvidenceError("Concurrent result modification")
            return n, value, before
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (n, value, stamp) in enumerate(pool.map(read_one, pending), 1):
                self.documents[n], self.stamps[n] = value, stamp
                self.dirty.add(str(self.cases[n]["image_id"]))
                if index % 50 == 0:
                    print(f"RESULT INDEX {self.folder.name}: {index}/{len(pending)} old records (one-time migration)", flush=True)
                    self.flush()
        self.flush()

    def flush(self):
        for image in self.dirty:
            entries = {n: {"record": self.documents[n], "stat": self.stamps[n]}
                       for n in self.groups[image] if n in self.documents and self.stamps[n] is not None}
            pack(self.folder / "_fast/results" / (image + ".json.gz"), entries)
        self.dirty.clear()
        self.last_flush = time.monotonic()
