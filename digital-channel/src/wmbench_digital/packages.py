"""First materialize a method's attacks; decoding starts only after this stage."""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import EvidenceError, freeze, inside
from .attacks import apply, png_bytes
from .spec import ATTACKS
from .storage import cache_root, result_path, read_result, complete


def member_name(case, attack):
    return f"{case['method']}/{case['mode']}/{attack['family']}/{attack['id']}/{case['id']}.png"


def prepare_packages(job):
    import importlib.metadata
    import zlib
    root, bank = Path(job["results"]), Path(job["bank"])
    freeze(root / "attack-runtime.json", {
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "scipy")},
        "zlib": zlib.ZLIB_RUNTIME_VERSION})
    cases = read_json(root / job["model"]["id"] / "cases.json")
    manifest_dir = root / job["model"]["id"] / "packages"
    scratch, fallback = Path(job["scratch"]), root / "_temporary"
    scratch.mkdir(parents=True, exist_ok=True)
    reserve = job["settings"]["ssd_reserve_gib"] * 1024**3
    location, manifests = None, []
    for index, case in enumerate(cases, 1):
        receipt = manifest_dir / (case["id"] + ".json")
        prior = read_json(receipt) if receipt.exists() else None
        if prior and prior["case_key"] != case["key"]:
            raise EvidenceError("Attack receipt refers to another carrier")
        if complete(read_result(result_path(root, case), case)):
            if prior:
                manifests.append(prior)
            continue
        if time.time() >= job["deadline"]:
            raise TimeoutError("Session budget reached while saving attacks; rerun to resume")
        if prior and Path(prior["archive"]).exists():
            if sha256(prior["archive"]) != prior["sha256"]:
                raise EvidenceError("Changed attack ZIP; preserve it for diagnosis")
            manifests.append(prior)
            continue
        # Safe upper bound for 78 RGB8 PNGs plus ZIP metadata, and model-install reserve.
        bound = case["size"][0] * case["size"][1] * 3 * 78 * 1.05 + 2*1024**2
        if location is None:
            remaining = (len(cases)-index+1) * bound
            location = scratch if shutil.disk_usage(scratch).free > reserve+remaining else fallback
            print(f"ATTACK STORAGE: {location}; conservative remaining bound {remaining/1024**3:.1f} GiB", flush=True)
        if location == scratch and shutil.disk_usage(scratch).free < reserve+bound:
            location = fallback
            print("SSD reserve reached: remaining ZIPs go to Drive temporary storage", flush=True)
        folder = cache_root(location, job["identity"], case["method"])
        archive = folder / (case["id"] + ".zip")
        partial = archive.with_suffix(".zip.partial")
        if sha256(inside(bank, case["output"])) != case["carrier_sha256"]:
            raise EvidenceError("Carrier changed since preflight")
        with Image.open(inside(bank, case["output"])) as image:
            carrier = np.array(image.convert("RGB"))
        entries = {}
        # A partial ZIP is never resumable evidence. Only this owned partial is overwritten.
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as zipped:
            for attack in ATTACKS[1:]:
                attacked, params = apply(carrier, attack, case["image_id"], case["replicate"])
                data = png_bytes(attacked)
                name = member_name(case, attack)
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                zipped.writestr(info, data)
                entries[attack["id"]] = {"member": name, "sha256": hashlib.sha256(data).hexdigest(),
                                          "parameters": params}
            info = zipfile.ZipInfo("manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
            zipped.writestr(info, json.dumps(entries, sort_keys=True).encode())
        with partial.open("r+b") as stream:
            os.fsync(stream.fileno())
        checksum = sha256(partial)
        if prior and (prior["sha256"] != checksum or prior["entries"] != entries):
            raise EvidenceError("Regenerated attacks differ from previous evidence")
        os.replace(partial, archive)
        manifest = dict(case_key=case["key"], archive=str(archive), sha256=checksum,
                        bytes=archive.stat().st_size, entries=entries)
        write_json(receipt, manifest)
        manifests.append(manifest)
        print(f"SAVED ATTACKS {index}/{len(cases)} {case['id']}: 78 PNGs, {manifest['bytes']/1024**2:.1f} MiB", flush=True)
    write_json(root / job["model"]["id"] / "packages-ready.json",
               {"case_keys": [c["key"] for c in cases], "manifests": manifests})


def saved_member(archive, entry):
    data = archive.read(entry["member"])
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise EvidenceError("Serialized attack image checksum mismatch")
    return data
