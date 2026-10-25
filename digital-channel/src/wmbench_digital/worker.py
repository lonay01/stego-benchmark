"""Blind extraction in the original method's isolated Python environment."""

import hashlib
import os
from pathlib import Path
import time
import traceback
import zipfile

from wmbench.io import read_json, write_json, atomic_bytes, sha256
from wmbench_full_payload.storage import EvidenceError, freeze, inside
from wmbench_full_payload.contracts import score
from wmbench.worker import adapter_instance, runtime_info, seed_everything
from wmbench.embedding_bank import inference_identity
from .spec import ATTACKS
from .storage import result_path, read_result, save_result, complete


def equal_evidence(old, new, label):
    if old != new:
        raise EvidenceError(f"Decoder provenance differs from the bank: {label}")


def load_adapter(job):
    model, root, bank = job["model"], Path(job["output"]), Path(job["bank"])
    meta = root / model["id"] / "_meta"
    original = bank / model["id"] / "_meta"
    prepared = read_json(meta / "prepared.json")
    seed_everything(0)
    if model["engine"] == "author":
        from wmbench_bank.adapters import AuthorAdapter
        adapter = AuthorAdapter(model["author_name"], prepared["source"], root / "_author")
        name = model["author_name"]
        for filename in ("environment_request.json", "installed_packages.json", "weight_audit.json"):
            equal_evidence(read_json(bank / "_author/_meta" / name / filename),
                           read_json(root / "_author/_meta" / name / filename), filename)
        equal_evidence(sha256(bank / "_author/_meta" / name / "requirements.lock"),
                       sha256(root / "_author/_meta" / name / "requirements.lock"), "author lock")
        for source_name in ([name, "taming", "imagenetc"] if name == "rosteals" else [name]):
            equal_evidence(read_json(bank / "_author/_meta/sources" / (source_name+".json")),
                           read_json(root / "_author/_meta/sources" / (source_name+".json")), "author source")
    else:
        current = inference_identity(prepared["identity"])
        expected = read_json(original / "model_identity.json")
        if model["id"] == "meta_hidden_revisited":
            # Only the previously audited generated-pyc exception. No bank repair hooks.
            from hidden_fnns_reference import canonical_meta, meta_source_classes
            from wmbench.adapters import meta_hidden
            current, expected = canonical_meta(current), canonical_meta(expected)
            meta_hidden.load_classes = meta_source_classes
        equal_evidence(expected, current, "model/checkpoint")
        equal_evidence(read_json(original / "dependency_identity.json"),
                       {k: prepared["identity"][k] for k in ("lock_sha256", "installed_sha256")}, "dependencies")
        adapter = adapter_instance(read_json(prepared["config"]))
        if hasattr(adapter, "initialize"):
            adapter.initialize()
        from wmbench_native_canary.adapters import validate_geometry
        equal_evidence(read_json(original / "geometry.json"), validate_geometry(adapter, model), "geometry")
    if hasattr(adapter, "native_bits") and adapter.native_bits != model["native_bits"]:
        raise EvidenceError("Payload length drift")
    return adapter


def measured(adapter, png, metadata, case, rng_seed):
    start = time.monotonic()
    try:
        seed_everything(rng_seed)
        result = adapter.decode(str(png), str(metadata))
        if (result.get("details") or {}).get("ecc_valid") is False and result.get("bits") is None:
            row = dict(status="rejected", ber=None, exact=False, decoded_bits=None,
                       information_bits=case["assignment"]["capacity"]["information_bits"], details=result.get("details"))
        else:
            row = score(result, case["assignment"])
        row.pop("expected_bits", None)  # Stored once per case; never sent to decoder.
    except Exception as error:
        row = dict(status="error", reason=repr(error), traceback=traceback.format_exc(),
                   ber=None, exact=None, decoded_bits=None)
    row["seconds"] = time.monotonic()-start
    return row


def decode(job):
    root, bank = Path(job["results"]), Path(job["bank"])
    name = job["model"]["id"]
    cases = read_json(root / name / "cases.json")
    ready = read_json(root / name / "packages-ready.json")
    equal_evidence(ready["case_keys"], [c["key"] for c in cases], "attack stage boundary")
    pending = [c for c in cases if not complete(read_result(result_path(root, c), c))]
    if not pending:
        return
    adapter = load_adapter(job)
    execution = {"runtime": runtime_info(), "session": job["session"]}
    write_json(root / name / "executions" / (job["session"]+".json"), execution)
    temporary = Path(job["slot"]) / "tmp/digital-decode"
    temporary.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, case in enumerate(pending, 1):
        if time.time() >= job["deadline"]:
            raise TimeoutError("Session budget reached; durable partial records retained")
        path = result_path(root, case)
        document = read_result(path, case) or dict(case_key=case["key"], case=case, rows={})
        prior = document["rows"]
        last_checkpoint = time.monotonic()
        # Each resumed case checks Clean again; decoding drift is not a robustness result.
        rng_seed = int(case["signature"][:8], 16)
        metadata = inside(bank, case["decoder_metadata"])
        blind = read_json(metadata)
        for field in ("expected_bits", "payload", "full_payload"):
            if isinstance(blind.get(field), list) and any(blind[field]):
                raise EvidenceError("Reference payload unexpectedly present in decoder metadata")
        clean = measured(adapter, inside(bank, case["output"]), metadata, case, rng_seed)
        for field in ("status", "decoded_bits", "ber", "exact"):
            if clean.get(field) != case["clean"].get(field):
                write_json(root / name / "clean-drift" / (case["id"]+".json"),
                           {"fresh": clean, "bank": case["clean"], "execution": execution})
                raise EvidenceError("Fresh Clean differs from saved bank control; no attack scores accepted")
        if "clean" not in prior:
            prior["clean"] = {**clean, "execution": execution, "image_sha256": case["carrier_sha256"]}
        manifest = read_json(root / name / "packages" / (case["id"]+".json"))
        if manifest["case_key"] != case["key"] or sha256(manifest["archive"]) != manifest["sha256"]:
            raise EvidenceError("Attack ZIP identity mismatch")
        with zipfile.ZipFile(manifest["archive"]) as archive:
            for number, attack in enumerate(ATTACKS[1:], 1):
                if time.time() >= job["deadline"]:
                    save_result(path, document)
                    raise TimeoutError("Budget reached during decode; rerun to resume")
                old = prior.get(attack["id"])
                if old and old["status"] in {"measured", "rejected"}:
                    continue  # Never retry a valid but poor BER.
                entry = manifest["entries"][attack["id"]]
                data = archive.read(entry["member"])
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise EvidenceError("Attack PNG checksum mismatch")
                target = temporary / "attack.png"
                atomic_bytes(target, data)
                row = measured(adapter, target, metadata, case, rng_seed)
                row.update(parameters=entry["parameters"], image_sha256=entry["sha256"], execution=execution)
                if old:
                    row["previous_technical_attempts"] = [*old.get("previous_technical_attempts", []),
                                                          {k: v for k, v in old.items() if k != "previous_technical_attempts"}]
                prior[attack["id"]] = row
                failures = failures+1 if row["status"] == "error" else 0
                if time.monotonic()-last_checkpoint >= 60 or failures:
                    save_result(path, document)
                    last_checkpoint = time.monotonic()
                if number % 10 == 0 or failures:
                    print(f"DECODE {index}/{len(pending)} {case['id']}: {number}/78 attacks, {row['status']}", flush=True)
                if failures >= 3:
                    raise RuntimeError("Three consecutive technical decoding errors; inspect logs and resume")
        save_result(path, document)
        print(f"COMMITTED {index}/{len(pending)} {case['id']}: {len(prior)}/79", flush=True)
    # Model object dies with this subprocess; VRAM is released between methods.
