"""Low-memory controller: one isolated method at a time, durable per-task results."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from wmbench.embedding_bank import freeze_inputs, _log_tail
from wmbench.environment import prepare_model, worker_environment
from wmbench.io import atomic_bytes, digest, read_json, sha256, write_csv, write_json

from .contracts import MODES, POLICY, capacity, models_for_trial
from .storage import EvidenceError, cached, disjoint, freeze, inside
from .worker import record_path, signature, task_id


def context(project, settings):
    source, output, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(source, output, scratch)
    if (output / "bank.json").exists():
        raise ValueError("This is a legacy bank. Choose a NEW full-payload output folder.")
    n = settings.get("n_images", 5)
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("n_images must be 1..100")
    models = models_for_trial(project)
    selected = list(models) if settings.get("models", "all") == "all" else settings["models"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - models.keys():
        raise ValueError("Select unique IDs from the 32-method catalogue")
    # This function takes a read-only fast branch when the old frozen manifest exists.
    if not (source / "_inputs/manifest.json").is_file():
        raise FileNotFoundError(f"Original bank _inputs not found: {source}. Do not redownload replacement COCO JPEGs.")
    data = freeze_inputs(Path("__unused_when_manifest_exists__"), source)
    identity = {
        "policy": POLICY, "dataset": data["content_hash"], "models": models,
        "source_code": digest({p.relative_to(project / "src").as_posix(): sha256(p)
                               for p in sorted((project / "src").rglob("*.py"))}),
    }
    freeze(output / "trial.json", identity)
    # Freeze all 100 inputs once; N controls a stable prefix, not identity.
    for row in data["images"]:
        target, origin = inside(output, row["path"]), inside(source, row["path"])
        if target.exists() and sha256(target) != row["sha256"]:
            raise EvidenceError(f"Copied original changed: {target}")
        if not target.exists():
            atomic_bytes(target, origin.read_bytes())
    freeze(output / "_inputs/manifest.json", data)
    legacy_manifest = source / "_inputs/legacy_manifest.csv"
    target_manifest = output / "_inputs/legacy_manifest.csv"
    if target_manifest.exists() and sha256(target_manifest) != sha256(legacy_manifest):
        raise EvidenceError("Copied source manifest changed")
    if not target_manifest.exists():
        atomic_bytes(target_manifest, legacy_manifest.read_bytes())
    freeze(output / "_inputs/payload_contracts.json", {name: capacity(m) for name, m in models.items()})
    (output / "_reports").mkdir(exist_ok=True)
    return {"output": output, "scratch": scratch, "bank": source, "models": models,
            "selected": selected, "images": data["images"][:n], "identity": digest(identity)}


def complete(root, model, images):
    if not capacity(model)["information_bits"]:
        return False
    for image in images:
        for replicate in range(1, 6):
            stem, key = task_id(image, replicate), signature(model, image, replicate)
            for mode in MODES:
                row = cached(root, record_path(root, model, stem, mode), key)
                if row is None or row["status"] not in ("saved", "unmatched"):
                    return False
                if row["status"] == "saved":
                    decoded = cached(root, record_path(root, model, stem, mode + "_decode"), key)
                    if decoded is None or decoded["status"] not in ("measured", "rejected"):
                        return False
        if model["id"] in ("fin", "hidden_benchmark", "distseal_posthoc_dcae"):
            if not cached(root, root / model["id"] / "_meta/diagnostics" / task_id(image, 1) / "complete.json",
                          signature(model, image, 1)):
                return False
    return True


def asset_sources(job, name, relative):
    source = Path(job["bank"])
    supplied = job["settings"].get("manual_assets_root")
    roots = ([Path(supplied)] if supplied else []) + [
        source / "_manual_assets",
        source.parent / "watermarking_embedding_diagnostics_recovery_v1/author_assets",
        source.parent / "watermarking_embedding_diagnostics_v1/author_assets",
    ]
    aliases = [relative, "last.ckpt"] if name == "sepwater" else [relative]
    return [root / name / alias for root in roots for alias in aliases]


def ensure_manual_assets(job):
    """Reuse known author files read-only; download only two already audited releases."""
    from bank_diagnostics.assets import fetch, OBSERVED_SHA256

    model, output = job["model"], Path(job["output"])
    name = model["id"]
    manual = output / "_assets"
    known = {"editguard": ("degrade.pth", "1fAC2EIrMfPKuQa_DdYdxmUwLBbbsTJXC"),
             "sepwater": ("last.ckpt", "17RXXzQ-rMGJ7Xd6e-Az7xeylMj2r0iQS")}
    for item in model.get("manual_assets", []):
        relative = item["path"]
        target = inside(manual / name, relative)
        if not target.is_file():
            origin = next((p for p in asset_sources(job, name, relative) if p.is_file()), None)
            if origin is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                shutil.copy2(origin, temporary)
                if sha256(origin) != sha256(temporary):
                    raise EvidenceError("Author weight copy verification failed")
                temporary.replace(target)
                write_json(target.with_name(target.name + ".origin.json"), {"source": str(origin), "sha256": sha256(target)})
            elif name in known and job["settings"].get("download_author_assets", True):
                filename, identifier = known[name]
                downloaded = fetch(manual, f"{name}/{filename}", identifier)
                if downloaded != target:
                    shutil.copy2(downloaded, target)
            else:
                return f"Missing verified author asset: {target}; {item.get('url', '')}"
        if name in known:
            key = f"{name}/{known[name][0]}"
            if sha256(target) != OBSERVED_SHA256[key]:
                raise EvidenceError(f"Weight differs from the previously audited release: {target}")
        destination = inside(Path(job["slot"]) / "assets" / name, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
    return None


def prepare(job):
    from .compat import constrained_builds

    model = job["model"]
    root, slot, project = (Path(job[k]) for k in ("output", "slot", "project"))
    meta = root / model["id"] / "_meta"
    reason = ensure_manual_assets(job)
    if reason:
        write_json(meta / "prepared.json", {"status": "blocked", "reason": reason})
        return
    for filename in ("requirements.lock", "request.json"):
        old = meta / "environment" / filename
        target = slot / "locks" / model["id"] / filename
        if old.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
    with constrained_builds(slot / "locks" / model["id"] / "requirements.lock"):
        prepared = prepare_model(project, slot, model, job["settings"].get("device", "cuda"))
    write_json(meta / "prepared.json", {"status": "ready", **prepared})
    write_json(meta / "executions" / f"{job['session']}_prepared.json", prepared)


def cleanup(slot, scratch, owner):
    slot, scratch = slot.resolve(), scratch.resolve()
    if slot.parent != scratch / "slots" or slot.is_symlink():
        raise EvidenceError("Unsafe scratch slot")
    if read_json(slot / ".full-payload-owner.json") != {"owner": owner}:
        raise EvidenceError("Scratch ownership mismatch")
    for name in ("environments", "assets", "sources", "cache", "tmp", "locks"):
        target = slot / name
        if target.is_symlink() or not target.resolve().is_relative_to(slot):
            raise EvidenceError(f"Unsafe scratch cleanup target: {target}")
        if target.exists():
            shutil.rmtree(target)


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process

    ctx = context(project, settings)
    root, scratch = ctx["output"], ctx["scratch"]
    started = time.monotonic()
    budget = float(settings.get("max_hours", 16)) * 3600
    limits = {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 150, "telemetry": True}
    with controller_lease(root / "_reports") as owned:
        report(root, settings)
        for index, name in enumerate(ctx["selected"], 1):
            owned()
            model = ctx["models"][name]
            meta = root / name / "_meta"
            state = meta / "status.json"
            if complete(root, model, ctx["images"]):
                print(f"[{index}/32] REUSE {name}: all sealed tasks verified", flush=True)
                continue
            if time.monotonic() - started >= budget:
                print("Session budget reached. Re-run to resume saved tasks.", flush=True)
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 30 * 1024**3:
                print("STOP: less than 30 GiB free scratch. Failed environments are retained; "
                      "use a fresh runtime or inspect them before continuing. Drive results are safe.", flush=True)
                break
            reason = None
            if not capacity(model)["information_bits"]:
                reason = "No verified capacity/adapter for Camera-Shooting Instance"
            elif model.get("backend") == "matlab" and not shutil.which("matlab"):
                reason = "Licensed MATLAB + Image Processing/Communications Toolboxes required; not provided by Colab"
            if reason:
                write_json(state, {"status": "blocked", "reason": reason})
                print(f"[{index}/32] BLOCKED {name}: {reason}", flush=True)
                report(root, settings)
                continue
            slot = scratch / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".full-payload-owner.json", {"owner": ctx["identity"]})
            session = str(time.time_ns())
            job = {"model": model, "project": str(project), "output": str(root), "slot": str(slot),
                   "bank": str(ctx["bank"]), "images": ctx["images"], "settings": settings, "session": session}
            job_path = meta / "jobs" / f"{session}.json"
            write_json(job_path, job)
            write_json(state, {"status": "running", "session": session})
            env = worker_environment(project, slot, model)
            env.update(UV_CACHE_DIR=str(slot / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
            failed = False
            try:
                for stage in ("prepare", "work"):
                    owned()
                    remaining = budget - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Session budget reached")
                    python = sys.executable if stage == "prepare" else read_json(meta / "prepared.json")["python"]
                    log = meta / "logs" / f"{session}_{stage}.log"
                    seconds = (10800 if name == "chunkyseal" else 5400) if stage == "prepare" else 21600
                    print(f"[{index}/32] {stage.upper()} {name}: {capacity(model)['information_bits']} information bits", flush=True)
                    result = monitored_process(
                        [sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                         "-m", "wmbench_full_payload.runner", stage, "--settings", str(job_path)],
                        env, log, slot, min(seconds, remaining), limits, assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{stage}.json", result)
                    if result["exit_code"] or result["reason"]:
                        print(_log_tail(log), flush=True)
                        raise RuntimeError(result["reason"] or f"{stage} exit={result['exit_code']}; see {log}")
                    if stage == "prepare" and read_json(meta / "prepared.json")["status"] == "blocked":
                        write_json(state, read_json(meta / "prepared.json"))
                        break
                else:
                    write_json(state, {"status": "processed", "session": session})
            except EvidenceError:
                raise
            except Exception as error:
                failed = True
                write_json(state, {"status": "error", "reason": str(error), "session": session})
                print(f"FAILED {name}: {error}", flush=True)
            finally:
                lock_root = slot / "locks" / name
                if lock_root.exists():
                    shutil.copytree(lock_root, meta / "environment", dirs_exist_ok=True)
                report(root, settings)
                if settings.get("cleanup", True) and not failed:
                    cleanup(slot, scratch, ctx["identity"])
                elif failed:
                    print(f"Retained failed scratch for retry: {slot}", flush=True)
    report(root, settings)


def report(root, settings):
    """No model loading, network access or full weight/image hashing in reports."""
    identity = read_json(root / "trial.json")
    images = read_json(root / "_inputs/manifest.json")["images"][:settings.get("n_images", 5)]
    selected = list(identity["models"]) if settings.get("models", "all") == "all" else settings["models"]
    rows, summary = [], []
    for name, model in identity["models"].items():
        spec = capacity(model)
        for image in images:
            for replicate in range(1, 6):
                stem = task_id(image, replicate)
                for mode in MODES:
                    path = record_path(root, model, stem, mode)
                    decode_path = record_path(root, model, stem, mode + "_decode")
                    encoded = read_json(path) if path.is_file() else {}
                    decoded = read_json(decode_path) if decode_path.is_file() else {}
                    rows.append({
                        "model": name, "image_id": image["image_id"], "replicate": replicate, "mode": mode,
                        "information_bits": spec["information_bits"], "adapter_bits": spec["adapter_bits"],
                        "interface": spec["interface"], "capacity_kind": spec["capacity_kind"],
                        "image_status": encoded.get("status", "pending"),
                        "decode_status": decoded.get("status", "pending"),
                        "ber": decoded.get("ber"), "exact": decoded.get("exact"),
                        "psnr": encoded.get("psnr"), "alpha": encoded.get("matching", {}).get("alpha"),
                        "reason": encoded.get("reason", decoded.get("reason", "")),
                    })
        own = [row for row in rows if row["model"] == name]
        native = [row for row in own if row["mode"] == "native"]
        measured = [row for row in native if row["ber"] is not None]
        errors = sum(row["decode_status"] == "error" or row["image_status"] == "error" for row in own)
        terminal = sum(row["image_status"] == "unmatched" or row["decode_status"] in ("measured", "rejected") for row in own)
        state_path = root / name / "_meta/status.json"
        state = read_json(state_path) if state_path.is_file() else {}
        exact = sum(row["exact"] is True for row in native)
        auxiliary_expected = len(images) if name in ("fin", "hidden_benchmark", "distseal_posthoc_dcae") else 0
        auxiliary_complete = 0
        if auxiliary_expected:
            for image in images:
                path = root / name / "_meta/diagnostics" / task_id(image, 1) / "complete.json"
                if path.is_file() and read_json(path).get("status") == "measured":
                    auxiliary_complete += 1
        if name not in selected:
            conclusion = "not_selected"
        elif terminal == len(own):
            if auxiliary_complete != auxiliary_expected:
                conclusion = "auxiliary_diagnostics_incomplete"
            else:
                conclusion = "native_exact_canary" if exact == len(native) else "completed_with_native_bit_errors_or_rejections"
        else:
            conclusion = state.get("status", "pending")
            if conclusion == "processed":
                conclusion = "technical_errors" if errors else "incomplete"
        summary.append({
            "model": name, "information_bits": spec["information_bits"], "adapter_bits": spec["adapter_bits"],
            "native_exact": exact, "native_expected": len(native), "native_measured": len(measured),
            "native_ber": sum(row["ber"] for row in measured) / len(measured) if measured else None,
            "saved_png": sum(row["image_status"] == "saved" for row in own),
            "matched_unavailable": sum(row["image_status"] == "unmatched" for row in own),
            "technical_errors": errors, "terminal_modes": terminal, "expected_modes": len(own),
            "auxiliary_complete": auxiliary_complete, "auxiliary_expected": auxiliary_expected,
            "status": conclusion, "reason": state.get("reason", ""),
        })
    folder = root / "_reports"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "results.csv", rows)
    write_csv(folder / "models.csv", summary)
    mode_summary = []
    for name in identity["models"]:
        for mode in MODES:
            group = [row for row in rows if row["model"] == name and row["mode"] == mode]
            measured = [row for row in group if row["ber"] is not None]
            mode_summary.append({"model": name, "mode": mode, "expected": len(group),
                "measured": len(measured), "exact": sum(row["exact"] is True for row in group),
                "mean_ber": sum(row["ber"] for row in measured) / len(measured) if measured else None,
                "unmatched": sum(row["image_status"] == "unmatched" for row in group),
                "rejected": sum(row["decode_status"] == "rejected" for row in group),
                "technical_errors": sum(row["image_status"] == "error" or row["decode_status"] == "error" for row in group)})
    write_csv(folder / "modes.csv", mode_summary)
    write_json(folder / "summary.json", {"n_images": len(images), "models": summary,
               "all_32_native_exact": all(row["native_exact"] == row["native_expected"] for row in summary),
               "planned_png": 32 * len(images) * 5 * 4, "note": "Technical completion is not zero BER or a capacity theorem"})
    lines = ["# Full-payload canary", "", f"{len(images)} images × 5 messages × 32 entries × 4 modes. No channel attacks.", "",
             "Native exact recovery and technical completion are separate. Nonzero BER is preserved, not retried.",
             "Unmatched PSNR targets are unavailable, not fake successful images. Raw and post-ECC results are labelled in results.csv.", "",
             "| Method | Information / adapter bits | Native exact | Native BER | Saved PNG | Unmatched | State |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        ber = "—" if row["native_ber"] is None else f"{row['native_ber']:.6f}"
        lines.append(f"| {row['model']} | {row['information_bits']} / {row['adapter_bits']} | "
                     f"{row['native_exact']}/{row['native_expected']} | {ber} | {row['saved_png']}/{row['expected_modes']} | "
                     f"{row['matched_unavailable']} | {row['status']} |")
    lines += ["", "## Blockers and next steps", "",
              "Inspect every nonzero Native BER, decode exception, unmatched target, and unresolved prerequisite before the 100-image stage.",
              "A five-image pass is an integration check, not a guarantee of 100-image reliability. Do not reduce payloads automatically.",
              "DWT4096 and SSL2048 test structural limits, not previously demonstrated error-free rates.",
              "FIN/HiDDeN resolution and DistSeal autoencoder controls are under each method's _meta/diagnostics (replicate 1, each image).",
              "Float-before-PNG observations are available only where the adapter exposes that boundary; unavailable is not a pass.", ""]
    for row in summary:
        if row["reason"]:
            lines += [f"- {row['model']}: {row['reason']}"]
    atomic_bytes(folder / "REPORT.md", ("\n".join(lines) + "\n").encode())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "work", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    if args.action == "prepare":
        prepare(settings)
    elif args.action == "work":
        from .worker import work
        work(settings)
    elif args.action == "report":
        report(Path(settings["output"]), settings)
    else:
        project = Path(__file__).resolve().parents[2]
        run(project, settings)


if __name__ == "__main__":
    main()
