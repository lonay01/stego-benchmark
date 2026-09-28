"""Sequential isolated environments with durable, independently resumed tasks."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from wmbench.embedding_bank import freeze_inputs, _log_tail
from wmbench.environment import worker_environment
from wmbench.io import atomic_bytes, digest, read_json, sha256, write_csv, write_json
from wmbench_full_payload.runner import prepare, cleanup
from wmbench_full_payload.storage import EvidenceError, disjoint, freeze, inside

from .contracts import POLICY, capacity, models_for_trial
from .worker import complete, record_path, task_id


def context(project, settings):
    source, output, scratch = (Path(settings[k]).expanduser().resolve() for k in ("bank", "output", "scratch"))
    disjoint(source, output, scratch)
    if (output / "bank.json").exists():
        raise ValueError("Select a NEW canary directory, not the production bank")
    n = settings.get("n_images", 5)
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("n_images must be 1..100; use 5 for the canary")
    models = models_for_trial(project)
    selected = list(models) if settings.get("models", "all") == "all" else settings["models"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - models.keys():
        raise ValueError("Select unique known model IDs")
    if not (source / "_inputs/manifest.json").is_file():
        raise FileNotFoundError(f"Set BANK_ROOT to the existing bank with _inputs/manifest.json: {source}")
    data = freeze_inputs(Path("__unused_with_existing_manifest__"), source)
    identity = {"policy": POLICY, "dataset": data["content_hash"], "models": models,
                "source_code": digest({p.relative_to(project / "src").as_posix(): sha256(p)
                                       for p in sorted((project / "src").rglob("*.py"))})}
    freeze(output / "trial.json", identity)
    for image in data["images"]:
        origin, target = inside(source, image["path"]), inside(output, image["path"])
        if target.exists() and sha256(target) != image["sha256"]:
            raise EvidenceError(f"Copied original changed: {target}")
        if not target.exists():
            atomic_bytes(target, origin.read_bytes())
    freeze(output / "_inputs/manifest.json", data)
    original_csv, target_csv = source / "_inputs/legacy_manifest.csv", output / "_inputs/legacy_manifest.csv"
    if target_csv.exists() and sha256(target_csv) != sha256(original_csv):
        raise EvidenceError("Copied legacy manifest changed")
    if not target_csv.exists():
        atomic_bytes(target_csv, original_csv.read_bytes())
    freeze(output / "_inputs/contracts.json", {k: {"payload": capacity(v), "geometry": v["native_geometry"],
           "exclusion": v.get("canary_exclusion")} for k, v in models.items()})
    (output / "_reports").mkdir(exist_ok=True)
    return {"output": output, "scratch": scratch, "bank": source, "models": models,
            "selected": selected, "images": data["images"][:n], "identity": digest(identity)}


def run(project, settings):
    from wmbench.colab_canary import controller_lease, monitored_process

    ctx = context(project, settings)
    root, scratch = ctx["output"], ctx["scratch"]
    started, budget = time.monotonic(), float(settings.get("max_hours", 16)) * 3600
    if not 0 < budget <= 7 * 86400:
        raise ValueError("max_hours must be positive and <=168")
    limits = {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 150, "telemetry": True}
    with controller_lease(root / "_reports") as owned:
        report(root, settings)
        for index, name in enumerate(ctx["selected"], 1):
            owned()
            model = ctx["models"][name]
            meta, state = root / name / "_meta", root / name / "_meta/status.json"
            label = f"[{index}/{len(ctx['selected'])}] {name}"
            if model.get("canary_exclusion"):
                write_json(state, {"status": "excluded", "reason": model["canary_exclusion"]})
                print(f"{label} EXCLUDED: {model['canary_exclusion']}", flush=True)
                continue
            if complete(root, model, ctx["images"]):
                print(f"{label} REUSE: all PNG/decoder checkpoints verified", flush=True)
                write_json(state, {"status": "verified", "n_images": len(ctx["images"])})
                continue
            if time.monotonic() - started >= budget:
                print("Session budget reached. Re-run the same notebook to resume.", flush=True)
                break
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < 30 * 1024**3:
                print("STOP: <30 GiB free local SSD; results on Drive are safe.", flush=True)
                break
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
                    print(f"{label} {stage.upper()}: {model['native_geometry']['width']}px / "
                          f"{capacity(model)['information_bits']} information bits", flush=True)
                    result = monitored_process([sys.executable, "-m", "wmbench.process_guard", "--", str(python),
                        "-m", "wmbench_native_canary.runner", stage, "--settings", str(job_path)],
                        env, log, slot, min(seconds, remaining), limits, assert_owned=owned)
                    write_json(meta / "timings" / f"{session}_{stage}.json", result)
                    if result["exit_code"] or result["reason"]:
                        print(_log_tail(log), flush=True)
                        raise RuntimeError(result["reason"] or f"{stage} exit={result['exit_code']}; {log}")
                    if stage == "prepare" and read_json(meta / "prepared.json")["status"] == "blocked":
                        write_json(state, read_json(meta / "prepared.json"))
                        break
                else:
                    if not complete(root, model, ctx["images"]):
                        raise EvidenceError("Worker returned without a complete verified round-trip set")
                    write_json(state, {"status": "verified", "session": session, "n_images": len(ctx["images"])})
            except EvidenceError:
                failed = True
                raise
            except Exception as error:
                failed = True
                write_json(state, {"status": "error", "reason": str(error), "session": session})
                print(f"{label} ERROR: {error}", flush=True)
            finally:
                locks = slot / "locks" / name
                if locks.exists():
                    shutil.copytree(locks, meta / "environment", dirs_exist_ok=True)
                report(root, settings)
                if settings.get("cleanup", True) and not failed:
                    cleanup(slot, scratch, ctx["identity"])
                elif failed:
                    print(f"Failed scratch retained for diagnosis: {slot}", flush=True)
    report(root, settings)


def mean(values):
    values = [v for v in values if v is not None]
    if "Infinity" in values:
        return "Infinity"
    return sum(values) / len(values) if values else None


def report(root, settings):
    """Read small records only; do not load a model, download or hash its weights."""
    identity = read_json(root / "trial.json")
    images = read_json(root / "_inputs/manifest.json")["images"][:settings.get("n_images", 5)]
    selected = list(identity["models"]) if settings.get("models", "all") == "all" else settings["models"]
    rows, summary = [], []
    for name, model in identity["models"].items():
        spec = capacity(model)
        excluded = model.get("canary_exclusion")
        own = []
        if not excluded:
            for image in images:
                for replicate in range(1, POLICY["replicates"] + 1):
                    stem = task_id(image, replicate)
                    enc_path = record_path(root, model, stem, "native")
                    dec_path = record_path(root, model, stem, "native_decode")
                    enc = read_json(enc_path) if enc_path.is_file() else {}
                    dec = read_json(dec_path) if dec_path.is_file() else {}
                    pre = enc.get("pre_serialization", {}).get("pre_png", {})
                    own.append({"model": name, "image_id": image["image_id"], "replicate": replicate,
                        "size": model["native_geometry"]["width"], "information_bits": spec["information_bits"],
                        "adapter_bits": spec["adapter_bits"], "bpp": spec["bits_per_pixel"],
                        "image_status": enc.get("status", "pending"), "decode_status": dec.get("status", "pending"),
                        "ber": dec.get("ber"), "exact": dec.get("exact"), "psnr": enc.get("psnr"),
                        "pre_png_status": pre.get("status", "pending"), "pre_png_ber": pre.get("ber"),
                        "pre_png_exact": pre.get("exact"), "pre_png_boundary": pre.get("boundary"),
                        "embed_seconds": enc.get("seconds"), "decode_seconds": dec.get("seconds"),
                        "reason": enc.get("reason", dec.get("reason", pre.get("reason", "")))})
        rows.extend(own)
        expected = len(own)
        terminal = sum(r["decode_status"] in ("measured", "rejected") for r in own)
        exact = sum(r["exact"] is True for r in own)
        state_path = root / name / "_meta/status.json"
        state = read_json(state_path) if state_path.is_file() else {}
        errors = sum(r["image_status"] == "error" or r["decode_status"] == "error" for r in own)
        pre_errors = sum(r["pre_png_status"] == "error" for r in own)
        status = state.get("status", "pending")
        if excluded:
            status = "excluded"
        elif name not in selected:
            status = "not_selected"
        elif expected and terminal == expected:
            status = "png_exact_canary" if exact == expected else "completed_with_bit_errors_or_rejections"
            if pre_errors:
                status += ";pre_png_diagnostic_error"
        elif status == "verified":
            status = "incomplete_for_requested_N"
        summary.append({"model": name, "size": model["native_geometry"]["width"],
            "information_bits": spec["information_bits"], "adapter_bits": spec["adapter_bits"],
            "bpp": spec["bits_per_pixel"], "expected": expected, "measured": terminal,
            "png_exact": exact, "png_ber": mean(r["ber"] for r in own),
            "pre_png_measured": sum(r["pre_png_status"] == "measured" for r in own),
            "pre_png_exact": sum(r["pre_png_exact"] is True for r in own),
            "pre_png_ber": mean(r["pre_png_ber"] for r in own), "pre_png_errors": pre_errors,
            "mean_psnr": mean(r["psnr"] for r in own), "technical_errors": errors,
            "status": status, "reason": excluded or state.get("reason", "")})
    report_dir = root / "_reports"
    report_dir.mkdir(exist_ok=True)
    write_csv(report_dir / "results.csv", rows)
    write_csv(report_dir / "models.csv", summary)
    active = [r for r in summary if r["model"] in selected and r["status"] != "excluded"]
    write_json(report_dir / "summary.json", {"n_images": len(images), "models": summary,
        "planned_png": sum(r["expected"] for r in active),
        "all_selected_png_exact": bool(active) and all(r["status"] == "png_exact_canary" for r in active),
        "note": "Canary only; no proof of reliable capacity, robustness, or architectural causality."})
    lines = ["# Native-resolution canary", "",
        f"{len(images)} frozen images × 5 messages. Native strength only; no attacks or matched PSNR.", "",
        "Exact recovery is counted over the whole useful message. Unavailable pre-PNG observations are not passes.",
        "Runtime shape checks and artifact hashes run in the worker/resume path; this quick report reads saved records.", "",
        "| Method | Carrier | Useful / raw bits | PNG exact | PNG BER | Before PNG exact / measured | Status |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        ber = "—" if row["png_ber"] is None else f"{row['png_ber']:.6f}"
        before = f"{row['pre_png_exact']}/{row['pre_png_measured']}" if row["pre_png_measured"] else "not measured"
        lines.append(f"| {row['model']} | {row['size']}² | {row['information_bits']} / {row['adapter_bits']} | "
                     f"{row['png_exact']}/{row['expected']} | {ber} | {before} | {row['status']} |")
    lines += ["", "## Interpretation", "",
        "- Review every exception and nonzero Native BER before expanding to 100 images. Do not drop hard messages or retry measured bit errors.",
        "- Different carriers and payloads compare configured systems, not isolated architectures or equal bit rates.",
        "- Center crops share a parent COCO image but not identical image content. No 512px residual transfer is applied externally.",
        "- Upstream wrapper-return observations may already include author-side quantization; read pre_png_boundary in results.csv.",
        "- DWT32 and SSL30 use the selected baseline settings, not the previous 4096/2048 structural-limit experiment.", ""]
    lines += [f"- {r['model']}: {r['reason']}" for r in summary if r["reason"]]
    atomic_bytes(report_dir / "REPORT.md", ("\n".join(lines) + "\n").encode())
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
        run(Path(__file__).resolve().parents[2], settings)


if __name__ == "__main__":
    main()
