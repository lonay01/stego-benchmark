"""Resumable diagnostic orchestration with a read-only historical image bank."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time

from wmbench.embedding_bank import POLICY, bank_registry, valid_record
from wmbench.environment import prepare_model, worker_environment, code_fingerprint
from wmbench.io import atomic_bytes, digest, read_json, sha256, write_json, write_csv
from wmbench.matching import load_rgb, psnr
from .core import BLOCKED, RECHECK, cached, freeze, inside, scalar_audit, separate_roots


def context(project: Path, settings: dict) -> dict:
    bank, output, scratch = (Path(settings[k]).resolve() for k in ("bank", "output", "scratch"))
    separate_roots(bank, output, scratch)
    if (bank / "bank.json").is_file():
        historical = read_json(bank / "bank.json")
        if historical.get("policy") != POLICY:
            raise ValueError("Historical bank policy differs from this diagnostic release")
    manifest = read_json(bank / "_inputs/manifest.json")
    images = manifest["images"][:int(settings.get("n_images", 5))]
    if len(images) != int(settings.get("n_images", 5)) or not images:
        raise ValueError("Requested number of existing historical images is unavailable")
    ids = [int(i["image_id"]) for i in images]
    if ids != sorted(set(ids)):
        raise ValueError("Historical manifest has duplicate or unordered IDs")
    for image in images:
        if sha256(inside(bank, image["path"])) != image["sha256"]:
            raise ValueError("Historical original checksum mismatch")
    for p in (output, scratch):
        p.mkdir(parents=True, exist_ok=True)
    selected = settings.get('models', list(RECHECK) + (list(BLOCKED) if settings.get('check_blocked', True) else []))
    if not selected or len(set(selected)) != len(selected) or any(n not in RECHECK + BLOCKED for n in selected):
        raise ValueError('Invalid diagnostic model selection')
    specification = {"schema": "bank-diagnostics-v1", "production_source": code_fingerprint(project),
                     "diagnostic_source": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                     "images": images, "policy": POLICY, "registry": digest(bank_registry(project)),
                     "models": selected, "check_blocked": settings.get("check_blocked", True),
                     "dependency_repairs": settings.get('dependency_repairs', False)}
    freeze(output / "diagnostic_spec.json", specification)
    return {"bank": bank, "output": output, "scratch": scratch, "images": images,
            "identity": digest(specification), "models": bank_registry(project), "selected": selected}


def audit_matching(ctx: dict):
    output, bank = ctx["output"], ctx["bank"]
    rows = []
    for model in ctx["models"]:
        for image in ctx["images"]:
            stem = f"{int(image['image_id']):012d}"
            for target in POLICY["targets"]:
                path = bank / model / "_meta/records" / f"{stem}_matched_{target}.json"
                if not path.is_file():
                    continue
                record = valid_record(bank, path)
                if record.get("status") != "match_not_found_within_policy":
                    continue
                native_record = valid_record(bank, path.with_name(f"{stem}_native.json"))
                if not native_record or native_record["status"] != "ok":
                    raise ValueError(f"Missing valid Native for matched audit: {path}")
                native = inside(bank, native_record["output"])
                source = inside(bank, image["path"])
                signature = digest({"run": ctx["identity"], "record": record,
                                    "native": sha256(native), "source": sha256(source)})
                result_path = output / "matching" / f"{model}_{stem}_{target}.json"
                result = cached(result_path, signature)
                if result is None:
                    result = scalar_audit(load_rgb(source), load_rgb(native), target,
                                          POLICY["tolerance"], POLICY["max_alpha"])
                    result.update(signature=signature, artifacts=[], model=model, image_id=image["image_id"],
                                  stored_alpha=record["alpha"], stored_psnr=record["psnr"])
                    # Independently verify a found scalar on the full image grid.
                    if result["within_policy"]:
                        import numpy as np
                        x = load_rgb(source).astype(np.float64)
                        y = load_rgb(native).astype(np.float64)
                        z = np.rint(np.clip(x + result["audit_alpha"] * (y - x), 0, 255)).astype(np.uint8)
                        actual = psnr(load_rgb(source), z)
                        if abs(actual - target) > POLICY["tolerance"]:
                            raise RuntimeError("Histogram and full-grid scalar search disagree")
                        result["verified_full_grid_psnr"] = actual
                    write_json(result_path, result)
                rows.append(result)
                print(f"PSNR {model} {stem} {target}: {result['finding']}", flush=True)
    write_json(output / "matching/summary.json", {"checked": len(rows),
               "within_policy": sum(r["within_policy"] for r in rows), "rows": rows})


def diagnostic_model(model: dict, assets: Path, bank: Path) -> tuple[dict, str | None]:
    """Overrides are confined to diagnostics, recorded in each prepared identity."""
    result = copy.deepcopy(model)
    name = model["id"]
    if name == "camera_shooting_instance":
        return result, "Author premodel.zip requires Baidu access; no validated instance-level payload adapter. Not runnable by installing packages alone."
    if name == "fang_screen" and not shutil.which("matlab"):
        return result, "Requires an existing licensed MATLAB with Image Processing and Communications Toolboxes. Cannot install a license automatically."
    if name == "omniguard":
        manual = bank / "_manual_assets/omniguard/omniguard-copyright-verified.pth"
        dest = assets / "omniguard/omniguard-copyright-verified.pth"
        if manual.is_file() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manual, dest)
        if not dest.is_file():
            return result, "Public checkpoint.zip is inventoried separately; a verified 100-bit copyright checkpoint is still required. Localization weights are not a replacement."
    if name == "sepwater":
        source = assets / "sepwater/last.ckpt"
        target = assets / "sepwater/sepwater.ckpt"
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
        # The original strict adapter tests whether this public release is complete.
        # Do not fabricate missing ae/control/decoder weights or silently merge checkpoints.
    for item in result.get("manual_assets", []):
        if not inside(assets / name, item["path"]).is_file():
            return result, f"Author asset missing: {name}/{item['path']}"
    return result, None


def prepare(job: dict):
    project, slot, out = (Path(job[k]) for k in ("project", "slot", "output"))
    model = job["model"]
    for item in model.get("manual_assets", []):
        source = inside(Path(job["assets"]) / model["id"], item["path"])
        target = inside(slot / "assets" / model["id"], item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Reuse the resolved dependency lock, but never an old 'installed' marker.
    prior = out / "environment" / model["id"]
    for filename in ("requirements.lock", "request.json"):
        old, new = prior / filename, slot / "locks" / model["id"] / filename
        if old.is_file() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    if model['id'] == 'editguard' and model.get('build_requirements') == ['basicsr==1.4.2']:
        from .dependency_fixes import constrained_builds
        with constrained_builds(slot / 'locks' / model['id'] / 'requirements.lock'):
            prepared = prepare_model(project, slot, model, job["device"])
    else:
        prepared = prepare_model(project, slot, model, job["device"])
    write_json(out / "prepared.json", prepared)
    write_json(out / "execution_history" / f"{time.time_ns()}_prepared.json", prepared)


def cleanup_slot(slot: Path, scratch: Path, identity: str):
    slot, scratch = slot.resolve(), scratch.resolve()
    if slot.parent != scratch / "slots" or slot.is_symlink():
        raise ValueError("Refusing cleanup outside diagnostic scratch slots")
    if read_json(slot / ".diagnostic-owner.json") != {"identity": identity}:
        raise ValueError("Scratch ownership mismatch")
    for name in ("assets", "sources", "environments", "cache", "locks", "tmp"):
        path = slot / name
        if path.is_symlink() or not path.resolve().is_relative_to(slot):
            raise ValueError(f"Refusing unsafe cleanup: {path}")
        if path.exists():
            shutil.rmtree(path)


def run(project: Path, settings: dict):
    from wmbench.colab_canary import controller_lease, monitored_process
    ctx = context(project, settings)
    output, scratch = ctx["output"], ctx["scratch"]
    limits = {"min_free_disk_gib": 20, "min_free_ram_gib": 2, "max_slot_gib": 150, "telemetry": True}
    started = time.monotonic()
    output.joinpath("_control").mkdir(exist_ok=True)
    with controller_lease(output / "_control") as owned:
        if settings.get('audit_matching', True):
            audit_matching(ctx)
        report(output)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(project / "src"), str(project / "diagnostics")))
        assets = output / "author_assets"
        previous = settings.get('previous_diagnostics')
        if previous:
            previous = Path(previous).resolve()
            separate_roots(previous, output, scratch)
            # Copy only the two already downloaded assets. Never modify the old run.
            for relative in ('editguard/degrade.pth', 'sepwater/last.ckpt'):
                source, dest = inside(previous / 'author_assets', relative), inside(assets, relative)
                if source.is_file() and not dest.exists():
                    print(f'Reuse author weights: {relative}', flush=True)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temporary = dest.with_name(dest.name + '.part')
                    shutil.copy2(source, temporary)
                    if sha256(source) != sha256(temporary):
                        raise ValueError('Copied author asset checksum mismatch')
                    temporary.replace(dest)
            write_json(output / 'previous_run.json', {'read_only_path': str(previous)})
        if settings.get("check_blocked", True):
            asset_slot = scratch / "asset_downloads"
            asset_slot.mkdir(exist_ok=True)
            cmd = [sys.executable, "-m", "wmbench.process_guard", "--", sys.executable,
                   "-m", "bank_diagnostics.assets", str(assets), str(ctx["bank"])]
            if not settings.get("download_author_assets", True):
                cmd.append("--no-download")
            stats = monitored_process(cmd, env, output / "logs/asset_downloads.log", asset_slot,
                                      3600, limits, assert_owned=owned)
            write_json(output / "asset_download_status.json", stats)
        names = ctx['selected']
        for index, name in enumerate(names, 1):
            owned()
            remaining = float(settings.get("max_hours", 12)) * 3600 - (time.monotonic() - started)
            if remaining <= 0:
                print("Session budget reached. Re-run this notebook to resume.", flush=True)
                break
            out = output / "models" / name
            out.mkdir(parents=True, exist_ok=True)
            completed = out / "complete.json"
            if cached(completed, ctx["identity"]):
                print(f"[{index}/{len(names)}] REUSE completed {name}", flush=True)
                continue
            model, reason = diagnostic_model(ctx["models"][name], assets, ctx["bank"])
            if reason:
                write_json(out / "status.json", {"status": "blocked", "reason": reason})
                print(f"BLOCKED {name}: {reason}", flush=True)
                report(output)
                continue
            if settings.get('dependency_repairs', False):
                from .dependency_fixes import repair_model
                model = repair_model(model)
            slot = scratch / "slots" / name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot / ".diagnostic-owner.json", {"identity": ctx["identity"]})
            job = {"project": str(project), "bank": str(ctx["bank"]), "output": str(out),
                   "slot": str(slot), "images": ctx["images"], "model": model,
                   "identity": ctx["identity"], "device": settings.get("device", "cuda"),
                   "assets": str(assets)}
            job_path = out / "job.json"
            write_json(job_path, job)
            model_env = worker_environment(project, slot, model)
            model_env.update(PYTHONPATH=env["PYTHONPATH"], UV_CACHE_DIR=str(slot / "cache/uv"),
                             UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
            print(f"[{index}/{len(names)}] {name}: prepare then diagnostic stages", flush=True)
            try:
                for action in ("prepare", "work"):
                    remaining = float(settings.get("max_hours", 12)) * 3600 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Session time budget exhausted")
                    command = ([sys.executable, "-m", "bank_diagnostics.controller", "prepare", "--job", str(job_path)]
                               if action == "prepare" else [read_json(out / "prepared.json")["python"],
                                   "-m", "bank_diagnostics.worker", str(job_path)])
                    command = [sys.executable, "-m", "wmbench.process_guard", "--", *command]
                    stamp = time.time_ns()
                    log = out / "logs" / f"{stamp}_{action}.log"
                    preparation_limit = settings.get('prepare_timeout_seconds', {}).get(name, 3600)
                    result = monitored_process(command, model_env, log, slot,
                                               min(remaining, preparation_limit if action == "prepare" else 1800),
                                               limits, assert_owned=owned)
                    write_json(out / "timings" / f"{stamp}_{action}.json", result)
                    if result["exit_code"] or result["reason"]:
                        from wmbench.embedding_bank import _log_tail
                        print(_log_tail(log), flush=True)
                        text = log.read_text(encoding='utf-8', errors='replace')
                        causes = [line for line in text.splitlines() if 'Error:' in line and not line.startswith('subprocess.CalledProcessError:')]
                        if causes:
                            print('ROOT CAUSE: ' + causes[-1], flush=True)
                        raise RuntimeError(f"{action} failed: {result['reason'] or result['exit_code']}; {log}")
                artifacts = [{"path": p.relative_to(out).as_posix(), "sha256": sha256(p)}
                             for p in sorted((out / "cases").rglob("*")) if p.is_file()]
                if not artifacts:
                    raise RuntimeError("Worker produced no diagnostic evidence")
                write_json(completed, {"signature": ctx["identity"], "artifacts": artifacts})
                write_json(out / "status.json", {"status": "completed", "reason": "Inspect BER; completed does not mean error-free recovery"})
            except Exception as error:
                write_json(out / "status.json", {"status": "technical_error", "reason": str(error)})
            finally:
                if (slot / "locks").exists():
                    shutil.copytree(slot / "locks", out / "environment", dirs_exist_ok=True)
                if settings.get("cleanup", True) and (completed.is_file() or not settings.get('keep_failed_scratch', False)):
                    cleanup_slot(slot, scratch, ctx["identity"])
                report(output)
    report(output)


def report(output: Path):
    """Only saved observations; never turn 'completed' into an accuracy claim."""
    rows = []
    states = []
    spec_path = output / 'diagnostic_spec.json'
    selected = read_json(spec_path)['models'] if spec_path.exists() else RECHECK + BLOCKED
    for model in selected:
        root = output / "models" / model
        status = read_json(root / "status.json") if (root / "status.json").exists() else {"status": "not_run"}
        states.append({"model": model, **status})
        for path in sorted((root / "cases").glob("*/*.json")):
            value = read_json(path)
            if "stage" not in value:
                continue
            rows.append({"model": model, "image_id": path.parent.name, "stage": value["stage"],
                         "status": value["status"], "control": value.get("control", False),
                         "common30_errors": value.get("common30_errors"), "native_errors": value.get("native_errors"),
                         "native_bits": value.get("payload_bits"), "seconds": value.get("seconds"),
                         "reason": value.get("reason", ""), "evidence": path.relative_to(output).as_posix()})
    write_csv(output / "stages.csv", rows)
    write_csv(output / "models.csv", states)
    lines = ["# Embedding diagnostics", "", "Historical bank is read-only. No production changes are applied.",
             "Completed means the diagnostic ran, not that every payload decoded correctly.", "",
             "| Method | Technical status | Repeat PNG common30 exact | Full payload exact |",
             "|---|---|---:|---:|"]
    prior_path = output / 'previous_run.json'
    if prior_path.exists():
        prior = read_json(prior_path)['read_only_path']
        lines[2:2] = [f'Previous completed checks remain in `{prior}/REPORT.md`. This report covers the recovery run only.', '']
    for state in states:
        checks = [r for r in rows if r["model"] == state["model"] and r["stage"] == "repeat_png_decode" and r["status"] == "measured"]
        n = len(checks)
        lines.append(f"| {state['model']} | {state['status']} | {sum(r['common30_errors']==0 for r in checks)}/{n} | {sum(r['native_errors']==0 for r in checks)}/{n} |")
    lines += ["", "## Blockers and errors", ""]
    lines += [f"- **{s['model']}**: {s.get('reason', s['status'])}" for s in states if s["status"] in ("blocked", "technical_error")]
    lines += ["", "## Stage comparisons", "", "| Method / stage | Measured | Common30 exact | Full payload exact |", "|---|---:|---:|---:|"]
    for key in sorted({(r["model"], r["stage"]) for r in rows if r["common30_errors"] is not None}):
        group = [r for r in rows if (r["model"], r["stage"]) == key and r["status"] == "measured"]
        lines.append(f"| {' / '.join(key)} | {len(group)} | {sum(r['common30_errors']==0 for r in group)} | {sum(r['native_errors']==0 for r in group)} |")
    matching = output / "matching/summary.json"
    if matching.exists():
        value = read_json(matching)
        lines += ["", "## Scalar PSNR audit", "", f"Checked {value['checked']} historical misses; independent search found {value['within_policy']} candidates within the unchanged tolerance.",
                  "No historical PNG or record was replaced. See matching/summary.json for brackets and per-image evidence."]
    lines += ["", "## How to interpret", "", "- FIN/HiDDeN: compare float, clipped float, PNG and 512 adaptation stages; do not select the best stage as a new benchmark without review.",
              "- Common30 and full payload are different columns. Error positions and logits are in per-stage JSON.",
              "- DistSeal: inspect ae_quality_control.json and ae_only_decode_control.json. Reconstruction is not a new matched-quality reference.",
              "- Downloaded weights are not a passed model; strict initialization and measured round trips are required.",
              "- Five images are a correctness pilot, not a robustness ranking or an FPR calibration."]
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit", "report", "prepare"))
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(read_json(args.job))
        return
    settings = read_json(args.settings)
    project = Path(__file__).resolve().parents[2]
    if args.action == "report":
        report(Path(settings["output"]))
    elif args.action == "audit":
        audit_matching(context(project, settings))
        report(Path(settings["output"]))
    else:
        run(project, settings)


if __name__ == "__main__":
    main()
