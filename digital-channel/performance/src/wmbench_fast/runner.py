"""Frozen benchmark semantics, bounded workers and explicit fast-resume receipts."""

import argparse
import os
from pathlib import Path
import time

from wmbench.io import read_json, write_json, atomic_bytes, sha256
from wmbench_full_payload.storage import EvidenceError, freeze, inside
from wmbench_digital import runner as original, report
from wmbench_digital.bank import context
from wmbench_digital.storage import complete, result_path
from wmbench_digital.spec import digest
from .cache import collect, ResultCache, pack, unpack, directory_inventory, fingerprint, verify_inputs

TABLES = ("counts.json", "reference_and_clean.csv", "attack_summary.csv", "macro_summary.csv",
          "case_levels.json.gz", "per_attack.csv.gz")


def verify_release():
    base = Path(__file__).resolve().parents[2]
    release = read_json(base / "performance-release.json")
    for filename, expected in release["files"].items():
        if sha256(inside(base, filename)) != expected:
            raise EvidenceError(f"Performance source changed: {filename}")
    return base / "production"


def snapshot(root, name, cases, with_hashes=False):
    folder = Path(root)/name
    inventory = directory_inventory(folder/"records")
    value = dict(case_keys=[c["key"] for c in cases],
                records={c["id"]+".json.gz": inventory.get(c["id"]+".json.gz") for c in cases},
                tables={n: fingerprint(folder/"tables"/n) for n in TABLES})
    if with_hashes:
        value["table_sha256"] = {n: sha256(folder/"tables"/n) for n in TABLES
                                  if value["tables"][n] is not None}
    return value


def tables_agree(folder, snap):
    """Drive can settle mtimes after a write. Only equal content may override it."""
    for name, stamp in snap["tables"].items():
        path = folder / "tables" / name
        current = fingerprint(path)
        if stamp is None or current is None:
            return False
        if current != stamp:
            expected = snap.get("table_sha256", {}).get(name)
            if expected is None or sha256(path) != expected:
                return False
    return True


def finished(root, name, cases, full_audit=False):
    path = Path(root)/name/"_fast/complete.json.gz"
    if full_audit or not path.exists():
        return False
    old = unpack(path)["snapshot"]
    current = snapshot(root, name, cases)
    return (old["case_keys"] == current["case_keys"] and old["records"] == current["records"]
            and tables_agree(Path(root)/name, old))


def mark_finished(root, name, cases):
    snap = snapshot(root, name, cases, with_hashes=True)
    if not all(snap["tables"].values()) or not all(snap["records"].values()):
        raise EvidenceError("Cannot commit incomplete method receipt")
    pack(Path(root)/name/"_fast/complete.json.gz", dict(snapshot=snap, time=time.time()))


def progress_report(root, selected, n_images, states):
    """Never label an old five-image table as a completed 100-image method."""
    rows = []
    for name in selected:
        current = states.get(name)
        if current is None:
            path = Path(root)/name/"status.json"
            old = read_json(path) if path.exists() else {}
            current = dict(status="not_checked_this_session", note="Historical status is not evidence for this image prefix",
                           historical_status=old.get("status", "not_started"))
        rows.append(dict(method=name, requested_images=n_images, **current))
    folder = Path(root)/"_reports"
    write_json(folder/"fast_progress.json", rows)
    text = ["# Digital channel — current session", "", f"Requested images: {n_images}.", "",
            "| Method | Status | Complete / eligible carriers |", "|---|---|---:|"]
    text += [f"| {r['method']} | {r['status']} | {r.get('complete_carriers', '—')} / {r.get('carriers', '—')} |" for r in rows]
    text += ["", "Not checked means unknown for this prefix, not a failed method. Existing historical tables are preserved.",
             "Per-method tables retain the original metric, bootstrap and rejection policies."]
    atomic_bytes(folder/"FAST_REPORT.md", ("\n".join(text)+"\n").encode())


def run(project, settings):
    from .lease_io import install
    install()
    from wmbench.colab_canary import controller_lease
    from wmbench_full_payload.runner import cleanup
    from . import scheduler
    from . import completed
    import stegastamp_paired_hotfix
    if os.name != "posix":
        raise RuntimeError("Use Linux/NVIDIA (Colab or GPU server)")
    for key in ("decode_workers", "attack_workers"):
        if type(settings[key]) is not int or not 1 <= settings[key] <= 4:
            raise ValueError(f"{key}: choose 1..4")
    if not 0 < settings["max_hours"] <= 168 or settings["ssd_reserve_gib"] < 35:
        raise ValueError("Invalid time budget / SSD reserve")
    stegastamp_paired_hotfix.install_controller_patch()
    root = Path(settings["output"]).resolve()
    (root/"_reports").mkdir(parents=True, exist_ok=True)
    with controller_lease(root/"_reports") as owned:
        ctx = context(project, settings)
        deadline = time.time()+settings["max_hours"]*3600
        states = {}
        for index, name in enumerate(ctx["selected"], 1):
            if time.time() >= deadline:
                break
            owned()
            print(f"METHOD {index}/{len(ctx['selected'])}: {name}", flush=True)
            model = ctx["models"][name]
            try:
                checked_at = time.monotonic()
                counts = completed.check(ctx, model, settings["full_audit"])
                if counts is not None:
                    states[name] = dict(status="complete_cached", **counts)
                    print(f"SKIP COMPLETE {name}: compact receipt + unchanged record/table fingerprints; "
                          f"{time.monotonic()-checked_at:.1f}s; no catalogue rewrite or GPU setup", flush=True)
                    progress_report(root, ctx["selected"], settings["n_images"], states)
                    continue
                cases = collect(ctx, model, settings["full_audit"])
                if finished(root, name, cases, settings["full_audit"]):
                    counts = read_json(root/name/"tables/counts.json")
                    completed.seal(ctx, model)
                    states[name] = dict(status="complete_cached", **counts)
                    print(f"SKIP COMPLETE {name}: catalogue + record fingerprints + report receipt agree; no GPU setup", flush=True)
                    progress_report(root, ctx["selected"], settings["n_images"], states)
                    continue
                cache = ResultCache(root, name, cases, settings["full_audit"])
                cache.warm()
                pending = [c for c in cases if not complete(cache.read(result_path(root, c), c))]
            except Exception as error:
                states[name] = dict(status="verification_error", reason=repr(error))
                print(f"STOP {name}: {error}", flush=True)
                break
            slot = ctx["scratch"]/"slots"/name
            slot.mkdir(parents=True, exist_ok=True)
            freeze(slot/".full-payload-owner.json", {"owner": ctx["identity"]})
            job = dict(model=model, project=str(project), bank=str(ctx["bank"]), results=str(root),
                       output=str(root/"_runtime"), slot=str(slot), scratch=str(ctx["scratch"]),
                       identity=ctx["identity"], session=str(time.time_ns()), deadline=deadline,
                       settings=settings, images=ctx["images"])
            path = root/name/"jobs"/(job["session"]+".json")
            write_json(path, job)
            status = root/name/"status.json"
            write_json(status, dict(status="running", session=job["session"], n_images=settings["n_images"]))
            try:
                # This preserves the reviewed StegaStamp migration and paired-Clean policy.
                if name == "stegastamp":
                    for case in pending:
                        verify_inputs(job, case)
                    original._stega_cuda_before(job, path, owned, cases)
                    cache = ResultCache(root, name, cases, True)  # A repair may have archived legacy records.
                    cache.warm()
                    pending = [c for c in cases if not complete(cache.read(result_path(root, c), c))]
                if pending:
                    scheduler.invoke(job, "attacks", path, owned)
                if any(not (root/name/"quality"/(c["id"]+".json")).exists() for c in cases):
                    original.invoke(job, "quality", path, owned)
                if pending:
                    original.invoke(job, "prepare", path, owned)
                    if name == "stegastamp":
                        original.invoke(job, "decode", path, owned)
                    else:
                        scheduler.decode(job, pending, owned)
                cache = ResultCache(root, name, cases, settings["full_audit"])
                cache.warm()
                from .report_io import method_report
                print(f"REPORT {name}: original metrics, SSD staging, verified Drive publication", flush=True)
                counts = method_report(root, name, ctx["scratch"] / "report-staging", cache, cases)
                cache.flush()
                if counts["complete_carriers"] != counts["carriers"]:
                    raise RuntimeError("Technical results incomplete; intermediates retained")
                ready = root/name/"packages-ready.json"
                manifests = read_json(ready)["manifests"] if ready.exists() else []
                # Original raw checksum/readback gate before destructive ZIP cleanup.
                original.remove_packages(root, cases, manifests, [ctx["scratch"], root/"_temporary"], ctx["identity"], name)
                mark_finished(root, name, cases)
                completed.seal(ctx, model)
                states[name] = dict(status="complete", **counts)
                write_json(status, dict(status="complete", session=job["session"], n_images=settings["n_images"],
                                        temporary_attacks_removed=True, bank_unchanged=True))
                print(f"COMPLETE {name}: results committed; verified disposable ZIPs removed", flush=True)
            except Exception as error:
                write_json(status, dict(status="error", reason=repr(error), session=job["session"], n_images=settings["n_images"]))
                states[name] = dict(status="error", reason=repr(error), carriers=len(cases))
                print(f"STOP {name}: {error}. Existing results and attack packages retained.", flush=True)
                break
            finally:
                cleanup(slot, ctx["scratch"], ctx["identity"])
                progress_report(root, ctx["selected"], settings["n_images"], states)
        progress_report(root, ctx["selected"], settings["n_images"], states)
        success = len(states) == len(ctx["selected"]) and all(s["status"] in {"complete", "complete_cached"} for s in states.values())
        if success:
            # Pairwise statistics are not rerun during an incomplete resume.
            receipt = root/"_reports/fast-pairwise.json"
            key = digest({n: sha256(root/n/"_fast/complete.json.gz") for n in ctx["selected"]})
            pairwise = settings.get("pairwise", True) and (not receipt.exists() or read_json(receipt)["key"] != key)
            report.final_report(root, pairwise=pairwise)
            if pairwise:
                write_json(receipt, {"key": key})
        print("ALL SELECTED METHODS COMPLETE" if success else "INCOMPLETE: see _reports/FAST_REPORT.md; resume the same output", flush=True)
        return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "validate", "decode", "attacks"])
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    project = verify_release()
    if args.action == "run":
        if not run(project, settings):
            raise SystemExit(2)
    elif args.action == "validate":
        ctx = context(project, settings)
        print(f"Frozen source valid: {len(ctx['selected'])} methods, {len(ctx['images'])} images; no bank PNG scan")
    elif args.action == "decode":
        from .worker import decode
        decode(settings)
    else:
        from .packages import prepare
        prepare(settings)


if __name__ == "__main__":
    main()
