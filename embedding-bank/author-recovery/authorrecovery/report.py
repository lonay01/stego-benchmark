"""Separate code correctness, data coverage, intrinsic BER and corrected ECC."""

from collections import defaultdict
import io
from pathlib import Path
import zipfile

from authornight.report import aggregate
from authornight.storage import atomic, cached, read, write


def report(root):
    root = Path(root)
    plan = read(root / "plan.json")
    summary = {"identity": plan["identity"], "parent_identity": plan["parent_identity"],
               "models": {}, "integrity_errors": [], "ecc": {}, "distseal": "not run"}
    expected = len(plan["images"]) * 5
    lines = ["# RoSteALS / SepWater recovery", "",
             "Full 100-bit payload; native author inference only. No retraining or matched PSNR.",
             "Parent evidence is read-only. Imported measurements retain explicit provenance.", "",
             "| Model / profile | Cases / planned | PNG BER | PNG exact | Failed checks |",
             "|---|---:|---:|---:|---:|"]

    def records(folder):
        rows = []
        for path in sorted((root / folder).glob("*.json")):
            try:
                rows.append(cached(root, path.relative_to(root).as_posix(), plan["identity"], verify=False))
            except Exception as error:
                summary["integrity_errors"].append({"path": path.relative_to(root).as_posix(), "error": str(error)})
        return rows

    for method in ("rosteals", "sepwater"):
        rows = records(f"{method}/coco/records")
        groups = {}
        for profile in ("raw100", "author_bch56"):
            group = aggregate([r for r in rows if r["profile"] == profile])
            group["planned"] = expected
            groups[profile] = group
            png = group["measurements"].get("author_png", {})
            ber = f"{png['ber']*100:.4f}%" if "ber" in png else "not measured"
            lines.append(f"| {method}/{profile} | {group['cases']}/{expected} | {ber} | {png.get('exact', '—')} | {len(group['failed_checks'])} |")
        summary["models"][method] = groups
    lines += ["", "## BCH correction (SepWater, text cohort only)", "",
              "Raw100 is not an ECC packet. Corrected text/data recovery is NOT full-payload exact recovery.",
              "Seven hexadecimal characters occupy 56 data positions but contain only 28 bits of entropy.",
              "This is a decoder compatibility diagnostic, not a new claim of 56 random useful bits.", "",
              "| Stage | Evaluated / planned | Legacy exact text | Corrected exact data | Rejected | Accepted but wrong |",
              "|---|---:|---:|---:|---:|---:|"]
    ecc_rows = records("sepwater/ecc_corrected")
    for stage in ("raw_internal", "author_png"):
        scores = [r["stages"][stage] for r in ecc_rows]
        counts = {"cases": len(scores), "planned": expected,
                  "legacy_text_exact": sum(bool(s["legacy_text_exact"]) for s in scores),
                  "data_exact": sum(s["data_exact"] for s in scores),
                  "rejected": sum(not s["accepted"] for s in scores),
                  "accepted_but_wrong": sum(s["accepted_but_wrong"] for s in scores)}
        summary["ecc"][stage] = counts
        lines.append(f"| {stage} | {len(scores)}/{expected} | {counts['legacy_text_exact']} | {counts['data_exact']} | {counts['rejected']} | {counts['accepted_but_wrong']} |")
    selftest = root / "_meta/recovery/bch_selftest.json"
    summary["ecc_selftest"] = read(selftest) if selftest.exists() else {"passed": False, "status": "not run"}
    lines += ["", "## Fixed corruption grids", "",
              "Starts from internal floor PNGs, not the original-resolution exports. Different author pools are not a shared model ranking.", "",
              "| Model | Complete grids / planned | Measured / planned tests | Error tests |",
              "|---|---:|---:|---:|"]
    summary["attacks"] = {}
    for method, per_grid in (("rosteals", 70), ("sepwater", 45)):
        grids = records(f"{method}/corruptions")
        complete, measured, errors = 0, 0, 0
        by_attack = defaultdict(list)
        for grid in grids:
            scores = grid["scores"]
            unique = {(s["attack"], s["severity"]) for s in scores}
            if len(unique) != len(scores):
                summary["integrity_errors"].append({"method": method, "error": "duplicate attack observations"})
                continue
            complete += len(scores) == per_grid and all(s["status"] == "measured" for s in scores)
            for score in scores:
                if score["status"] == "measured":
                    measured += 1
                    by_attack[score["attack"]].append(score)
                else:
                    errors += 1
        summary["attacks"][method] = {"complete_grids": complete, "planned_grids": expected,
            "measured_tests": measured, "planned_tests": expected*per_grid, "error_tests": errors,
            "per_attack": {name: {"tests": len(scores), "ber": sum(s["errors"] for s in scores)/sum(s["nbits"] for s in scores)}
                           for name, scores in by_attack.items()}}
        lines.append(f"| {method} | {complete}/{expected} | {measured}/{expected*per_grid} | {errors} |")
    latest = {}
    for path in sorted((root / "_logs").glob("*.outcome.json")):
        outcome = read(path)
        latest[outcome["method"] + "/" + outcome["phase"]] = outcome
    summary["latest_stages"] = latest
    summary["preflights"] = {m: read(root / "_meta/recovery" / (m + "_preflight.json"))
                             if (root / "_meta/recovery" / (m + "_preflight.json")).exists() else {"passed": False}
                             for m in ("rosteals", "sepwater")}
    coverage = all(g["cases"] == expected for groups in summary["models"].values() for g in groups.values())
    coverage &= len(ecc_rows) == expected and all(a["complete_grids"] == expected for a in summary["attacks"].values())
    checks_ok = all(not g["failed_checks"] for groups in summary["models"].values() for g in groups.values())
    healthy = not summary["integrity_errors"] and checks_ok and summary["ecc_selftest"].get("passed", False)
    healthy &= all(p["passed"] for p in summary["preflights"].values())
    healthy &= all(o["status"] == "returned" for o in latest.values())
    summary["coverage_complete"] = bool(coverage)
    summary["checks_passed"] = bool(healthy)
    summary["complete"] = bool(coverage and healthy)
    lines += ["", f"**Coverage complete: {bool(coverage)}. Checks passed: {bool(healthy)}.**", "",
              "Completion does not require BER=0. Intrinsic errors must not be tuned away on the test set.",
              "Missing/time-budget-limited work is not a model failure. Failed preflights block new GPU measurements for that model.",
              "SepWater remains the released 224px checkpoint, not a validated 256px paper reproduction.",
              "DistSeal was not scheduled. See DISTSEAL_RESUME.md for its frozen run and continuation instructions.", "",
              "## Latest stages", ""]
    lines += [f"- {name}: {o['status']} ({o['seconds']/60:.1f} min). Log: `{o['log']}`." for name, o in sorted(latest.items())]
    if summary["integrity_errors"]:
        lines += ["", "**Evidence integrity errors: do not interpret aggregates until resolved.**"]
    write(root / "_reports/summary.json", summary)
    atomic(root / "_reports/report.md", "\n".join(lines).encode())
    return summary


def bundle(root):
    root = Path(root)
    report(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        selected = [root / "plan.json", *sorted((root / "_reports").glob("*")),
                    *sorted((root / "_meta").rglob("*.json")), *sorted((root / "_logs").glob("*.outcome.json"))]
        for method in ("rosteals", "sepwater"):
            for folder in ("coco/records", "corruptions", "ecc_corrected"):
                selected += sorted((root / method / folder).glob("*.json"))
        selected += [root / "DISTSEAL_RESUME.md"]
        for path in selected:
            if path.is_file():
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted((root / "_logs").glob("*.log")):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size-65536))
                archive.writestr("_logs/tails/" + path.name, stream.read())
    destination = root / "RoSteALS_SepWater_RECOVERY_report.zip"
    atomic(destination, buffer.getvalue())
    return destination
