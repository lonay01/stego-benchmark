"""Count actual RoSteALS evidence, never mark empty/partial runs successful."""

from collections import defaultdict
import io
from pathlib import Path
import zipfile

from authornight.report import aggregate
from authornight.storage import atomic, cached, read, write


def report(root):
    root = Path(root)
    plan = read(root / "plan.json")
    expected = len(plan["images"]) * 5
    summary = {"identity": plan["identity"], "parent_identity": plan["parent_identity"],
               "method": "rosteals", "profiles": {}, "integrity_errors": [],
               "sepwater": "not run; previous results unchanged", "distseal": "not run"}

    def records(folder):
        rows = []
        for path in sorted((root / folder).glob("*.json")):
            try:
                row = cached(root, path.relative_to(root).as_posix(), plan["identity"], verify=False)
                rows.append(row)
            except Exception as error:
                summary["integrity_errors"].append({"path": str(path), "error": str(error)})
        return rows

    expected_keys = {(int(x["image_id"]), rep, profile) for x in plan["images"]
                     for rep in range(5) for profile in ("raw100", "author_bch56")}
    rows, seen = [], set()
    for row in records("rosteals/coco/records"):
        key = (row.get("image_id"), row.get("repetition"), row.get("profile"))
        if key not in expected_keys or key in seen:
            summary["integrity_errors"].append({"error": "Unexpected/duplicate native case", "key": key})
            continue
        seen.add(key)
        rows.append(row)
    lines = ["# RoSteALS: author-path import fix", "",
             "Full payload, fixed COCO inputs and author export. No training or matched PSNR.",
             "SepWater and DistSeal are not scheduled; their old evidence remains unchanged.", "",
             "| Profile | Measured / planned | PNG BER | Exact full payload | Failed checks |",
             "|---|---:|---:|---:|---:|"]
    for profile in ("raw100", "author_bch56"):
        group = aggregate([r for r in rows if r["profile"] == profile])
        group["planned"] = expected
        summary["profiles"][profile] = group
        png = group["measurements"].get("author_png", {})
        ber = f"{png['ber']*100:.4f}%" if png.get("ber") is not None else "not measured"
        lines.append(f"| {profile} | {group['cases']}/{expected} | {ber} | {png.get('exact', '—')} | {len(group['failed_checks'])} |")
    native_complete = seen == expected_keys and all(
        g["measurements"].get("author_png", {}).get("cases", 0) == expected
        for g in summary["profiles"].values())
    native_checks = bool(rows) and all(r.get("checks") and all(r["checks"].values()) for r in rows)
    groups = records("rosteals/corruptions")
    by_attack = defaultdict(list)
    complete_grids = measured = errors = 0
    grid_file = root / "_meta/recovery/rosteals_attack_grid.json"
    attack_grid = {tuple(x) for x in read(grid_file)} if grid_file.exists() else set()
    expected_stems = {f"{int(x['image_id']):012d}_m{r}_raw100" for x in plan["images"] for r in range(5)}
    native_hashes = {f"rosteals/coco/records/{r['image_id']:012d}_m{r['repetition']}_{r['profile']}.json":
                     r["record_sha256"] for r in rows}
    grid_seen = set()
    for group in groups:
        source = group.get("source_record", "")
        stem = Path(source).stem
        scores = group["scores"]
        unique = {(s["attack"], s["severity"]) for s in scores}
        if (stem not in expected_stems or stem in grid_seen or len(unique) != len(scores)
                or len(attack_grid) != 70 or unique - attack_grid
                or group.get("source_record_sha256") != native_hashes.get(source)):
            summary["integrity_errors"].append({"error": "Invalid/stale attack grid", "source": source})
            continue
        grid_seen.add(stem)
        complete_grids += unique == attack_grid and all(s["status"] == "measured" for s in scores)
        for score in scores:
            if score["status"] == "measured":
                measured += 1
                by_attack[score["attack"]].append(score)
            else:
                errors += 1
    summary["attacks"] = {"complete_grids": complete_grids, "planned_grids": expected,
        "measured_tests": measured, "planned_tests": expected * 70, "error_tests": errors,
        "per_attack": {k: {"tests": len(v), "ber": sum(s["errors"] for s in v)/sum(s["nbits"] for s in v)}
                       for k, v in by_attack.items()}}
    latest = {}
    for path in sorted((root / "_logs").glob("*.outcome.json")):
        value = read(path)
        latest[value["phase"]] = value
    summary["latest_stages"] = latest
    checks = {}
    for key, path in (("imports", "_meta/fix/import_check.json"),
                      ("preflight", "_meta/recovery/rosteals_preflight.json")):
        checks[key] = read(root / path) if (root / path).exists() else {"passed": False}
    summary["runtime_checks"] = checks
    summary["native_complete"] = native_complete
    summary["native_checks_passed"] = bool(native_checks and not summary["integrity_errors"])
    summary["coverage_complete"] = native_complete and complete_grids == expected
    # A completed first-image check is not success if a later runtime stage failed.
    summary["checks_passed"] = bool(summary["native_checks_passed"] and errors == 0
        and all(x.get("passed") for x in checks.values())
        and all(x["status"] == "returned" for x in latest.values()))
    summary["complete"] = summary["coverage_complete"] and summary["checks_passed"]
    lines += ["", "## Corruptions", "",
        f"Measured {measured}/{expected*70}; complete grids {complete_grids}/{expected}; errors {errors}.",
        "Author test pool, five severities, raw100 internal PNGs. Not the paper's random-severity distribution.", "",
        f"**Native complete: {native_complete}. Overall complete: {summary['complete']}.**", "",
        "BER need not be zero. A technical failure is not a watermark-decoding score.",
        "The BCH text cohort is separate from 100 random bits; seven hex characters carry 28 bits of entropy.",
        "Author ECC helper outputs are kept unchanged, not silently replaced with SepWater's corrected decoder.", "",
        "## Latest stages", ""]
    lines += [f"- {k}: {v['status']}, {v['seconds']/60:.1f} min; `{v['log']}`" for k, v in latest.items()]
    if summary["integrity_errors"]:
        lines += ["", "Evidence integrity errors: inspect summary.json before interpreting scores."]
    write(root / "_reports/summary.json", summary)
    atomic(root / "_reports/report.md", "\n".join(lines).encode())
    return summary


def bundle(root):
    root = Path(root)
    report(root)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        selected = [root / "plan.json", root / "_inputs/originals.json",
                    *sorted((root / "_reports").glob("*")),
                    *sorted((root / "_meta").rglob("*.json")),
                    *sorted((root / "_logs").glob("*.outcome.json")),
                    *sorted((root / "rosteals/coco/records").glob("*.json")),
                    *sorted((root / "rosteals/corruptions").glob("*.json"))]
        for path in selected:
            if path.is_file():
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted((root / "_logs").glob("*.log")):
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - 65536))
                archive.writestr("_logs/tails/" + path.name, log.read())
        for path in sorted((root / "rosteals/coco/artifacts").glob("*.zip")):
            with zipfile.ZipFile(path) as images:
                if "cli.log" in images.namelist():
                    archive.writestr("_logs/author_cli/" + path.stem + ".log", images.read("cli.log")[-65536:])
    destination = root / "RoSteALS_AUTHOR_FIX_report.zip"
    atomic(destination, stream.getvalue())
    return destination
