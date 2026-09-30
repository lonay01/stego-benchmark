"""Cheap partial reports, explicit stage denominators and cohort separation."""

from pathlib import Path
import zipfile

from wmbench.io import atomic_bytes, read_json, write_json
from wmbench_quality_review.report import aggregate, metric, recovery
from .core import result


def report(root, *, archive=False):
    root = Path(root)
    if not (root / "plan.json").is_file():
        return None
    plan = read_json(root / "plan.json")
    groups, per_image, details = {}, {}, {}
    lines = ["# Four-method overnight quality check", "",
        f"{plan['n_images']} images × 5 full messages × 7 predefined configurations. "
        "Native only: no channel attacks, matching, training, shortened messages or selected winners.", "",
        "First five images replay old cases. Additional images are the next manifest entries, "
        "not necessarily unseen in every earlier diagnostic. Extra serializer/repeat controls "
        "are run on new measurements, not silently attributed to imported v1 cases.", ""]
    total = measured = imported = errors = check_failures = 0
    for cohort in ("historical", "additional"):
        lines += [f"## {cohort.title()} images", "",
            "| Method / profile | Measured | Raw exact; BER | PNG exact; BER | Source→AE dB | Source→marked dB | AE→marked dB |",
            "|---|---:|---|---|---:|---:|---:|"]
        for name, cases in plan["cases"].items():
            selected = [c for c in cases if c["cohort"] == cohort]
            if not selected:
                continue
            for profile in plan["profiles"][name]:
                # No full image/float hash scan for every report refresh.
                rows = [result(root, c, profile, plan["identity"], verify=False) for c in selected]
                key = f"{cohort}/{name}/{profile}"
                group = aggregate(rows)
                group["imported"] = sum(bool(r and r.get("imported_from")) for r in rows)
                groups[key] = group
                p = group["psnr_db"]
                lines.append(f"| {name} / {profile} | {group['measured']}/{group['expected']} | "
                    f"{recovery(group, 'raw')} | {recovery(group, 'native_png_direct')} | "
                    f"{metric(p['original_to_ae'])} | {metric(p['original_to_marked'])} | {metric(p['ae_to_marked'])} |")
                total += group["expected"]
                measured += group["measured"]
                imported += group["imported"]
                errors += group["errors"]
                check_failures += sum(group["check_failures"].values())
                for case, row in zip(selected, rows):
                    image_key = f"{key}/{case['image']['image_id']}"
                    per_image.setdefault(image_key, []).append(row)
                    if row and row.get("reason"):
                        details.setdefault(key, set()).add(row["reason"])
    checks_observed = any(g["check_failures"] for g in groups.values())
    summary = {"complete": measured == total, "measured": measured, "expected": total,
               "imported": imported, "technical_errors": errors,
               "unmeasured": total - measured, "check_failures": check_failures,
               "checks_ok": check_failures == 0 if checks_observed else None,
               "groups": groups,
               "note": "Measured coverage is not zero BER, paper reproduction or readiness for attacks."}
    lines += ["", "## Completion", "",
        f"Measured **{measured}/{total}**, including **{imported} imported v1 measurements**. "
        f"Technical-error records: **{errors}**. Failed observed consistency checks: **{check_failures}**.",
        f"Measurement coverage complete: **{summary['complete']}**. "
        "Unattempted and technically failed cases are not decoder BER=100% and not successful cases.", "",
        "## Extra controls (denominators include only observed stages)", "",
        "| Cohort / method / profile | Floor PNG exact; BER | Half-up PNG exact; BER | Float reload exact; BER |",
        "|---|---|---|---|"]
    for key, group in groups.items():
        lines.append(f"| {key} | {recovery(group, 'floor_png')} | {recovery(group, 'half_up_png')} | {recovery(group, 'float_reload')} |")
    lines += ["", "## Interpretation", "",
        "- Crop and resize expose different image content; do not treat this as an isolated architecture effect.",
        "- SepWater uses 224px for both profiles. Its previous 256px failures remain in the old report; they are not successes in this run.",
        "- RoSteALS resize256 retains the prior direct-core/round-even diagnostic, while historical retains the file API. Report both preprocessing and serializer differences.",
        "- EditGuard clean/degrade are different author checkpoints. The main decoder retains its own quantizer; the no-quantizer control is labelled separately.",
        "- Source→AE, source→marked and AE→marked PSNR have different references and are not additive error budgets.",
        "- Clean/AE bit scores are negative controls, not calibrated false-positive rates.",
        "- Nonzero BER is a result and is not retried. No automatic selection of a favourable serializer or checkpoint.",
        "- Summaries average equal-length cases within a method; methods have different payload lengths. Use per_image.json for image-level analysis, not 100 independent-image claims.",
        "- Reporting reads sealed record signatures; the execution/resume path checks artifact hashes. No models are loaded by reporting.", "",
        "## Technical status", ""]
    for name in plan["cases"]:
        path = root / name / "_meta/status.json"
        state = read_json(path) if path.exists() else {"status": "not_started"}
        lines.append(f"- {name}: {state.get('status')}; {state.get('reason', '')}")
    for key, reasons in details.items():
        for reason in sorted(reasons):
            lines.append(f"- {key}: `{reason.replace('`', chr(39))}`")
    output = root / "_reports"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "per_image.json", {k: aggregate(v) for k, v in per_image.items()})
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    if archive:
        target = root / "Four_Method_NIGHT_report.zip"
        temporary = target.with_suffix(".part")
        candidates = [p for p in root.rglob("*") if p.is_file()
            and p.suffix in (".json", ".md", ".log", ".txt", ".lock")
            and "_release" not in p.parts and "attempts" not in p.parts
            and p.stat().st_size < 8 * 1024**2]
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(candidates):
                zf.write(path, path.relative_to(root).as_posix())
        temporary.replace(target)
    return summary
