"""Report measured cohorts with explicit denominators; missing is never zero BER."""

from collections import defaultdict
import csv
import io
from pathlib import Path
import statistics

from .storage import atomic, cached, read, write


def aggregate(rows):
    result = {"cases": len(rows), "measurements": {}, "quality": {}, "failed_checks": []}
    for index, row in enumerate(rows):
        for name, ok in row.get("checks", {}).items():
            if not ok:
                result["failed_checks"].append({"case": row.get("image_id", index), "check": name})
    stages = sorted({name for row in rows for name in row.get("measurements", {})})
    for stage in stages:
        scores = [row["measurements"][stage] for row in rows if stage in row.get("measurements", {})]
        nbits = sum(x["nbits"] for x in scores)
        result["measurements"][stage] = {"cases": len(scores), "nbits": nbits,
            "ber": sum(x["errors"] for x in scores)/nbits if nbits else None,
            "exact": sum(x["exact"] for x in scores),
            "word_accuracy_threshold_0_2": statistics.mean(x["word_accuracy_threshold_0_2"] for x in scores),
            "author_ecc_text_exact": sum(x.get("author_ecc_text_exact", False) for x in scores)
                                      if any("author_ecc_text_exact" in x for x in scores) else None}
    for comparison in sorted({k for row in rows for k in row.get("quality", {})}):
        pairs = [row["quality"][comparison] for row in rows if comparison in row.get("quality", {})]
        result["quality"][comparison] = {}
        for metric in ("psnr_db", "ssim", "lpips_raw", "sifid_raw"):
            values = [p[metric] for p in pairs if p.get(metric) is not None]
            result["quality"][comparison][metric] = statistics.mean(values) if values else None
            result["quality"][comparison][metric + "_n"] = len(values)
        result["quality"][comparison]["identical_pixels"] = sum(p.get("identical_pixels", False) for p in pairs)
    return result


def report(root):
    root = Path(root)
    plan = read(root / "plan.json")
    groups = defaultdict(list)
    integrity_errors, technical_errors = [], []
    for method in ("rosteals", "sepwater", "distseal"):
        for cohort in ("coco", "generated"):
            for path in sorted((root / method / cohort / "records").glob("*.json")):
                try:
                    # Verify small record hashes here, not multi-GB artifacts.
                    # Workers verify actual artifacts before reuse.
                    row = cached(root, path.relative_to(root).as_posix(), plan["identity"], verify=False)
                    samples = row.get("samples", [row])
                    groups[(method, cohort, row["profile"])].extend(samples)
                except Exception as error:
                    integrity_errors.append({"path": str(path), "error": str(error)})
    summary = {"identity": plan["identity"], "groups": {}, "attacks": {}, "integrity_errors": integrity_errors,
               "fid": {"status": "unavailable", "reason": "ImageNet validation not supplied"},
               "full_paper_reproduction": False, "paper_limits": plan["paper_limits"]}
    lines = ["# Three-method author-path diagnostics", "",
             "This is an inference reproduction and diagnostic run, not a reproduction of training.",
             "Native author pipelines only. No matched PSNR, payload shortening, cherry-picked messages or retraining.", "",
             "## COCO and generated-image recovery", "",
             "BER is measured over the entire decoder payload. Exact means zero bit errors.",
             "Rows have different datasets/messages and must not be ranked as one common benchmark.", "",
             "| Method / cohort / payload | Completed / planned | Raw BER | PNG BER | PNG exact | Check failures |",
             "|---|---:|---:|---:|---:|---:|"]
    expected = {(m, "coco", p): len(plan["images"])*5 for m, profiles in plan["profiles"].items() for p in profiles}
    expected[("distseal", "generated", "author_fixed64")] = 1000*plan["generation"]["per_class"]
    flat = []
    for key, count in expected.items():
        group = aggregate(groups[key])
        group["planned"] = count
        group["coverage_complete"] = group["cases"] == count
        label = "/".join(key)
        summary["groups"][label] = group
        raw = group["measurements"].get("raw_internal", {})
        png = group["measurements"].get("author_png", {})
        def fmt(x):
            return "—" if x is None else f"{x*100:.4f}%"
        lines.append(f"| {label} | {group['cases']} / {count} | {fmt(raw.get('ber'))} | {fmt(png.get('ber'))} | {png.get('exact', '—')} | {len(group['failed_checks'])} |")
        flat.append({"method": key[0], "cohort": key[1], "profile": key[2], "completed": group["cases"],
                     "planned": count, "raw_ber": raw.get("ber"), "png_ber": png.get("ber"),
                     "png_exact": png.get("exact"), "failed_checks": len(group["failed_checks"])})
    lines += ["", "## Quality", "", "All quality comparisons are retained in `summary.json`.",
              "Source → AE and AE → marked are separate: the autoencoder can account for much of source → marked distortion.",
              "RoSteALS/SepWater source → PNG uses the author's original-resolution residual export; internal-size quality is separate.",
              "Generated DistSeal compares marked and unmarked decodes of the same sampled latent, not unrelated images.", "",
              "| Cohort / comparison | PSNR dB | SSIM | LPIPS | SIFID |",
              "|---|---:|---:|---:|---:|"]
    for label, group in summary['groups'].items():
        for comparison, values in group['quality'].items():
            fields = ['—' if values.get(k) is None else f'{values[k]:.5f}' for k in ('psnr_db', 'ssim', 'lpips_raw', 'sifid_raw')]
            lines.append('| '+label+' / '+comparison+' | '+' | '.join(fields)+' |')
    lines += ["",
              "## Robustness coverage", "", "| Method / cohort | Completed grids / planned | Measured tests | Technical errors |",
              "|---|---:|---:|---:|"]
    for method, folder, n, per_grid in (
            ("rosteals", "corruptions", len(plan["images"])*5, 70),
            ("sepwater", "corruptions", len(plan["images"])*5, 45),
            ("distseal", "coco_attacks", len(plan["images"])*5, 87),
            ("distseal", "generated_attacks", 1000*plan["generation"]["per_class"], 87)):
        tests, errors, grids = 0, 0, 0
        by_attack = defaultdict(list)
        for path in sorted((root / method / folder).glob("*.json")):
            try:
                row = cached(root, path.relative_to(root).as_posix(), plan["identity"], verify=False)
                samples = {x.get("sample", path.stem) for x in row["scores"]}
                grids += len(samples)
                for score in row["scores"]:
                    if score.get("status", "measured") == "measured":
                        tests += 1
                        by_attack[score["attack"]].append(score)
                    else:
                        errors += 1
                        technical_errors.append(score)
            except Exception as error:
                integrity_errors.append({"path": str(path), "error": str(error)})
        values = {name: {"tests": len(scores), "ber": sum(x["errors"] for x in scores)/sum(x["nbits"] for x in scores),
                          "exact": sum(x["exact"] for x in scores)} for name, scores in by_attack.items()}
        summary["attacks"][method+"/"+folder] = {"completed_grids": grids, "planned_grids": n,
            "measured_tests": tests, "planned_tests": n*per_grid, "technical_errors": errors,
            "per_attack": values, "macro_ber_observed_categories": statistics.mean(v["ber"] for v in values.values()) if values else None}
        lines.append(f"| {method}/{folder} | {grids} / {n} | {tests} | {errors} |")
    outcomes = [read(p) for p in sorted((root / "_logs").glob("*.outcome.json"))]
    summary["stage_history"] = outcomes
    summary["technical_errors"] = technical_errors
    summary['inception_score'] = []
    for path in sorted((root/'distseal/inception').glob('*.json')):
        summary['inception_score'].append(cached(root, path.relative_to(root).as_posix(), plan['identity'], verify=False))
    lines += ['', '## Inception Score', '']
    if summary['inception_score']:
        for row in summary['inception_score']:
            lines.append(f"- {row['variant']}: n={row['samples']}; {row['metrics']}; partial={row['partial_sample_count']}.")
    else:
        lines.append('Not measured yet: needs at least 1,000 generated samples and time in the final stage.')
    lines += ["", "## Limits and next action", ""] + ["- " + x for x in plan["paper_limits"]]
    lines += ["- A time-budget stop or missing record is unfinished work, not evidence of poor watermark quality.",
              "- A loader/download/dependency error is a technical failure, not BER = 100%.",
              "- Review `_logs/*.log`, `_meta/*/*audit.json`, and failed checks before interpreting model quality.",
              "- Rerun the same notebook with the same output folder. Verified completed records are reused.",
              "- BER > 0 is a measured result. The notebook never selects a serializer/message to make BER look better.",
              "- RoSteALS paper word accuracy (BER < 20%) is not the same as exact message recovery.",
              "- SepWater's published 256px metric is not a reference target for the released 224px checkpoint.",
              "", "FID against ImageNet validation: **unavailable**. No substitute reference is used.", ""]
    if integrity_errors:
        lines += ["**Evidence integrity errors: inspect summary.json before using any aggregate.**", ""]
    output = root / "_reports"
    write(output / "summary.json", summary)
    atomic(output / "report.md", "\n".join(lines).encode())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    atomic(output / "recovery.csv", buffer.getvalue().encode())
    return summary


def report_bundle(root):
    """Small evidence archive, never include model weights or the image bank."""
    import zipfile
    root = Path(root)
    report(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        selected = [root/'plan.json', *sorted((root/'_reports').glob('*')),
                    *sorted((root/'_meta').rglob('*.json')),
                    *sorted((root/'_logs').glob('*.outcome.json')),
                    *sorted((root/'distseal/inception').glob('*.json'))]
        for method in ('rosteals', 'sepwater', 'distseal'):
            selected += sorted((root/method/'coco/records').glob('*.json'))
        for path in selected:
            if path.is_file():
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted((root/'_logs').glob('*.log')):
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-65536))
                archive.writestr('_logs/tails/'+path.name, stream.read())
    target = root/'Three_Method_AUTHOR_NIGHT_report.zip'
    atomic(target, buffer.getvalue())
    return target
