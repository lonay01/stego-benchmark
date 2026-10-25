"""Image-cluster estimates, paired degradation and explicit missing coverage."""

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from wmbench.io import atomic_bytes, read_json, write_json, write_csv
from wmbench_digital.report import mean, comparison, METRICS as DIGITAL_RECOVERY_METRICS
from .spec import ATTACKS
from .storage import complete, read_result, result_path

RECOVERY_METRICS = [*DIGITAL_RECOVERY_METRICS, "bit_accuracy_percent"]
QUALITY_METRICS = ["attack_psnr_finite", "attack_ssim", "attack_lpips", "mae_inside_mask", "mae_outside_mask", "attack_seconds_amortized"]
METRICS = [*RECOVERY_METRICS, *QUALITY_METRICS]


def summarize(rows):
    # Use the declared neural seed, not the frozen digital helper's policy.
    import numpy as np
    from .spec import POLICY
    result = {}
    for key in METRICS:
        values = np.array([r[key] for r in rows if r.get(key) is not None])
        if len(values):
            rng = np.random.default_rng(POLICY["bootstrap"]["seed"])
            boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
            avg, low, high = float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        else:
            avg = low = high = None
        result.update({key: avg, key+"_ci_low": low, key+"_ci_high": high, key+"_images": len(values)})
    return result


def method_report(root, name):
    root = Path(root)
    cases = read_json(root / name / "cases.json")
    directory = root / name / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    cells, controls = [], []
    counter = Counter()
    keys = ["method", "mode", "image_id", "replicate", "attack", "family", "level", "repeat", "status", "reason",
        "information_bits", "interface", "image_sha256", "seed", "psnr", "ssim", "lpips",
        "mae_inside_mask", "mae_outside_mask", "attack_seconds_amortized", *RECOVERY_METRICS]
    output = directory / "per_attack.csv.gz"
    temp = output.with_suffix(".partial")
    with gzip.open(temp, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for case in cases:
            record = read_result(result_path(root, case), case)
            counter["carriers"] += 1
            counter["complete_carriers"] += int(complete(record))
            rows = record["rows"] if record else {}
            clean = rows.get("clean", {})  # Never call a historical row a fresh measured Clean.
            cap = case["assignment"]["capacity"]
            descriptor = dict(information_bits=cap["information_bits"], interface=cap["interface"],
                width=case["size"][0], height=case["size"][1],
                bits_per_pixel=cap["information_bits"]/(case["size"][0]*case["size"][1]))
            grouped = defaultdict(list)
            for attack in ATTACKS:
                row = rows.get(attack["id"], {"status": "missing"})
                metrics = comparison(row, clean)
                metrics["bit_accuracy_percent"] = 100-metrics["ber_percent"] if metrics["ber_percent"] is not None else None
                counter[row["status"]] += 1
                quality = row.get("attack_quality") or {}
                writer.writerow(dict(method=name, mode=case["mode"], image_id=case["image_id"], replicate=case["replicate"],
                    attack=attack["id"], family=attack["family"], level=attack["level"], repeat=attack["repeat"],
                    status=row["status"], reason=row.get("reason"), image_sha256=row.get("image_sha256"), seed=row.get("seed"),
                    information_bits=cap["information_bits"], interface=cap["interface"], **metrics, **quality,
                    attack_seconds_amortized=(row.get("attack_timing") or {}).get("amortized_attack_seconds")))
                metrics.update(attack_psnr_finite=quality.get("psnr") if isinstance(quality.get("psnr"), (int, float)) else None,
                    attack_ssim=quality.get("ssim"), attack_lpips=quality.get("lpips"),
                    mae_inside_mask=quality.get("mae_inside_mask"), mae_outside_mask=quality.get("mae_outside_mask"),
                    attack_seconds_amortized=(row.get("attack_timing") or {}).get("amortized_attack_seconds"))
                grouped[(attack["family"], attack["level"])].append((row, metrics))
            for (family, level), values in grouped.items():
                cells.append(dict(method=name, mode=case["mode"], image_id=case["image_id"], replicate=case["replicate"],
                    family=family, level=level, **descriptor, planned=len(values),
                    **{s: sum(r["status"] == s for r, _ in values) for s in
                       ("measured", "rejected", "blocked_generation", "error", "missing")},
                    valid=sum(r["status"] == "measured" for r, _ in values),
                    paired=sum(m["delta_ber_pp"] is not None for _, m in values),
                    exact_valid=sum(r["status"] in {"measured", "rejected"} for r, _ in values),
                    exact_paired=sum(m["exact_loss_pp"] is not None for _, m in values),
                    **{k: mean([m[k] for _, m in values]) for k in METRICS}))
            quality_file = root / name / "quality" / (case["id"]+".json")
            quality = read_json(quality_file) if quality_file.exists() else {}
            if quality and quality["case_key"] != case["key"]:
                raise ValueError("Embedding quality belongs to a different carrier")
            resize = (record or {}).get("resize_only", {})
            controls.append(dict(method=name, mode=case["mode"], image_id=case["image_id"], replicate=case["replicate"],
                **descriptor, historical_bank_ber=case["clean"].get("ber"), fresh_clean_ber=clean.get("ber"),
                clean_status=clean.get("status", "missing"), resize_only_ber=resize.get("ber"),
                resize_only_status=resize.get("status", "missing"),
                resize_only_delta_ber_pp=comparison(resize, clean)["delta_ber_pp"],
                embedding_psnr=case["psnr"], embedding_ssim=quality.get("ssim"), embedding_lpips=quality.get("lpips"),
                alpha=(case.get("matching") or {}).get("alpha"), embedding_seconds=case["embedding_seconds"],
                author_psnr="not imported; see archived reference document", author_ber="not imported; no invented numbers"))
    temp.replace(output)
    write_csv(directory / "reference_and_clean.csv", controls)
    atomic_bytes(directory / "case_levels.json.gz", gzip.compress(json.dumps(cells).encode(), mtime=0))
    counters = ("planned", "valid", "rejected", "blocked_generation", "error", "missing", "paired", "exact_valid", "exact_paired")
    by_image = defaultdict(list)
    for row in cells:
        by_image[(row["mode"], row["family"], row["level"], row["image_id"])].append(row)
    images = [dict(mode=m, family=f, level=l, image_id=i,
                   **{k: mean([r[k] for r in values]) for k in METRICS},
                   **{k: sum(r[k] for r in values) for k in counters})
              for (m, f, l, i), values in by_image.items()]
    levels = defaultdict(list)
    for row in images:
        levels[(row["mode"], row["family"], row["level"])].append(row)
    write_csv(directory / "attack_summary.csv", [dict(method=name, mode=m, family=f, level=l,
        **descriptor, **{k: sum(r[k] for r in v) for k in counters}, **summarize(v)) for (m, f, l), v in levels.items()])
    macro = defaultdict(list)
    for mode, image in sorted({(r["mode"], r["image_id"]) for r in images}):
        values = [r for r in images if r["mode"] == mode and r["image_id"] == image and r["family"] != "clean"]
        result = {}
        for key in RECOVERY_METRICS:
            required = ("exact_paired" if "exact" in key and key != "exact_percent" else "exact_valid" if key == "exact_percent"
                        else "paired" if "paired" in key or key == "delta_ber_pp" else "valid")
            families = defaultdict(list)
            for row in values:
                families[row["family"]].append(row[key])
            good = len(families) == 4 and all(r[required] == r["planned"] and r[key] is not None for r in values)
            result[key] = mean([mean(v) for v in families.values()]) if good else None
        macro[mode].append(result)
    write_csv(directory / "macro_summary.csv", [dict(method=name, mode=m, **descriptor,
        eligible_images=len(v), complete_ber_images=sum(r["ber_percent"] is not None for r in v), **summarize(v)) for m, v in macro.items()])
    coverage = read_json(root / name / "coverage.json")
    write_csv(directory / "matching_coverage.csv", [dict(method=name, mode=m, planned=len(v),
        saved=sum(r["status"] == "saved" for r in v), unmatched=sum(r["status"] == "unmatched" for r in v))
        for m in ("native", "matched_39", "matched_42", "matched_45") if (v := [r for r in coverage if r["mode"] == m])])
    counter["planned_including_unmatched"] = len(coverage)*21
    counter["unmatched_excluded_evaluations"] = sum(r["status"] == "unmatched" for r in coverage)*21
    write_json(directory / "counts.json", dict(counter))
    return dict(counter)


def paired_report(root, methods):
    """Only common successful matching pairs with all repeats present."""
    import itertools
    result = []
    def read(name):
        path = root / name / "tables/case_levels.json.gz"
        return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []
    for left, right in itertools.combinations(methods, 2):
        a, b = read(left), read(right)
        key = lambda r: (r["mode"], r["family"], r["level"], r["image_id"], r["replicate"])
        maps = [{key(r): r for r in rows if r["mode"] != "native" and r["valid"] == r["planned"]} for rows in (a, b)]
        groups = defaultdict(list)
        for k in sorted(maps[0].keys() & maps[1].keys()):
            x, y = maps[0][k], maps[1][k]
            groups[k[:3]].append((k[3], x["ber_percent"]-y["ber_percent"]))
        for (mode, family, level), values in groups.items():
            image_values = defaultdict(list)
            for image, delta in values:
                image_values[image].append(delta)
            stats = summarize([{**{k: None for k in METRICS}, "ber_percent": mean(v)} for v in image_values.values()])
            result.append(dict(left=left, right=right, mode=mode, family=family, level=level,
                left_interface=a[0]["interface"], right_interface=b[0]["interface"],
                left_information_bits=a[0]["information_bits"], right_information_bits=b[0]["information_bits"],
                common_pairs=len(values), common_images=len(image_values), difference_pp=stats["ber_percent"],
                ci_low=stats["ber_percent_ci_low"], ci_high=stats["ber_percent_ci_high"]))
    write_csv(root / "_reports/pairwise_matched.csv", result)


def final_report(root, pairwise=False):
    root = Path(root)
    selection = read_json(root / "selection.json")
    rows = []
    for name in selection["models"]:
        counts, status = root / name / "tables/counts.json", root / name / "status.json"
        rows.append(dict(method=name, **(read_json(counts) if counts.exists() else {}),
                         **(read_json(status) if status.exists() else {"status": "not_started"})))
    folder = root / "_reports"
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "coverage.csv", rows)
    timing_rows = []
    for name in selection["models"]:
        for path in sorted((root / name / "timings").glob("*.json")):
            value = read_json(path)
            timing_rows.append(dict(method=name, stage_attempt=path.stem, wall_seconds=value.get("wall_seconds"),
                exit_code=value.get("exit_code"), reason=value.get("reason"), peak_rss_gib=value.get("peak_process_tree_rss_gib"),
                peak_gpu_gib=value.get("sampled_device_vram_peak_gib")))
    write_csv(folder / "performance.csv", timing_rows)
    lines = ["# Neural channel", "", f"Selected COCO images: {selection['n_images']}. Five full messages, four embedding modes, 21 conditions.", "",
             "| Method | Complete carriers | Eligible carriers | Blocked outputs | Status |", "|---|---:|---:|---:|---|"]
    lines += [f"| {r['method']} | {r.get('complete_carriers', 0)} | {r.get('carriers', 0)} | {r.get('blocked_generation', 0)} | {r['status']} |" for r in rows]
    lines += ["", "The input bank and digital experiment are unchanged. Completed attack packages are deleted only after verified durable decoding.",
        "BER/exact aggregates are percentages; paired deltas are percentage points. Detailed reference_and_clean.csv BER columns are fractions.",
        "Quality: source/native carrier in reference_and_clean.csv; 512-grid neural input/output quality in per_attack.csv.gz.",
        "Resize-only is an auxiliary control, not a 22nd condition. Clean is excluded from the equally weighted four-family macro.",
        "Missing, safety-blocked and technical failures are not BER=0 or BER=1. Inspect coverage alongside each aggregate.",
        "Five images test execution, not publication-level confidence. Author results are not invented or used as subtraction baselines."]
    atomic_bytes(folder / "REPORT.md", ("\n".join(lines)+"\n").encode())
    if pairwise:
        paired_report(root, selection["models"])
