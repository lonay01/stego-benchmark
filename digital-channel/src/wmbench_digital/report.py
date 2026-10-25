"""Streaming per-carrier results; image-cluster statistics, never bit pooling."""

from collections import defaultdict, Counter
import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import numpy as np

from wmbench.io import atomic_bytes, read_json, write_json, write_csv
from .spec import ATTACKS, POLICY
from .storage import read_result, result_path, complete

METRICS = ["ber_percent", "exact_percent", "paired_clean_ber_percent", "paired_attack_ber_percent",
           "delta_ber_pp", "paired_clean_exact_percent", "paired_attack_exact_percent", "exact_loss_pp",
           "decode_seconds"]
LEVELS = list(dict.fromkeys((a["family"], a["level"]) for a in ATTACKS))


def mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def interval(values):
    values = np.array([x for x in values if x is not None], dtype=float)
    if not len(values):
        return None, None, None
    rng = np.random.default_rng(POLICY["bootstrap"]["seed"])
    # Identical index stream for equal image counts. Pairing occurs BEFORE resampling.
    boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def comparison(row, clean):
    good = row.get("status") == "measured" and row.get("ber") is not None
    paired = good and clean.get("status") == "measured" and clean.get("ber") is not None
    exact_good = row.get("status") in {"measured", "rejected"}
    exact_paired = exact_good and clean.get("status") in {"measured", "rejected"}
    return dict(ber_percent=row["ber"]*100 if good else None,
                exact_percent=float(row["exact"])*100 if exact_good else None,
                paired_clean_ber_percent=clean["ber"]*100 if paired else None,
                paired_attack_ber_percent=row["ber"]*100 if paired else None,
                delta_ber_pp=(row["ber"]-clean["ber"])*100 if paired else None,
                paired_clean_exact_percent=float(clean["exact"])*100 if exact_paired else None,
                paired_attack_exact_percent=float(row["exact"])*100 if exact_paired else None,
                exact_loss_pp=(float(clean["exact"])-float(row["exact"]))*100 if exact_paired else None,
                decode_seconds=row.get("seconds"))


def summarize_images(rows):
    output = {}
    for metric in METRICS:
        avg, low, high = interval([r.get(metric) for r in rows])
        output.update({metric: avg, metric+"_ci_low": low, metric+"_ci_high": high,
                       metric+"_images": sum(r.get(metric) is not None for r in rows)})
    return output


def method_report(root, name):
    root = Path(root)
    cases = read_json(root / name / "cases.json")
    selection = read_json(root / "selection.json")
    # Index only currently selected image prefix, even when previous runs were larger.
    ids = list(dict.fromkeys(c["image_id"] for c in cases))[:selection["n_images"]]
    cases = [c for c in cases if c["image_id"] in ids]
    directory = root / name / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "per_attack.csv.gz.partial"
    fields = ["method", "mode", "image_id", "replicate", "attack", "family", "level", "repeat",
              "status", "reason", "information_bits", "interface", "image_sha256", *METRICS]
    cells, counter = [], Counter()
    quality_groups = defaultdict(list)
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            record = read_result(result_path(root, case), case)
            counter["carriers"] += 1
            counter["complete_carriers"] += int(complete(record))
            rows = record["rows"] if record else {}
            # Baseline can be audited bank Clean even while new extraction is missing.
            clean = rows.get("clean", case["clean"])
            groups = defaultdict(list)
            cap = case["assignment"]["capacity"]
            descriptor = dict(information_bits=cap["information_bits"],
                adapter_message_bits=len(case["assignment"]["payload"]), interface=cap["interface"],
                width=case["size"][0], height=case["size"][1],
                bits_per_pixel=cap["information_bits"]/(case["size"][0]*case["size"][1]))
            for attack in ATTACKS:
                row = rows.get(attack["id"], {"status": "missing"})
                metrics = comparison(row, clean)
                counter[row["status"]] += 1
                writer.writerow(dict(method=name, mode=case["mode"], image_id=case["image_id"],
                    replicate=case["replicate"], attack=attack["id"], family=attack["family"], level=attack["level"],
                    repeat=attack["repeat"], status=row["status"], reason=row.get("reason"),
                    information_bits=cap["information_bits"], interface=cap["interface"],
                    image_sha256=row.get("image_sha256"), **metrics))
                groups[(attack["family"], attack["level"])].append((row, metrics))
            for (family, level), values in groups.items():
                cell = dict(method=name, mode=case["mode"], image_id=case["image_id"], replicate=case["replicate"], **descriptor,
                            family=family, level=level, planned=len(values), valid=sum(r["status"] == "measured" for r, _ in values),
                            rejected=sum(r["status"] == "rejected" for r, _ in values),
                            errors=sum(r["status"] == "error" for r, _ in values),
                            missing=sum(r["status"] == "missing" for r, _ in values),
                            paired=sum(m["delta_ber_pp"] is not None for _, m in values),
                            exact_paired=sum(m["exact_loss_pp"] is not None for _, m in values),
                            **{k: mean([m[k] for _, m in values]) for k in METRICS})
                cells.append(cell)
            quality_path = root / name / "quality" / (case["id"]+".json")
            quality = read_json(quality_path) if quality_path.exists() else {}
            if quality and quality["case_key"] != case["key"]:
                raise ValueError("Quality record belongs to a different carrier")
            quality_groups[case["mode"]].append({"image_id": case["image_id"], "psnr": case["psnr"],
                "ssim": quality.get("ssim"), "lpips": quality.get("lpips"),
                "alpha": (case.get("matching") or {}).get("alpha"), "embedding_seconds": case["embedding_seconds"]})
    temporary.replace(directory / "per_attack.csv.gz")
    atomic_bytes(directory / "case_levels.json.gz", gzip.compress(json.dumps(cells).encode(), mtime=0))
    grouped = defaultdict(list)
    for cell in cells:
        grouped[(cell["mode"], cell["family"], cell["level"], cell["image_id"])].append(cell)
    images = []
    for (mode, family, level, image), values in grouped.items():
        images.append(dict(mode=mode, family=family, level=level, image_id=image,
            **{k: mean([r[k] for r in values]) for k in METRICS},
            **{k: sum(r[k] for r in values) for k in ("planned", "valid", "rejected", "errors", "missing", "paired", "exact_paired")}))
    by_level = defaultdict(list)
    for row in images:
        by_level[(row["mode"], row["family"], row["level"])].append(row)
    summaries = []
    for (mode, family, level), values in by_level.items():
        summaries.append(dict(method=name, mode=mode, family=family, level=level, images=len(values), **descriptor,
            **{k: sum(r[k] for r in values) for k in ("planned", "valid", "rejected", "errors", "missing", "paired", "exact_paired")},
            **summarize_images(values)))
    write_csv(directory / "attack_summary.csv", summaries)
    # Macro values require ALL planned observations for that metric. Missing levels
    # or rejected BER never disappear into an apparently complete overall score.
    macro_images = defaultdict(list)
    for mode in ("native", "matched_39", "matched_42", "matched_45"):
        for image in ids:
            rows = [r for r in images if r["mode"] == mode and r["image_id"] == image and r["family"] != "clean"]
            if not rows:
                continue
            out = {}
            for metric in METRICS:
                required = "exact_paired" if "exact" in metric else "paired" if "paired" in metric or metric == "delta_ber_pp" else "valid"
                valid = all(r[required] == r["planned"] and r[metric] is not None for r in rows)
                families = defaultdict(list)
                for r in rows:
                    families[r["family"]].append(r[metric])
                out[metric] = mean([mean(v) for v in families.values()]) if valid and len(families) == 8 else None
            macro_images[mode].append(out)
    write_csv(directory / "macro_summary.csv", [dict(method=name, mode=m, eligible_images=len(v), **descriptor,
        complete_ber_images=sum(r["ber_percent"] is not None for r in v),
        macro_ber_complete=all(r["ber_percent"] is not None for r in v),
        note="Check contributing counts: incomplete images have no macro metric", **summarize_images(v))
        for m, v in macro_images.items()])
    coverage = read_json(root / name / "coverage.json")
    clean_table = []
    for mode, values in quality_groups.items():
        config = next(c for c in cases if c["mode"] == mode)
        cap = config["assignment"]["capacity"]
        row = dict(method=name, mode=mode, width=config["size"][0], height=config["size"][1],
                   information_bits=cap["information_bits"], interface=cap["interface"],
                   bits_per_pixel=cap["information_bits"]/(config["size"][0]*config["size"][1]),
                   payload_contract=json.dumps(cap), planned_bank_cases=len([c for c in coverage if c["mode"] == mode]),
                   saved_bank_cases=len(values), unmatched=sum(c["status"] == "unmatched" for c in coverage if c["mode"] == mode),
                   quality_measured=sum(v["ssim"] is not None and v["lpips"] is not None for v in values),
                   author_psnr="not verified", author_clean_ber="not verified", author_comparability="context only; no numeric claim")
        for metric in ("psnr", "ssim", "lpips", "alpha", "embedding_seconds"):
            per_image = defaultdict(list)
            for v in values:
                value = v.get(metric)
                if isinstance(value, (float, int)) or value == "Infinity":
                    per_image[v["image_id"]].append(float(value))
            row[metric] = mean([mean(v) for v in per_image.values()])
        baseline = next(r for r in summaries if r["mode"] == mode and r["family"] == "clean")
        row.update({k: v for k, v in baseline.items() if k not in {"method", "mode", "family", "level"}})
        clean_table.append(row)
    write_csv(directory / "reference_and_clean.csv", clean_table)
    write_json(directory / "counts.json", dict(counter))
    return dict(counter)


def final_report(root, pairwise=False):
    root = Path(root)
    selection = read_json(root / "selection.json")
    summaries = []
    for name in selection["models"]:
        path = root / name / "tables/counts.json"
        status = read_json(root / name / "status.json") if (root / name / "status.json").exists() else {"status": "not_started"}
        summaries.append({"method": name, **(read_json(path) if path.exists() else {}), "status": status["status"],
                          "reason": status.get("reason", "")})
    directory = root / "_reports"
    directory.mkdir(exist_ok=True)
    write_csv(directory / "coverage.csv", summaries)
    text = ["# Digital channel", "", f"Selected images: {selection['n_images']}. Five full messages per image; 79 conditions per saved carrier.", "",
            "| Method | Complete carriers | Saved carriers | Status |", "|---|---:|---:|---|"]
    text += [f"| {r['method']} | {r.get('complete_carriers', 0)} | {r.get('carriers', 0)} | {r['status']} |" for r in summaries]
    text += ["", "Per-method tables: `tables/reference_and_clean.csv`, `attack_summary.csv`, `macro_summary.csv`, `per_attack.csv.gz`.",
             "", "BER/exact values are percentages; deltas are percentage points. Confidence intervals resample images (2,000 paired resamples).",
             "Author numbers are not verified in this release and are not invented. This is not a reproduction claim.",
             "A completed 5-image canary validates plumbing, not statistical robustness. Review coverage and Clean before increasing N."]
    atomic_bytes(directory / "REPORT.md", ("\n".join(text)+"\n").encode())
    if pairwise:
        pairwise_report(root, selection["models"])


def pairwise_report(root, methods):
    """Common image/message pairs for Matched Quality; exclude incomplete repeats."""
    def load(name):
        path = root / name / "tables/case_levels.json.gz"
        if not path.exists():
            return {}
        return {(r["mode"], r["family"], r["level"], r["image_id"], r["replicate"]): r
                for r in json.loads(gzip.decompress(path.read_bytes())) if r["mode"] != "native"}
    rows = []
    for left, right in itertools.combinations(methods, 2):
        print(f"PAIRWISE REPORT {left} / {right}", flush=True)
        a, b = load(left), load(right)
        groups = defaultdict(list)
        for key in a.keys() & b.keys():
            x, y = a[key], b[key]
            if x["valid"] == x["planned"] and y["valid"] == y["planned"]:
                groups[key[:3]].append((key[3], x["ber_percent"], y["ber_percent"]))
        for (mode, family, level), values in groups.items():
            images = defaultdict(list)
            for image, x, y in values:
                images[image].append((x, y))
            means = np.array([np.mean(v, axis=0) for v in images.values()])
            avg, low, high = interval(means[:, 0]-means[:, 1])
            rows.append(dict(left=left, right=right, mode=mode, family=family, level=level,
                             left_interface=next(iter(a.values()))["interface"], right_interface=next(iter(b.values()))["interface"],
                             left_information_bits=next(iter(a.values()))["information_bits"],
                             right_information_bits=next(iter(b.values()))["information_bits"],
                             common_pairs=len(values), common_images=len(images), left_ber_percent=float(means[:, 0].mean()),
                             right_ber_percent=float(means[:, 1].mean()), difference_pp=avg, ci_low=low, ci_high=high,
                             note="Common matched and fully decoded pairs only; see matching/rejection coverage separately"))
    write_csv(root / "_reports/pairwise_matched.csv", rows)
