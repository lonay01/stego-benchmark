"""Coverage-first reporting, image-cluster intervals and independent negative calibration."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .io import digest, read_json, write_csv, write_json
from .protocol import BASE_SEED
from .store import JobStore


def conservative_threshold(scores: list[float], target_fpr: float = 0.01) -> dict:
    values = np.asarray(scores, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration requires finite independent negative scores")
    allowed = int(np.floor(len(values) * target_fpr))
    # Detection is score >= threshold; ties are handled conservatively by nextafter.
    threshold = float(np.nextafter(np.sort(values)[::-1][allowed], np.inf))
    return {
        "threshold": threshold,
        "comparison": ">=",
        "target_clean_fpr": target_fpr,
        "calibration_count": len(values),
        "achieved_clean_fpr": float(np.mean(values >= threshold)),
        "calibration_unit": "independent image",
        "adequate_for_core": len(values) >= 1000,
    }


def image_bootstrap(values: dict[str, list[float]], repeats: int = 10000) -> dict:
    means = np.array([np.mean(v) for _, v in sorted(values.items()) if v], dtype=float)
    if len(means) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "images": 0}
    rng = np.random.default_rng(BASE_SEED)
    samples = np.empty(repeats)
    for start in range(0, repeats, 1000):
        end = min(start + 1000, repeats)
        indices = rng.integers(0, len(means), size=(end - start, len(means)))
        samples[start:end] = means[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(means.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "images": len(means),
    }


def build_report(
    store: JobStore,
    planned: list[dict],
    config: dict,
    inventory: list[dict],
    embedding_units: int,
    input_count: int,
    received_count: int,
):
    validation, records, failures, missing = [], [], [], []
    for spec in planned:
        record = store.valid(spec)
        if record is None and store.path(spec).exists():
            raw = read_json(store.path(spec))
            if raw.get("status") == "failed":
                record = raw
        status = record["status"] if record else "missing_or_invalid"
        row = {
            "job_id": store.identifier(spec),
            "stage": spec["stage"],
            "model_id": spec["model_id"],
            "image_id": spec.get("image_id"),
            "mode": spec.get("mode"),
            "status": status,
        }
        validation.append(row)
        if record:
            records.append(record)
            if status != "ok":
                failures.append({**row, "details": record["data"]})
        else:
            missing.append(row)
    tables = store.root / "tables"
    for name, rows in (
        ("validation_report", validation),
        ("failed_jobs", failures),
        ("missing_jobs", missing),
    ):
        write_csv(tables / f"{name}.csv", rows)
    by_stage = defaultdict(list)
    decodes = []
    for record in records:
        spec = record["spec"]
        row = {
            "job_id": record["job_id"],
            "status": record["status"],
            "seconds": record["seconds"],
            "model_id": spec["model_id"],
            "environment": spec["environment"],
            "operator": spec["operator"],
            "image_id": spec.get("image_id"),
            "mode": spec.get("mode"),
            "track": spec.get("track"),
            "group": spec.get("group"),
            "role": spec.get("role"),
            "attack": spec.get("attack", {}).get("id", "identity"),
            "family": spec.get("attack", {}).get("family", "identity"),
            "repeat": spec.get("attack", {}).get("repeat", 0),
            "seed": spec.get("seed"),
            "comparison": spec.get("comparison"),
            **record["data"],
        }
        by_stage[spec["stage"]].append(row)
        if spec["stage"] == "decode" and record["status"] == "ok":
            decodes.append(row)
    names = {
        "embed": "embedding_native",
        "origin": "origin_generation",
        "origin_clean": "origin_calibration_generation",
        "match": "embedding_matched",
        "attack": "generation_results",
        "decode": "decode_results",
        "quality": "quality_results",
    }
    for stage, name in names.items():
        write_csv(tables / f"{name}.csv", by_stage[stage])
    write_csv(tables / "embedding_results.csv", by_stage["embed"] + by_stage["match"])
    write_csv(tables / "negative_controls.csv", [r for r in decodes if r["role"] != "positive"])
    calibration = defaultdict(list)
    for row in decodes:
        if row["role"] == "calibration" and row.get("score") is not None:
            calibration[(row["model_id"], row.get("score_name", "adapter_score"))].append(row)
    thresholds = {}
    for (model_id, score_name), rows in calibration.items():
        if len({r["image_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate calibration image; cannot count it as independent")
        key = model_id + ":" + score_name
        thresholds[key] = {
            **conservative_threshold([r["score"] for r in rows]),
            "score_name": score_name,
            "model_id": model_id,
            "calibration_jobs_sha256": digest(sorted(r["job_id"] for r in rows)),
        }
    write_json(tables / "calibration_thresholds.json", thresholds)
    groups = defaultdict(list)
    for row in decodes:
        if row["role"] in ("positive", "negative_test"):
            groups[(row["model_id"], row["mode"], row["attack"], row["group"], row["role"])].append(
                row
            )
    summaries = []
    for (model_id, mode, attack, group, role), rows in sorted(groups.items()):
        output = {
            "model_id": model_id,
            "mode": mode,
            "attack": attack,
            "group": group,
            "role": role,
            "family": rows[0]["family"],
            "rows": len(rows),
            "images": len({r["image_id"] for r in rows}),
            "expected_images": config["image_limit"],
        }
        for metric in ("ber", "exact", "native_ber", "native_exact", "score"):
            clusters = defaultdict(list)
            for row in rows:
                if row.get(metric) is not None:
                    clusters[row["image_id"]].append(row[metric])
            estimate = image_bootstrap(clusters, config.get("bootstrap_repeats", 10000))
            output.update(
                {f"{metric}_{key}": value for key, value in estimate.items() if key != "images"}
            )
        key = model_id + ":" + rows[0].get("score_name", "adapter_score")
        if key in thresholds:
            threshold = thresholds[key]["threshold"]
            clusters = defaultdict(list)
            for row in rows:
                if row.get("score") is not None:
                    clusters[row["image_id"]].append(int(row["score"] >= threshold))
            metric = "tpr_at_calibrated_clean_threshold" if role == "positive" else "observed_fpr"
            output.update(
                {
                    metric + "_" + k: v
                    for k, v in image_bootstrap(
                        clusters, config.get("bootstrap_repeats", 10000)
                    ).items()
                    if k != "images"
                }
            )
            output["threshold_id"] = digest(thresholds[key])
            # Bootstrap alone degenerates at all-zero FPR. Also expose the conservative
            # image-level any-positive interval, not a fictitious per-repeat sample size.
            if role == "negative_test":
                flags = [int(any(values)) for values in clusters.values()]
                n, k = len(flags), sum(flags)
                z = 1.959963984540054
                p = k / n
                denominator = 1 + z * z / n
                center = (p + z * z / (2 * n)) / denominator
                half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
                output.update(
                    {
                        "negative_images_any_false_positive": k,
                        "negative_independent_images": n,
                        "any_false_positive_image_rate_wilson95_low": float(max(0, center - half)),
                        "any_false_positive_image_rate_wilson95_high": float(min(1, center + half)),
                    }
                )
        else:
            output["threshold_status"] = "not_calibrated_no_TPR_claim"
        if any("detected" in row for row in rows):
            decisions = defaultdict(list)
            for row in rows:
                if "detected" in row:
                    decisions[row["image_id"]].append(int(row["detected"]))
            label = "native_detector_tpr" if role == "positive" else "native_detector_observed_fpr"
            output.update(
                {
                    label + "_" + k: v
                    for k, v in image_bootstrap(
                        decisions, config.get("bootstrap_repeats", 10000)
                    ).items()
                    if k != "images"
                }
            )
            output["native_detector_note"] = (
                "author fixed decision; not an empirically calibrated 1% threshold"
            )
        expected_repeats = (
            1 if rows[0]["family"] in ("identity", "vae") else config.get("repeats", 1)
        )
        output["expected_rows"] = config["image_limit"] * expected_repeats
        actual_pairs = {(r["image_id"], r["repeat"]) for r in rows}
        output["coverage_complete"] = (
            output["images"] == config["image_limit"]
            and len(actual_pairs) == output["expected_rows"]
        )
        summaries.append(output)
    write_csv(tables / "summary_by_model_mode_attack.csv", summaries)
    # Family means: images equal weight; each attack level equal weight; identity excluded.
    family_cells = defaultdict(list)
    for row in summaries:
        if (
            row["role"] == "positive"
            and row["family"] != "identity"
            and row.get("ber_mean") is not None
        ):
            family_cells[(row["model_id"], row["mode"], row["group"], row["family"])].append(row)
    from .protocol import attack_manifest

    expected_attacks = defaultdict(set)
    for attack in attack_manifest(config.get("repeats", 1), config.get("identity_only", False)):
        expected_attacks[attack["family"]].add(attack["id"])
    family_rows = [
        {
            "model_id": k[0],
            "mode": k[1],
            "group": k[2],
            "family": k[3],
            "ber_mean": float(np.mean([r["ber_mean"] for r in rows])),
            "attack_levels": len(rows),
            "expected_attack_levels": len(expected_attacks[k[3]]),
            "coverage_complete": (
                all(r["coverage_complete"] for r in rows)
                and {r["attack"] for r in rows} == expected_attacks[k[3]]
            ),
        }
        for k, rows in sorted(family_cells.items())
    ]
    write_csv(tables / "summary_by_family.csv", family_rows)
    # Paired mode effects: average repeats within image, then bootstrap paired image differences.
    mode_cells = defaultdict(lambda: defaultdict(dict))
    for row in decodes:
        if row["role"] == "positive" and row.get("ber") is not None:
            key = (row["model_id"], row["attack"], row["group"], row["image_id"])
            mode_cells[key][row["mode"]][(row["repeat"], row["seed"])] = row["ber"]
    pairs = defaultdict(dict)
    for (model_id, attack, group, image_id), modes in mode_cells.items():
        if "native" not in modes:
            continue
        for mode in modes:
            if mode != "native":
                shared = sorted(set(modes[mode]) & set(modes["native"]))
                if shared:
                    pairs[(model_id, attack, group, mode)][image_id] = [
                        float(np.mean([modes[mode][r] - modes["native"][r] for r in shared]))
                    ]
    write_csv(
        tables / "paired_mode_differences.csv",
        [
            {
                "model_id": k[0],
                "attack": k[1],
                "group": k[2],
                "matched_mode": k[3],
                "contrast": "BER_matched_minus_BER_native",
                **image_bootstrap(values, config.get("bootstrap_repeats", 10000)),
            }
            for k, values in sorted(pairs.items())
        ],
    )
    timing_groups = defaultdict(list)
    for stage, rows in by_stage.items():
        for row in rows:
            timing_groups[(stage, row["operator"], row["attack"])].append(row)
    runtime = [
        {
            "stage": key[0],
            "operator": key[1],
            "attack": key[2],
            "rows": len(rows),
            "seconds_total": sum(r["seconds"] for r in rows),
            "seconds_median": float(np.median([r["seconds"] for r in rows])),
            "seconds_p95": float(np.quantile([r["seconds"] for r in rows], 0.95)),
        }
        for key, rows in sorted(timing_groups.items())
    ]
    write_csv(tables / "runtime_report.csv", runtime)
    unresolved = [
        r for r in inventory if r["status"] in ("integration_required", "needs_manual_asset")
    ]
    measurements_complete = (
        not failures
        and not missing
        and all(store.valid(s, include_terminal=False) for s in planned)
    )
    score_models = {
        row["model_id"]
        for row in decodes
        if row["role"] == "positive" and row.get("score") is not None
    }
    calibrated_models = {v["model_id"] for v in thresholds.values() if v["adequate_for_core"]}
    calibration_complete = bool(score_models) and score_models <= calibrated_models
    readiness = {
        "planned_jobs": len(planned),
        "valid_ok": sum(r["status"] == "ok" for r in records),
        "terminal_non_ok": len(failures),
        "missing": len(missing),
        "requested_models": len(inventory),
        "embedding_units_expected": embedding_units,
        "marked_inputs_obtained": input_count,
        "received_inputs_obtained": received_count,
        "all_planned_measurements_obtained": measurements_complete,
        "calibration_complete_for_available_scores": calibration_complete,
        "origin_calibration": "own_backbone_generated_clean_corpus_from_heldout_captions_when_configured",
        "catalog_manual_or_unintegrated": unresolved,
        "full_protocol_complete": False,
        "report_kind": "partial_or_smoke_coverage_report",
        "note": "No universal cross-track leaderboard. GPU validation, independent calibration and blinded quality review are separate completion gates.",
    }
    write_json(tables / "completion.json", readiness)
    # Blind quality audit hides model/mode/BER; key is kept separately.
    candidates = defaultdict(list)
    for record in records:
        if record["spec"]["stage"] == "attack" and record["status"] == "ok":
            candidates[record["spec"]["attack"]["family"]].append(record)
    audit, audit_key = [], []
    rng = np.random.default_rng(BASE_SEED)
    for family, rows in sorted(candidates.items()):
        count = min(len(rows), max(30, int(np.ceil(0.1 * len(rows)))))
        for index in rng.choice(len(rows), count, replace=False):
            record = rows[index]
            anonymous = digest({"job": record["job_id"], "audit": BASE_SEED})[:16]
            audit.append(
                {
                    "audit_id": anonymous,
                    "image_path": record["data"]["image_path"],
                    "prompt": record["spec"]["attack"].get("prompt", record["spec"]["prompt"]),
                    "content_preserved": "",
                    "instruction_fulfilled": "",
                    "major_defects": "",
                }
            )
            audit_key.append({"audit_id": anonymous, "job_id": record["job_id"], "family": family})
    review_path = tables / "blind_quality_review.csv"
    if review_path.exists():
        import csv

        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {r["audit_id"]: r for r in csv.DictReader(handle)}
        for row in audit:
            if row["audit_id"] in existing:
                for field in ("content_preserved", "instruction_fulfilled", "major_defects"):
                    row[field] = existing[row["audit_id"]].get(field, "")
        # Keep annotations on formerly sampled items, even when a later report adds images.
        retained = {r["audit_id"] for r in audit}
        audit.extend(r for key, r in existing.items() if key not in retained)
    write_csv(review_path, audit)
    key_path = tables / "blind_quality_review_key_DO_NOT_SHOW_REVIEWER.csv"
    if key_path.exists():
        import csv

        with key_path.open(encoding="utf-8-sig", newline="") as handle:
            existing_key = {r["audit_id"]: r for r in csv.DictReader(handle)}
        existing_key.update({r["audit_id"]: r for r in audit_key})
        audit_key = list(existing_key.values())
    write_csv(key_path, audit_key)
    review_complete = bool(audit) and all(
        all(
            row.get(field, "") != ""
            for field in ("content_preserved", "instruction_fulfilled", "major_defects")
        )
        for row in audit
    )
    readiness["blind_quality_review_complete"] = review_complete
    readiness["full_protocol_complete"] = (
        measurements_complete
        and calibration_complete
        and review_complete
        and config.get("image_limit", 0) >= 100
        and config.get("quality_metrics", False)
    )
    readiness["report_kind"] = (
        "complete_configured_protocol"
        if readiness["full_protocol_complete"]
        else "partial_or_smoke_coverage_report"
    )
    write_json(tables / "completion.json", readiness)
    print(
        f"Report: {tables}; {readiness['valid_ok']}/{len(planned)} valid jobs. See completion.json.",
        flush=True,
    )
