"""Strict coverage accounting for the complete, portable neural-channel protocol.

The materialized DAG is not the experimental denominator: an upstream failure
must not make every unplanned descendant disappear from the coverage report.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .io import atomic_bytes, read_json, sha256, write_csv, write_json
from .protocol import attack_manifest, paired_seed
from .report import build_report, image_bootstrap
from .store import JobStore

INPUT_FIELDS = ("input_path", "source_path", "first_path", "second_path", "metadata_path")
UNAVAILABLE = {
    "blocked",
    "unavailable",
    "unsupported",
    "integration_required",
    "needs_manual_asset",
    "hardware_blocked",
    "blocked_prerequisite",
}


class _ValidatedStore:
    """Reuse one validation pass while the ordinary report reads the same records."""

    def __init__(self, store: JobStore, planned: list[dict]):
        self.store = store
        self.root = store.root
        self.records = {}
        self.file_hashes = {}
        for spec in planned:
            record = store.valid(spec)
            if record is not None:
                for field in INPUT_FIELDS:
                    expected = spec.get(field + "_sha256")
                    if field in spec and expected:
                        path = str(spec[field])
                        if path not in self.file_hashes:
                            try:
                                self.file_hashes[path] = sha256(path)
                            except OSError:
                                self.file_hashes[path] = None
                        if self.file_hashes[path] != expected:
                            record = None
                            break
            self.records[store.identifier(spec)] = record

    def valid(self, spec: dict, include_terminal: bool = True) -> dict | None:
        record = self.records.get(self.store.identifier(spec))
        if record is not None and (include_terminal or record["status"] == "ok"):
            return record
        return None

    def path(self, spec: dict) -> Path:
        return self.store.path(spec)

    def identifier(self, spec: dict) -> str:
        return self.store.identifier(spec)


def _manifest_ids(path, count: int | None = None) -> tuple[set[str], str | None]:
    try:
        images = read_json(path)["images"]
        if count is not None:
            if len(images) < count:
                return set(), "manifest has fewer images than requested"
            images = images[:count]
        ids = [str(image["image_id"]) for image in images]
        if len(ids) != len(set(ids)):
            return set(), "manifest contains duplicate image IDs"
        return set(ids), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return set(), f"{type(error).__name__}: {error}"


def _availability(model: dict, dependencies: dict) -> tuple[bool, str, str]:
    value = dependencies.get(model["id"], {})
    status = value.get("status", "not_started") if isinstance(value, dict) else str(value)
    reason = value.get("reason", "") if isinstance(value, dict) else ""
    catalogue = model.get("status", model.get("capability", {}).get("status", ""))
    if model.get("track") not in ("posthoc", "origin"):
        return False, "unsupported_carrier", reason or "Not a COCO image/prompt carrier"
    if status in UNAVAILABLE or (
        catalogue == "integration_required" and status not in ("ready", "passed", "ok")
    ):
        return False, status if status in UNAVAILABLE else catalogue, reason or catalogue
    # Pending/failed preparations remain part of the executable denominator.
    return True, status, reason


def _coordinate(record: dict, attack: dict, expected_ids: set[str]) -> tuple | None:
    spec = record["spec"]
    image_id = str(spec.get("image_id", ""))
    actual = spec.get("attack", {"id": "identity", "repeat": 0})
    repeat = actual.get("repeat", 0)
    if image_id not in expected_ids or actual.get("id") != attack["id"]:
        return None
    if repeat not in attack["repeats"]:
        return None
    # Old decodes from a different seed schedule are not paired evidence.
    if spec.get("stage") == "decode" and spec.get("seed") != paired_seed(
        image_id, attack["id"], repeat
    ):
        return None
    return image_id, repeat


def _pairs(records: list[dict], attack: dict, ids: set[str]) -> set[tuple]:
    return {point for record in records if (point := _coordinate(record, attack, ids)) is not None}


def _quality_metrics_present(record: dict) -> bool:
    data = record["data"]
    required = (
        "ssim",
        "lpips",
        "dino_cosine",
        "clip_target_before",
        "clip_target_after",
        "clip_target_delta",
    )
    if record["spec"].get("attack", {}).get("family") == "inpainting":
        required += ("mean_abs_change_inside", "mean_abs_change_outside")
    return (
        data.get("psnr") == "Infinity"
        or (isinstance(data.get("psnr"), (int, float)) and math.isfinite(data["psnr"]))
    ) and all(
        isinstance(data.get(key), (int, float)) and math.isfinite(data[key]) for key in required
    )


def _timing_rows(records: list[dict], config: dict) -> list[dict]:
    """Observed same-environment medians; not a hardware-speed or wall-time promise."""
    groups = defaultdict(list)
    for record in records:
        seconds = record.get("seconds")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            continue
        spec = record["spec"]
        provenance = record.get("data", {}).get("execution_provenance") or {}
        groups[
            (
                spec["stage"],
                spec.get("operator"),
                spec.get("model_id"),
                spec.get("environment"),
                spec.get("role", ""),
                spec.get("attack", {}).get("id", "identity"),
                provenance.get("dependency_execution_fingerprint", spec.get("environment")),
            )
        ].append(record)
    rows = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(values, key=lambda record: record.get("finished_unix", 0))
        warm = [record["seconds"] for record in ordered[1:]]
        is_calibration = key[4] == "calibration" or key[0] == "origin_clean"
        estimate = None
        if warm:
            scale = 1 if is_calibration else 100 / config["image_limit"]
            estimate = median(warm) * len(values) * scale
        rows.append(
            {
                "stage": key[0],
                "operator": key[1],
                "model_id": key[2],
                "environment": key[3],
                "execution_environment": key[6],
                "role": key[4],
                "attack": key[5],
                "observed_jobs": len(values),
                "observed_seconds_total": sum(record["seconds"] for record in values),
                "first_observed_job_seconds": ordered[0]["seconds"],
                "warm_observed_jobs": len(warm),
                "warm_seconds_median": median(warm) if warm else None,
                "conditional_observed_work_scaled_to_100_seconds": estimate,
                "calibration_is_fixed_cost": is_calibration,
                "estimate_note": "Observed subset only; excludes missing work, downloads, initialization, I/O overhead, and different-hardware speed. First observed job excluded from warm proxy; not a guaranteed warm measurement.",
            }
        )
    return rows


def build_full_report(
    store: JobStore,
    planned: list[dict],
    config: dict,
    inventory: list[dict],
    dependency_status: dict,
) -> dict:
    """Write expected-cell coverage, strict completion and existing metric tables.

    All denominators are fixed by configuration/manifests, never by surviving
    records. Synthetic unit tests exercise this function without implying GPU
    validation or measured watermark performance.
    """
    planned = list({store.identifier(spec): spec for spec in planned}.values())
    checked = _ValidatedStore(store, planned)
    good = [record for record in checked.records.values() if record and record["status"] == "ok"]
    count = int(config["image_limit"])
    calibration_count = int(config.get("calibration_count", 1000))
    test_ids, test_error = _manifest_ids(config.get("dataset_manifest"), count)
    calibration_ids, calibration_error = _manifest_ids(config.get("negative_calibration_manifest"))
    if len(calibration_ids) != calibration_count:
        calibration_error = (
            calibration_error or "calibration manifest size differs from configuration"
        )
    if test_ids & calibration_ids:
        calibration_error = "calibration and test IDs overlap"
    manifest_errors = {
        key: value
        for key, value in (("test", test_error), ("calibration", calibration_error))
        if value
    }
    attacks_by_id = {}
    for attack in attack_manifest(config.get("repeats", 3), config.get("identity_only", False)):
        row = attacks_by_id.setdefault(attack["id"], {**attack, "repeats": set()})
        row["repeats"].add(attack["repeat"])
    indexed, by_model, clean_quality_index = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in good:
        spec = record["spec"]
        by_model[spec["model_id"]].append(record)
        if spec["stage"] == "quality" and spec.get("comparison") == "channel_negative_test":
            clean_quality_index[spec["model_id"]].append(record)
        indexed[
            (
                spec["model_id"],
                spec["stage"],
                spec.get("mode"),
                spec.get("role"),
                spec.get("comparison"),
            )
        ].append(record)

    base_error = None
    try:
        build_report(
            checked,
            planned,
            config,
            inventory,
            count * len(inventory),
            sum(record["spec"]["stage"] in ("embed", "origin", "match") for record in good),
            sum(
                record["spec"]["stage"] == "decode" and record["spec"].get("role") != "calibration"
                for record in good
            ),
        )
    except Exception as error:
        # Preserve coverage diagnostics even when malformed/calibration records
        # prevent statistical tables. Such a report can never be complete.
        base_error = f"{type(error).__name__}: {error}"

    coverage, applicability, calibration_rows, collisions, paired = [], [], [], [], []
    for model in inventory:
        model_id = model["id"]
        model_records = by_model[model_id]
        available, dependency_state, reason = _availability(model, dependency_status)
        applicability.append(
            {
                "model_id": model_id,
                "track": model["track"],
                "available_for_this_run": available,
                "dependency_status": dependency_state,
                "reason": reason,
                "native_applicable": model["track"] in ("posthoc", "origin"),
                "matched42_applicable": model["track"] == "posthoc",
            }
        )
        controls = [
            record
            for record in model_records
            if record["spec"]["stage"] == "decode" and record["spec"].get("role") == "negative_test"
        ]
        calibration = [
            record
            for record in model_records
            if record["spec"]["stage"] == "decode" and record["spec"].get("role") == "calibration"
        ]
        calibration_seen = [str(record["spec"].get("image_id")) for record in calibration]
        clean_generated_ids = {
            str(record["spec"].get("image_id"))
            for record in model_records
            if record["spec"]["stage"] == "origin_clean"
        }
        clean_origin_complete = model["track"] != "origin" or clean_generated_ids == calibration_ids
        score_names = {
            record["data"].get("score_name", "adapter_score")
            for record in calibration
            if isinstance(record["data"].get("score"), (int, float))
            and math.isfinite(record["data"]["score"])
        }
        scored = (
            len(score_names) == 1
            and bool(calibration)
            and all(
                isinstance(record["data"].get("score"), (int, float))
                and math.isfinite(record["data"]["score"])
                for record in calibration
            )
        )
        decision_only = bool(calibration) and all(
            record["data"].get("score") is None
            and isinstance(record["data"].get("detected"), (bool, int))
            and record["data"]["detected"] in (0, 1)
            for record in calibration
        )
        calibration_ok = (
            not calibration_error
            and len(calibration_seen) == calibration_count
            and set(calibration_seen) == calibration_ids
            and (scored or decision_only)
            and clean_origin_complete
        )
        calibration_rows.append(
            {
                "model_id": model_id,
                "available_for_this_run": available,
                "expected_independent_images": calibration_count,
                "observed_independent_images": len(set(calibration_seen) & calibration_ids),
                "observed_rows": len(calibration),
                "coverage_complete": calibration_ok,
                "at_least_1000_independent_negatives": calibration_count >= 1000,
                "score_names": sorted(score_names),
                "origin_clean_generation_complete": clean_origin_complete,
                "calibration_kind": "empirical_score_threshold"
                if scored
                else (
                    "author_fixed_decision_negative_audit"
                    if decision_only
                    else "missing_or_mixed_outputs"
                ),
                "calibrated_score_threshold_applicable": scored,
                "fixed_decision_false_positives": sum(
                    bool(record["data"]["detected"]) for record in calibration
                )
                if decision_only
                else None,
                "note": "Fixed author decisions are audited, not converted into a fitted 1% threshold.",
            }
        )
        for record in controls:
            spec, data = record["spec"], record["data"]
            if spec.get("attack", {}).get("id", "identity") == "identity" and (
                data.get("exact") == 1 or data.get("native_exact") == 1
            ):
                collisions.append(
                    {
                        "model_id": model_id,
                        "image_id": spec.get("image_id"),
                        "job_id": record["job_id"],
                        "common_exact": data.get("exact"),
                        "native_exact": data.get("native_exact"),
                        "status": "review_required",
                        "reason": "Unmarked identity control exactly matches the queried bit message. Investigate decoder/key contract; this is not proof of leakage and must not be rerun until favorable.",
                    }
                )
        for mode in ("native", "matched42"):
            supported = model["track"] == "posthoc" or (
                model["track"] == "origin" and mode == "native"
            )
            positives = indexed[(model_id, "decode", mode, "positive", None)]
            embedded = [
                record
                for record in model_records
                if record["spec"]["stage"]
                in (("embed", "origin") if mode == "native" else ("match",))
            ]
            embedded_ids = {str(record["spec"].get("image_id")) for record in embedded} & test_ids
            for attack in attacks_by_id.values():
                expected = {
                    (image_id, repeat) for image_id in test_ids for repeat in attack["repeats"]
                }
                positive_pairs, control_pairs = (
                    _pairs(positives, attack, test_ids),
                    _pairs(controls, attack, test_ids),
                )
                positive_observations = sum(
                    _coordinate(record, attack, test_ids) is not None for record in positives
                )
                control_observations = sum(
                    _coordinate(record, attack, test_ids) is not None for record in controls
                )
                quality_expected = attack["id"] != "identity"
                comparisons = (
                    ("channel_positive", "paired_marked_vs_clean_output")
                    if quality_expected
                    else (() if model["track"] == "origin" else ("embedding",))
                )
                quality_details = {}
                for comparison in comparisons:
                    records = [
                        record
                        for record in indexed[(model_id, "quality", mode, None, comparison)]
                        if _quality_metrics_present(record)
                    ]
                    quality_details[comparison] = len(_pairs(records, attack, test_ids))
                if quality_expected:
                    clean_quality = clean_quality_index[model_id]
                    if model["track"] == "posthoc":
                        clean_quality = clean_quality + clean_quality_index["clean_control"]
                    clean_quality = [
                        record for record in clean_quality if _quality_metrics_present(record)
                    ]
                    quality_details["channel_negative_test"] = len(
                        _pairs(clean_quality, attack, test_ids)
                    )
                expected_count = count * len(attack["repeats"])
                quality_ok = bool(config.get("quality_metrics", False)) and all(
                    value == expected_count for value in quality_details.values()
                )
                complete = (
                    supported
                    and available
                    and not test_error
                    and len(embedded_ids) == count
                    and positive_pairs == expected
                    and control_pairs == expected
                    and positive_observations == len(expected)
                    and control_observations == len(expected)
                    and quality_ok
                )
                status = (
                    "unsupported_mode"
                    if not supported
                    else "unavailable"
                    if not available
                    else "complete"
                    if complete
                    else "incomplete"
                )
                coverage.append(
                    {
                        "model_id": model_id,
                        "track": model["track"],
                        "mode": mode,
                        "attack": attack["id"],
                        "family": attack["family"],
                        "applicable": supported,
                        "available_for_this_run": available,
                        "status": status,
                        "reason": reason
                        if supported
                        else (
                            "Generation-origin methods are Native-only; scalar residual matching is not defined"
                            if model["track"] == "origin"
                            else "Other carrier; requires a separate protocol"
                        ),
                        "expected_images": count,
                        "expected_repeats": len(attack["repeats"]),
                        "expected_positive_decodes": expected_count if supported else 0,
                        "expected_control_decodes": expected_count if supported else 0,
                        "observed_positive_decodes": len(positive_pairs) if supported else 0,
                        "observed_control_decodes": len(control_pairs) if supported else 0,
                        "duplicate_positive_coordinates": positive_observations
                        - len(positive_pairs),
                        "duplicate_control_coordinates": control_observations - len(control_pairs),
                        "control_reused_across_modes": model["track"] == "posthoc",
                        "observed_embedding_images": len(embedded_ids) if supported else 0,
                        "quality_comparisons_observed": quality_details if supported else {},
                        "quality_complete": quality_ok if supported else False,
                        "coverage_complete": complete,
                    }
                )
                if mode == "matched42" and supported:
                    native = indexed[(model_id, "decode", "native", "positive", None)]
                    native_values = {
                        _coordinate(record, attack, test_ids): record["data"].get("ber")
                        for record in native
                    }
                    matched_values = {
                        _coordinate(record, attack, test_ids): record["data"].get("ber")
                        for record in positives
                    }
                    shared = {
                        point
                        for point in expected
                        if isinstance(native_values.get(point), (int, float))
                        and isinstance(matched_values.get(point), (int, float))
                    }
                    differences = defaultdict(list)
                    for point in shared:
                        differences[point[0]].append(matched_values[point] - native_values[point])
                    paired.append(
                        {
                            "model_id": model_id,
                            "attack": attack["id"],
                            "contrast": "BER_matched_minus_BER_native",
                            "expected_images": count,
                            "expected_pairs": expected_count,
                            "observed_pairs": len(shared),
                            "coverage_complete": len(shared) == expected_count and not test_error,
                            "metric_applicable": any(
                                record["data"].get("ber") is not None
                                for record in native + positives
                            ),
                            **image_bootstrap(differences, config.get("bootstrap_repeats", 10000)),
                        }
                    )

    executable = {row["model_id"] for row in applicability if row["available_for_this_run"]}
    available_cells = [
        row for row in coverage if row["applicable"] and row["available_for_this_run"]
    ]
    available_coverage = bool(available_cells) and all(
        row["coverage_complete"] for row in available_cells
    )
    calibration_complete = bool(executable) and all(
        row["coverage_complete"] and row["at_least_1000_independent_negatives"]
        for row in calibration_rows
        if row["model_id"] in executable
    )
    all_planned_ok = bool(planned) and len(good) == len(planned)
    # Terminal blocks for unavailable catalogue members are expected, not a
    # green result for those members; other failed planned jobs still block.
    executable_plans_ok = all(
        checked.valid(spec, include_terminal=False) is not None
        for spec in planned
        if spec["model_id"] in executable or spec["model_id"] == "clean_control"
    )
    configured_protocol = (
        config.get("repeats", 3) == 3
        and not config.get("identity_only", False)
        and set(config.get("modes", ("native", "matched42"))) == {"native", "matched42"}
    )
    computational_complete = (
        available_coverage
        and calibration_complete
        and executable_plans_ok
        and not manifest_errors
        and base_error is None
        and configured_protocol
    )
    model_completion = []
    for item in applicability:
        model_id = item["model_id"]
        cells = [row for row in coverage if row["model_id"] == model_id and row["applicable"]]
        calibration_row = next(row for row in calibration_rows if row["model_id"] == model_id)
        measured = bool(cells) and all(row["coverage_complete"] for row in cells)
        calibrated = (
            calibration_row["coverage_complete"]
            and calibration_row["at_least_1000_independent_negatives"]
        )
        review = any(row["model_id"] == model_id for row in collisions)
        status = (
            "unavailable"
            if not item["available_for_this_run"]
            else (
                "review_required"
                if measured and calibrated and review
                else ("computationally_complete" if measured and calibrated else "incomplete")
            )
        )
        model_completion.append(
            {
                "model_id": model_id,
                "status": status,
                "applicable_cells": len(cells),
                "complete_cells": sum(row["coverage_complete"] for row in cells),
                "test_measurements_complete": measured,
                "independent_negative_audit_complete": calibrated,
                "review_required": review,
                "dependency_status": item["dependency_status"],
                "reason": item["reason"],
            }
        )
    summary = {
        "schema": "neural_full_coverage_v1",
        "image_limit": count,
        "conditions_per_image_mode": sum(
            len(attack["repeats"]) for attack in attacks_by_id.values()
        ),
        "requested_catalog_models": len(inventory),
        "available_models": len(executable),
        "unavailable_models": [
            row["model_id"] for row in applicability if not row["available_for_this_run"]
        ],
        "planned_unique_jobs": len(planned),
        "validated_ok_jobs": len(good),
        "all_materialized_jobs_ok": all_planned_ok,
        "available_expected_measurements_complete": available_coverage,
        "calibration_complete_for_available_models": calibration_complete,
        "full_computational_complete": computational_complete,
        "all_catalog_models_computational_complete": computational_complete
        and len(executable) == len(inventory),
        "review_required_models": sorted({row["model_id"] for row in collisions}),
        "scientific_interpretation_ready": False,
        "manifest_errors": manifest_errors,
        "base_report_error": base_error,
        "coverage_counts": dict(Counter(row["status"] for row in coverage)),
        "model_completion_counts": dict(Counter(row["status"] for row in model_completion)),
        "note": "Computation, complete catalogue coverage, calibration adequacy and scientific review are separate gates. No claim that five images establish robustness or that GPU types produce bitwise-identical outputs.",
    }
    tables = store.root / "tables"
    blocked = [
        {
            "job_id": record["job_id"],
            "model_id": record["spec"].get("model_id"),
            "operator": record["spec"].get("operator"),
            "image_id": record["spec"].get("image_id"),
            "role": record["spec"].get("role"),
            "attack": record["spec"].get("attack"),
            "reason": record["data"].get("reason"),
        }
        for record in checked.records.values()
        if record and record["status"] == "generation_blocked"
    ]
    summary["generation_blocked_jobs"] = len(blocked)
    amendments = []
    for path in sorted((store.root / "amendments").glob("*.json")):
        item = read_json(path)
        amendments.append(
            {
                key: item.get(key)
                for key in (
                    "amendment",
                    "previous_code",
                    "current_code",
                    "certificate_sha256",
                    "policy",
                    "runtime_migration",
                )
            }
        )
    summary["code_amendments"] = amendments
    migrations = [item["runtime_migration"] for item in amendments if item.get("runtime_migration")]
    summary["heterogeneous_runtime_continuation"] = bool(migrations)
    summary["runtime_migrations"] = migrations
    write_json(tables / "execution_amendments.json", amendments)
    write_csv(tables / "generation_blocked.csv", blocked)
    write_csv(tables / "coverage_matrix.csv", coverage)
    write_csv(tables / "model_applicability.csv", applicability)
    write_csv(tables / "model_completion.csv", model_completion)
    write_csv(tables / "calibration_coverage.csv", calibration_rows)
    write_csv(tables / "clean_control_review_required.csv", collisions)
    write_csv(tables / "paired_mode_differences_coverage.csv", paired)
    write_csv(tables / "runtime_extrapolation.csv", _timing_rows(good, config))
    write_json(tables / "dependency_status.json", dependency_status)
    write_json(
        tables / "statistical_report_status.json",
        {
            "valid_for_current_plans": base_error is None,
            "error": base_error,
            "note": "If invalid, older metric CSV files are not current-run statistical evidence; use coverage_matrix.csv to diagnose missing/invalid work.",
        },
    )
    completion_path = tables / "completion.json"
    base_completion = read_json(completion_path) if completion_path.exists() else {}
    base_completion["expected_matrix_complete"] = computational_complete
    base_completion["full_protocol_complete"] = bool(
        base_completion.get("full_protocol_complete")
        and summary["all_catalog_models_computational_complete"]
        and not collisions
        and not migrations
    )
    base_completion["report_kind"] = (
        "complete_configured_protocol"
        if base_completion["full_protocol_complete"]
        else "coverage_first_partial_or_review_pending"
    )
    summary["scientific_interpretation_ready"] = base_completion["full_protocol_complete"]
    write_json(completion_path, base_completion)
    write_json(tables / "full_completion.json", summary)
    lines = [
        "# Neural-channel benchmark — coverage report",
        "",
        *( ["**RUNTIME REVIEW REQUIRED:** this run combines saved measurements with an explicitly approved Python/glibc update. Numerical equivalence has NOT been established. Old records are unchanged; new records carry actual execution provenance. Review environment effects before pooling results for publication.", ""] if migrations else [] ),
        f"Test images: **{count}**; conditions per applicable image/mode: **{summary['conditions_per_image_mode']}**.",
        "",
        f"Catalog: {len(inventory)} requested; {len(executable)} currently executable; {len(inventory) - len(executable)} unavailable.",
        f"Validated materialized jobs: {len(good)}/{len(planned)}. This fraction alone is not full coverage.",
        "",
        f"Available-model full computation: **{computational_complete}**.",
        f"All-catalog full computation: **{summary['all_catalog_models_computational_complete']}**.",
        f"Independent calibration complete: **{calibration_complete}**.",
        f"Scientific review ready: **{summary['scientific_interpretation_ready']}**.",
        f"Safety-blocked channel jobs: **{len(blocked)}**; no detector score, no seed resampling. See tables/generation_blocked.csv. Missing descendants remain in the coverage denominator.",
        f"Audited code amendments: **{len(amendments)}**; see tables/execution_amendments.json. Original records/protocol identities are retained.",
        "",
        "## What to inspect",
        "",
        "- `tables/coverage_matrix.csv`: every model, mode and attack, including missing descendants and unsupported modes.",
        "- `tables/model_applicability.csv`: exact prerequisites and exclusions; blocked is not passed.",
        "- `tables/model_completion.csv`: model-by-model computation and negative-audit completion, with review-required flags.",
        "- `tables/calibration_coverage.csv`: independent negatives; small test controls do not replace calibration.",
        "- `tables/paired_mode_differences_coverage.csv`: paired image/repeat coverage beside BER differences.",
        "- `tables/runtime_extrapolation.csv`: same-environment observed-work scaling only, not a server/runtime guarantee.",
        "- `tables/blind_quality_review.csv`: human content/quality review remains a separate step.",
        "",
        "Origin models are Native-only and generate their own carriers from fixed COCO captions. Matched42 is not invented for them.",
        "A common 30-bit evaluated subset does not equalize full native payload capacity. Zero-bit scores are not BER.",
        "",
    ]
    if summary["unavailable_models"]:
        lines += ["Unavailable: " + ", ".join(summary["unavailable_models"]) + ".", ""]
    if collisions:
        lines += [
            "**Review required:** exact queried-message match on an unmarked identity control: "
            + ", ".join(summary["review_required_models"])
            + ". Investigate; do not cherry-pick a rerun.",
            "",
        ]
    if manifest_errors or base_error:
        lines += [f"Report/manifest validation issues: {manifest_errors}; {base_error or ''}", ""]
    atomic_bytes(store.root / "RUN_REPORT.md", "\n".join(lines).encode("utf-8"))
    return summary
