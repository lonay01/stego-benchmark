"""A fast report from small JSON records. Never initializes a neural network."""

from collections import Counter

from wmbench.io import atomic_bytes, read_json, write_csv, write_json
from .core import stage_names


def report(root):
    plan = read_json(root / "plan.json")
    rows, models = [], []
    stages_by_case = {}
    for name, cases in plan["cases"].items():
        own = []
        for case in cases:
            folder = root / name / "cases" / case["stem"]
            stages = {}
            for stage in stage_names(name):
                path = folder / "stages" / f"{stage}.json"
                data = read_json(path) if path.is_file() else {"status": "pending"}
                stages[stage] = data
                row = {"model": name, "case": case["stem"], "stage": stage,
                       "control": stage.endswith("control"), "status": data["status"],
                       "ber": data.get("ber"), "exact": data.get("exact"),
                       "information_errors": data.get("information_errors"),
                       "psnr": data.get("psnr"), "reason": data.get("reason", "")}
                own.append(row)
            stages_by_case[(name, case["stem"])] = stages
        rows.extend(own)
        entries = {}
        for stage in stage_names(name):
            selected = [r for r in own if r["stage"] == stage]
            measured = [r for r in selected if r["status"] == "measured"]
            entries[stage] = {"measured": len(measured), "exact": sum(r["exact"] is True for r in measured),
                "mean_ber": sum(r["ber"] for r in measured) / len(measured) if measured else None,
                "errors": sum(r["status"] == "error" for r in selected),
                "unavailable": sum(r["status"] == "unavailable" for r in selected)}
        state_path = root / name / "_meta/status.json"
        state = read_json(state_path) if state_path.is_file() else {"status": "pending"}
        terminal = all(r["status"] in ("measured", "rejected", "unavailable") for r in own)
        audit_path = root / name / "_meta/audit.json"
        audit = read_json(audit_path) if audit_path.is_file() else {}
        flags = Counter()
        for case in cases:
            stages = stages_by_case[(name, case["stem"])]
            for stage, field in (("historical_png", "same_bits_as_history"),
                                 ("baseline_png", "same_pixels_as_history"),
                                 ("baseline_png", "same_bits_as_history"),
                                 ("round_even_png", "direct_memory_matches_png"),
                                 ("round_even_png", "file_api_matches_direct")):
                if stages[stage].get(field) is False:
                    flags[f"{stage}.{field}"] += 1
        models.append({"model": name, "expected_cases": len(cases), "stages": entries,
            "status": "diagnostics_complete" if terminal else state["status"],
            "reason": state.get("reason", ""), "replay_mismatches": dict(flags),
            "runtime_changes": audit.get("runtime_changes", {}),
            "stage_statuses": dict(Counter(r["status"] for r in own))})
    differences = []
    for (name, stem), stages in stages_by_case.items():
        for left, right in (("raw_float", "clipped_float"), ("clipped_float", "round_even_memory"),
                            ("round_even_memory", "round_even_png"), ("historical_png", "baseline_png")):
            a, b = stages[left], stages[right]
            if a.get("status") == b.get("status") == "measured":
                differences.append({"model": name, "case": stem, "from": left, "to": right,
                    "errors_before": a["information_errors"], "errors_after": b["information_errors"],
                    "changed_decoded_positions": [i for i, (x, y) in enumerate(zip(a["decoded_bits"], b["decoded_bits"])) if x != y],
                    "exact_lost": a["exact"] and not b["exact"]})
    folder = root / "_reports"
    write_csv(folder / "stages.csv", rows)
    write_csv(folder / "transitions.csv", differences)
    write_json(folder / "summary.json", {"models": models, "all_diagnostics_complete": all(
        m["status"] == "diagnostics_complete" for m in models), "transitions": differences})
    def cell(model, stage):
        s = model["stages"][stage]
        if not s["measured"]:
            return "not measured"
        return f"{s['exact']}/{s['measured']}; {100*s['mean_ber']:.4f}%"
    lines = ["# Five-method native diagnostics", "",
        "Same five crops and five messages as the historical canary. Cells show exact messages / measured messages; BER (%).",
        "No new weights, training, attacks, payload shortening or production adapter changes.", "",
        "| Method | Expected cases | Historical PNG replay | Baseline re-embed | Raw float | Clipped float | Round8 in memory | Round8 PNG/API | Status |",
        "|---|---:|---|---|---|---|---|---|---|"]
    for m in models:
        values = [cell(m, s) for s in ("historical_png", "baseline_png", "raw_float", "clipped_float", "round_even_memory", "round_even_png")]
        lines.append(f"| {m['model']} | {m['expected_cases']} | " + " | ".join(values) + f" | {m['status']} |")
    lines += ["", "## Fixed serialization alternatives (diagnostic, not selected winners)", "",
              "| Method | Round-to-even | Floor | Half-up | SepWater author float64 floor |",
              "|---|---|---|---|---|"]
    for m in models:
        author = cell(m, "author_export_png") if m["model"] == "sepwater" else "not applicable"
        lines.append(f"| {m['model']} | {cell(m, 'round_even_png')} | {cell(m, 'floor_png')} | {cell(m, 'half_up_png')} | {author} |")
    lines += ["", "## Interpretation", "",
        "- Raw -> clipped isolates range clipping; clipped -> Round8 memory isolates quantization.",
        "- Every PNG is verified pixel-identical to its uint8 input. PNG compression itself is lossless.",
        "- Round8 memory -> PNG/API may expose decoder preprocessing. Inspect direct_memory_matches_png and file_api_matches_direct separately.",
        "- historical_png and baseline_png records include per-case comparisons with the old decoded bits and original saved pixels.",
        "- Autoencoder-only controls reuse the same sampled latent. Original->AE, original->marked and AE->marked PSNRs are separate and non-additive.",
        "- Clean/AE decoding coincidences are controls, not FPR estimates. All controls remain outside the main table.",
        "- A completed diagnostic can contain bit errors. Do not choose a serializer or threshold separately for each message.",
        "- Repeated historical bit errors with matching artifacts support reproducibility, not proof that an adapter is correct.",
        "- A change of runtime is recorded; numerical replay differences must not automatically be called adapter bugs.", ""]
    for m in models:
        if m["reason"]:
            lines.append(f"- {m['model']}: {m['reason']}")
        if m["replay_mismatches"]:
            lines.append(f"- {m['model']} replay/API mismatches: {m['replay_mismatches']}")
        if m["runtime_changes"]:
            lines.append(f"- {m['model']} changed runtime fields: {', '.join(m['runtime_changes'])}; see audit.json.")
    atomic_bytes(folder / "REPORT.md", ("\n".join(lines) + "\n").encode())
    return models
