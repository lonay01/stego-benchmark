"""Small JSON-only report; does not import torch or run any network."""

from wmbench.io import atomic_bytes, read_json, write_csv, write_json
from .core import POLICY, REFERENCE_METHODS, key_for, phase_names, row_path


def agreement(adapter, reference, cross, weights_equal):
    if not all(r.get("status") == "measured" for r in (adapter, reference, cross)):
        return "incomplete"
    comparisons = [reference[k] for k in ("decoder_on_adapter_raw", "decoder_on_adapter_png")]
    comparisons += [cross[k] for k in ("decoder_on_reference_raw", "decoder_on_reference_png")]
    okay = (weights_equal and reference["encoder_float"]["close"] and
            adapter["file_api_matches_direct"] and cross["file_api_matches_direct"] and
            all(r["same_bits"] and r["logits"]["close"] for r in comparisons))
    return "core_agreement" if okay else "review_difference"


def report(root):
    path = root / "plan.json"
    if not path.exists():
        raise FileNotFoundError("Run input preflight first; no report plan exists yet")
    plan = read_json(path)
    if plan.get("policy", {}).get("schema") != POLICY["schema"]:
        raise ValueError("Not an author-reference plan; refusing to replace another experiment's report")
    rows, summaries, statuses = [], [], {}
    text = ["# Upstream reference checks", "",
            "Five historical crops × five full payloads per method. SepMark/DistSeal additionally use the next five frozen images.",
            "Core agreement is not zero BER. Historical and holdout results remain separate. No production changes.", "",
            "| Method | Cohort | Complete | Adapter PNG exact; BER | Upstream export exact; BER | Core agreement |",
            "|---|---|---:|---|---|---|"]
    for name, cases in plan["cases"].items():
        status_path = root / name / "_meta/status.json"
        statuses[name] = read_json(status_path) if status_path.is_file() else {"status": "pending"}
        audit = root / name / "_meta/weight_agreement.json"
        weights_equal = audit.is_file() and read_json(audit).get("identical_inference_tensors") is True
        states = []
        for case in cases:
            phases = {}
            for phase in phase_names(name):
                p = row_path(root, case, phase)
                row = read_json(p) if p.exists() else {"status": "pending"}
                if p.exists() and row.get("signature") != key_for(plan["identity"], case, phase):
                    raise ValueError(f"Report task identity changed: {p}")
                phases[phase] = row
                for key in ("raw_score", "png_score", "author_score", "same_rgb_author_score",
                            "on_adapter_raw", "on_adapter_png", "on_reference_raw", "on_reference_png"):
                    if key in row:
                        metric = row[key]
                        rows.append({"model": name, "cohort": case["cohort"], "case": case["stem"],
                                     "phase": phase, "measurement": key, "ber": metric["ber"],
                                     "exact": metric["exact"], "errors": metric["information_errors"],
                                     "bits": metric["information_bits"], "seconds": row["seconds"]})
            a, r, c = (phases.get(k, {}) for k in ("adapter", "reference", "cross"))
            state = agreement(a, r, c, weights_equal) if name in REFERENCE_METHODS else "serializer_only"
            states.append({"case": case["stem"], "cohort": case["cohort"], "agreement": state,
                           "complete": all(v.get("status") == "measured" for v in phases.values()),
                           "historical_replay": a.get("history"),
                           "technical_errors": [v.get("reason") for v in phases.values() if v.get("status") == "error"]})
        for cohort in dict.fromkeys(c["cohort"] for c in cases):
            own = [r for r in rows if r["model"] == name and r["cohort"] == cohort]
            chosen = [s for s in states if s["cohort"] == cohort]
            def stats(phase, measurement):
                values = [v for v in own if v["phase"] == phase and v["measurement"] == measurement]
                return {"measured": len(values), "exact": sum(v["exact"] for v in values),
                        "ber": sum(v["errors"] for v in values) / sum(v["bits"] for v in values) if values else None}
            a = stats("adapter", "png_score")
            b = stats("reference", "png_score") if name in REFERENCE_METHODS else stats("adapter", "author_score")
            entry = {"model": name, "cohort": cohort, "expected": len(chosen),
                     "complete": sum(v["complete"] for v in chosen), "adapter": a, "upstream_export": b,
                     "core_agreement": sum(v["agreement"] == "core_agreement" for v in chosen),
                     "core_review": sum(v["agreement"] == "review_difference" for v in chosen)}
            summaries.append(entry)
            def label(value):
                return f"{value['exact']}/{value['measured']}; {100 * value['ber']:.4f}%" if value["measured"] else "pending"
            agree = f"{entry['core_agreement']}/{len(chosen)}; review {entry['core_review']}" if name in REFERENCE_METHODS else "not tested (serializer check)"
            text.append(f"| {name} | {cohort} | {entry['complete']}/{len(chosen)} | {label(a)} | {label(b)} | {agree} |")
        write_json(root / name / "_meta/case_assessment.json", states)
    text += ["", "## Execution status", ""]
    for name, state in statuses.items():
        text.append(f"- {name}: {state['status']}; {state.get('phase', '')} {state.get('reason', '')}".rstrip())
    text += ["", "## Interpretation", "",
        "- `core_agreement`: freshly loaded upstream inference tensors agree, raw embeddings are within fixed tolerances, and both decoders agree on both branches' pixels. It does not reproduce a paper's entire evaluation or promise error-free messages.",
        "- `review_difference`: inspect tensor differences, logits, preprocessing and file API checks. Do not call this model failure or silently change thresholds.",
        "- Upstream export may legitimately differ from adapter PNG; serialization is outside the core-agreement flag.",
        "- HiDDeN is the selected third-party PyTorch combined-noise checkpoint, not the original Torch7 implementation. RoSteALS uses WIBE's pinned fork/config/checkpoint.",
        "- Raw100 RoSteALS/SepWater bypass text/BCH helpers. The reference uses the authors' neural core, not text recovery.",
        "- SepMark: author's half-up export, not the floor variant that won on the old five images. DistSeal: repository `save_img` floor export; its post-hoc demo measures float tensors and does not itself specify a PNG round-trip.",
        "- The holdout is next-five validation relative to this diagnostic, not proof that those COCO images were unseen during model training.",
        "- Incomplete/error rows are technical failures, not BER=0. Saved nonzero BER is final and never retried to improve a score.",
        "- No result automatically promotes a production change. Review these tables before updating adapters.", ""]
    folder = root / "_reports"
    write_json(folder / "summary.json", {"policy": plan["policy"], "holdout_ids": plan["holdout_ids"],
                                        "rows": summaries, "execution_status": statuses})
    write_csv(folder / "measurements.csv", rows)
    atomic_bytes(folder / "REPORT.md", "\n".join(text).encode("utf-8"))
    print(f"Report: {folder / 'REPORT.md'}", flush=True)
