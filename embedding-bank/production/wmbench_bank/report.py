"""Fast reports from small records; no model downloads or weight re-hashing."""

from collections import defaultdict
from pathlib import Path

from wmbench.io import atomic_bytes, digest, read_json, write_csv, write_json
from wmbench_full_payload.worker import record_path, task_id

from .contracts import MODES, capacity


def image_mean(rows, field):
    rows = list(rows)
    if any(row.get(field) == "Infinity" for row in rows):
        return "Infinity"
    groups = defaultdict(list)
    for row in rows:
        if isinstance(row.get(field), (float, int)):
            groups[row["image_id"]].append(row[field])
    values = [sum(v)/len(v) for v in groups.values()]
    return sum(values)/len(values) if values else None


def report(root, settings, refresh_methods=()):
    root = Path(root)
    identity = read_json(root / "trial.json")
    images = read_json(root / "_inputs/manifest.json")["images"][:settings["n_images"]]
    selected = list(identity["models"]) if settings.get("models", "all") == "all" else settings["models"]
    rows, summary = [], []
    identity_hash = digest(identity)
    for name in selected:
        model = identity["models"][name]
        spec = capacity(model)
        status_file = root / name / "_meta/status.json"
        state = read_json(status_file) if status_file.is_file() else {}
        cache_path = root / "_reports/method_indexes" / f"{name}.json"
        cache = read_json(cache_path) if cache_path.is_file() else {}
        if cache and cache.get("identity") != identity_hash:
            raise ValueError("Report index belongs to another scientific run")
        indexed = {(r["mode"], r["image_id"], r["replicate"]): r for r in cache.get("rows", [])}
        refreshed = []
        for mode in MODES:
            own = []
            for image in images:
                for replicate in range(1, 6):
                    stem = task_id(image, replicate)
                    lookup = (mode, image["image_id"], replicate)
                    if name not in refresh_methods and lookup in indexed:
                        own.append(indexed[lookup])
                        continue
                    paths = [record_path(root, model, stem, s) for s in (mode, mode+"_decode")]
                    enc, dec = [read_json(p) if name in refresh_methods and p.exists() else {} for p in paths]
                    row = {"model": name, "mode": mode, "image_id": image["image_id"],
                        "replicate": replicate, "width": model["native_geometry"]["width"],
                        "information_bits": spec["information_bits"], "bpp": spec["bits_per_pixel"],
                        "encode_status": enc.get("status", "pending"), "decode_status": dec.get("status", "pending"),
                        "psnr_db": enc.get("psnr"), "ber_fraction": dec.get("ber"),
                        "ber_percent": 100*dec["ber"] if dec.get("ber") is not None else None,
                        "exact": dec.get("exact"), "information_errors": dec.get("information_errors"),
                        "alpha": enc.get("matching", {}).get("alpha"),
                        "embed_or_match_seconds": enc.get("seconds"), "decode_seconds": dec.get("seconds"),
                        "image_path": enc.get("output"), "reason": enc.get("reason", dec.get("reason", ""))}
                    own.append(row)
                    if name in refresh_methods:
                        refreshed.append(row)
            measured = [r for r in own if r["decode_status"] in ("measured", "rejected")]
            unmatched = sum(r["encode_status"] == "unmatched" for r in own)
            saved = sum(r["encode_status"] == "saved" for r in own)
            terminal = len(measured)+unmatched == len(own)
            summary.append({"model": name, "mode": mode, "size": model["native_geometry"]["width"],
                "information_bits": spec["information_bits"], "bpp": spec["bits_per_pixel"],
                "expected": len(own), "saved": saved, "unmatched": unmatched,
                "measured_or_rejected": len(measured), "valid_ber": sum(r["ber_fraction"] is not None for r in own),
                "exact_messages": sum(r["exact"] is True for r in own),
                "mean_psnr_db": image_mean(own, "psnr_db"), "clean_ber_percent": image_mean(own, "ber_percent"),
                "exact_recovery_fraction": image_mean(measured, "exact"),
                "status": ("complete" if terminal else "incomplete" if state.get("status") == "complete"
                           else state.get("status", "pending")),
                "technical_errors": sum(r["encode_status"] == "error" or r["decode_status"] == "error" for r in own),
                "reason": state.get("reason", "")})
            rows.extend(own)
        if name in refresh_methods:
            # Keep entries outside a temporarily smaller N. This reporting
            # cache is never accepted as artifact evidence by the resume path.
            indexed.update({(r["mode"], r["image_id"], r["replicate"]): r for r in refreshed})
            write_json(cache_path, {"identity": identity_hash, "rows": list(indexed.values())})
    folder = root / "_reports"
    write_csv(folder / "clean_results.csv", rows)
    write_csv(folder / "clean_summary.csv", summary)
    result = {"n_images": len(images), "selected_methods": len(selected), "replicates": 5,
        "planned_carriers": len(rows), "saved_carriers": sum(r["saved"] for r in summary),
        "all_selected_technical_complete": bool(summary) and all(r["status"] == "complete" for r in summary),
        "all_targets_available": all(r["saved"] == r["expected"] for r in summary), "models": summary,
        "note": "Complete means measured, not error-free. No channel attacks executed."}
    write_json(folder / "summary.json", result)
    def fmt(v):
        return "—" if v is None else v if isinstance(v, str) else f"{v:.4f}"
    lines = ["# Embedding bank and clean decoding", "",
        f"{len(images)} source images × 5 messages × {len(selected)} methods × 4 modes. No attacks.", "",
        "BER is in percent; averages weight source images equally. Exact recovery is length dependent.",
        "Report indexes refresh when a method finishes/stops. During execution consult _meta/progress.json.",
        "Unmatched targets are not saved as successful images. Missing/error outputs have no BER.", "",
        "| Method | Mode | Size | Bits | Saved / planned | Clean measured | Exact | PSNR dB | BER % | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['mode']} | {row['size']} | {row['information_bits']} | "
            f"{row['saved']}/{row['expected']} | {row['measured_or_rejected']} | {row['exact_messages']} | "
            f"{fmt(row['mean_psnr_db'])} | {fmt(row['clean_ber_percent'])} | {row['status']} |")
    lines += ["", "## Before increasing N", "",
        "Review Native BER, all method errors, and unmatched coverage. Nonzero BER is not an installation failure.",
        "Keep the same source ZIP and OUTPUT_ROOT; only increase N_IMAGES. Old PNGs and valid decoder records are reused.",
        "Author paper numbers are contextual references, not computed by this stage. SSIM/LPIPS and paired attack deltas",
        "are not calculated here; this release supplies exact carriers, clean bit strings, PSNR and task coverage for them.",
        "Clean success alone does not establish channel robustness or reproducibility of a paper's experiment.", ""]
    lines += [f"- {r['model']}/{r['mode']}: {r['reason']}" for r in summary if r["reason"]]
    atomic_bytes(folder / "REPORT.md", ("\n".join(lines)+"\n").encode())
    return result
