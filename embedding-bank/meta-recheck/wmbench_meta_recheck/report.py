"""Small report: configurations remain separate; clean errors are not hidden."""

from pathlib import Path
import zipfile

from wmbench.io import atomic_bytes, read_json, write_csv, write_json
from wmbench_full_payload.storage import EvidenceError
from . import meta


def stats(values):
    return {"measured": len(values), "exact": sum(v["exact"] for v in values),
            "ber": sum(v["information_errors"] for v in values) / sum(v["information_bits"] for v in values) if values else None}


def label(s):
    return f"{s['exact']}/{s['measured']}; {s['ber'] * 100:.4f}%" if s["measured"] else "pending"


def report(root):
    if read_json(root / "recheck.json").get("schema") != "meta-sepwater-rosteals-recheck-v1":
        raise EvidenceError("Not this diagnostic's output folder")
    text = ["# Meta HiDDeN, SepWater and RoSteALS", "",
            "Clean native checks only: five frozen COCO images, five full messages per image. No attacks, ECC, PSNR matching or training.",
            "Completion means measured, not error-free. These configurations differ in payload, resolution and strength; this is not a model ranking.", "",
            "## Meta Revisited HiDDeN (not original 2018 HiDDeN)", "",
            "| Fixed configuration | Complete | Raw exact; BER | Clipped exact; BER | PNG exact; BER | Mean PNG PSNR |",
            "|---|---:|---|---|---|---:|"]
    rows, summaries = [], []
    mroot = root / "meta_hidden"
    plan_path = mroot / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        for profile in meta.PROFILES:
            measured = []
            for case in [c for c in plan["cases"] if c["profile"] == profile]:
                value = meta.row(mroot, case, plan["identity"], verify=False)
                if value and value["status"] == "measured":
                    measured.append(value)
                    for stage, result in value["stages"].items():
                        rows.append({"model": meta.MODEL["id"], "profile": profile, "case": case["stem"], "stage": stage,
                                     "ber": result["ber"], "exact": result["exact"], "errors": result["information_errors"],
                                     "bits": result["information_bits"], "seconds": value["seconds"]})
            stages = {s: stats([r["stages"][s] for r in measured]) for s in ("raw", "clipped", "png")}
            quality = [r["psnr_original_to_png"] for r in measured if isinstance(r["psnr_original_to_png"], (int, float))]
            psnr = sum(quality) / len(quality) if quality else None
            summaries.append({"profile": profile, "complete": len(measured), "expected": 25,
                              "stages": stages, "mean_png_psnr": psnr})
            quality_label = f"{psnr:.3f}" if psnr is not None else "pending"
            text.append(f"| {profile} | {len(measured)}/25 | {label(stages['raw'])} | {label(stages['clipped'])} | {label(stages['png'])} | {quality_label} |")
    text += ["", "- checkpoint_256: central 256 crop, 48 bits, no JND, scaling_w=0.3 (configuration stored in checkpoint).",
             "- demo_512: unchanged RGB512, 48 bits, JND, scaling_w=1.5 (official demo). Same payloads in both profiles.",
             "- Both are fixed in advance. We do not choose a winning strength/size/serializer based on these five images.",
             "- Direct raw decoding, clipping, 8-bit rounding in memory and PNG reload are stored separately; clean-image decoding is a control, not calibrated detection/FPR.", ""]
    ref = root / "references/_reports/summary.json"
    reference = read_json(ref) if ref.exists() else {}
    text += ["## RoSteALS and SepWater: frozen adapter versus upstream core", "",
             "| Method | Complete | Adapter PNG exact; BER | Upstream PNG exact; BER | Core agreement |",
             "|---|---:|---|---|---|"]
    for item in reference.get("rows", []):
        text.append(f"| {item['model']} | {item['complete']}/{item['expected']} | {label(item['adapter'])} | {label(item['upstream_export'])} | {item['core_agreement']}/{item['expected']}; review {item['core_review']} |")
    text += ["", "Same historical raw100 payloads, sources and selected checkpoints. RoSteALS uses WIBE's pinned fork/distributed config, not a newly chosen checkpoint.",
             "Both branches use fresh processes, compare inference weights, float output and cross-decoding. Adding RoSteALS' parent to the import path repairs our reference harness, not the model.",
             "Source-defined PNG export and the adapter's PNG rule are distinct. Core agreement does not promise zero BER. No production adapters are changed.", "", "## Execution", ""]
    state_path = mroot / "_meta/status.json"
    states = {"meta_hidden": read_json(state_path) if state_path.exists() else {"status": "pending"},
              **reference.get("execution_status", {})}
    for name, state in states.items():
        text.append(f"- {name}: {state.get('status')}; {state.get('phase', '')} {state.get('reason', '')}")
    text += ["", "Full vectors, logits, technical errors and per-stage timings are in the review ZIP. Each source PNG/NPY stays on Drive, outside the small ZIP.",
             "This JSON-only report does not reread large tensors; resume performs SHA256 artifact validation. A report is not a fresh integrity audit.", ""]
    write_json(root / "_reports/summary.json", {"meta": summaries, "references": reference, "status": states})
    write_csv(root / "_reports/meta_measurements.csv", rows)
    atomic_bytes(root / "_reports/REPORT.md", "\n".join(text).encode())
    print(f"Report: {root / '_reports/REPORT.md'}", flush=True)


def review_zip(root):
    destination = root / "Meta_SepWater_RoSteALS_report.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for base in (root / "_reports", root / "meta_hidden", root / "references"):
            if not base.exists():
                continue
            for path in base.rglob("*"):
                relative = path.relative_to(root)
                if path.is_file() and not {"_assets", "_release"}.intersection(relative.parts) and path.suffix.lower() in {".json", ".csv", ".md", ".log", ".txt", ".lock"}:
                    archive.write(path, relative.as_posix())
        archive.write(root / "recheck.json", "recheck.json")
    print("Review ZIP:", destination, flush=True)
    return destination
