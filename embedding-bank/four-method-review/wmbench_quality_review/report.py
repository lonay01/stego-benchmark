"""Small JSON/Markdown summaries; reporting never initializes a network."""

from pathlib import Path
import statistics
import zipfile

from wmbench.io import atomic_bytes, read_json, write_json
from .core import result


def mean(values):
    return statistics.mean(values) if values else None


def aggregate(rows):
    measured = [r for r in rows if r and r.get("status") == "measured"]
    stages = sorted({s for r in measured for s in r["measurements"]})
    summary = {"measured": len(measured), "expected": len(rows),
               "errors": sum(bool(r and r.get("status") == "error") for r in rows)}
    summary["stages"] = {}
    for stage in stages:
        data = [r["measurements"][stage] for r in measured if stage in r["measurements"]]
        summary["stages"][stage] = {"n": len(data), "exact": sum(v["exact"] for v in data),
                                     "ber_percent": 100 * mean([v["ber"] for v in data])}
    summary["psnr_db"] = {key: mean([r["psnr_db"][key] for r in measured if key in r["psnr_db"]])
                          for key in ("original_to_ae", "original_to_marked", "ae_to_marked")}
    summary["check_failures"] = {key: sum(r["checks"].get(key) is False for r in measured)
                                  for key in sorted({k for r in measured for k in r["checks"]})}
    summary["complete"] = len(measured) == len(rows)
    return summary


def metric(value):
    return "—" if value is None else f"{value:.3f}"


def recovery(group, stage):
    value = group["stages"].get(stage)
    return "—" if not value else f"{value['exact']}/{value['n']}; {value['ber_percent']:.4f}%"


def report(root):
    root = Path(root)
    if not (root / "plan.json").is_file():
        return
    plan = read_json(root / "plan.json")
    groups = {}
    lines = ["# Four-method quality review", "",
             "Five historical images × five full-length messages per configuration. "
             "Eight fixed configurations, 200 planned cases. No attacks, matching, training, "
             "payload shortening, or automatic promotion into production.", "",
             "PSNR columns are mean per-case dB; they cannot be added or subtracted as error budgets. "
             "AE uses the same sampled latent before embedding. Original-to-marked remains the main quality measure.", "",
             "| Method / configuration | Done | Raw exact; BER | Saved PNG exact; BER | Original → AE | Original → marked | AE → marked |",
             "|---|---:|---|---|---:|---:|---:|"]
    for name, cases in plan["cases"].items():
        for profile in plan["profiles"][name]:
            rows = [result(root, case, profile, plan["identity"]) for case in cases]
            group = aggregate(rows)
            groups[f"{name}/{profile}"] = group
            stage = "native_png_api" if profile in ("historical", "clean_checkpoint") else "native_png_direct"
            p = group["psnr_db"]
            lines.append(f"| {name} / {profile} | {group['measured']}/{group['expected']} | "
                         f"{recovery(group, 'raw')} | {recovery(group, stage)} | "
                         f"{metric(p['original_to_ae'])} | {metric(p['original_to_marked'])} | {metric(p['ae_to_marked'])} |")
    checks_ok = all(not any(g["check_failures"].values()) for g in groups.values())
    complete = all(g["complete"] for g in groups.values())
    lines += ["", f"Measurement coverage complete: **{complete}**. Observed consistency checks without failures: **{checks_ok}**.",
              "These flags do not mean zero BER, paper reproduction, or readiness for all channel attacks.", "",
              "## Reading the comparisons", "",
              "- `historical`: replay the frozen adapter, weights, crops and messages. Pixel/bit mismatches with history remain visible in summary.json.",
              "- `resize256`: bilinear resize of the full frozen RGB512 original; `crop256`: its central 256px pixels. These are separate diagnostic inputs, not replacements for the historical crop.",
              "- RoSteALS/SepWater 256px comparisons use the native latent core and direct decoder, not SepWater's fixed-224 file wrapper. Failure at 256 is reported, not silently resized.",
              "- EditGuard compares author `degrade.pth` and `clean.pth`. `raw` retains the author's decoder quantization; `raw_without_decoder_quantization` is a separate control. Only the copyright component is scored.",
              "- Clean-input and AE-only bit recovery are controls, not calibrated false-positive rates. No detector threshold is fitted here.",
              "- Lower AE PSNR identifies reconstruction distortion in this configuration; it does not by itself identify a faulty checkpoint, normalization, or model architecture.",
              "- Do not pick a checkpoint, crop or serializer on these five images and present the same images as independent validation. Freeze the choice and validate on other images first.", "",
              "## Technical status", ""]
    for name in plan["cases"]:
        status = root / name / "_meta/status.json"
        data = read_json(status) if status.exists() else {"status": "not_started"}
        lines.append(f"- {name}: {data.get('status')}; {data.get('reason', '')}")
    output = root / "_reports"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"complete": complete, "checks_ok": checks_ok, "groups": groups})
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    # Small records only; actual tensors/checkpoints remain in the result directory.
    archive = root / "Four_Method_QUALITY_report.zip"
    temporary = archive.with_suffix(".part")
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".log")
                  and "_release" not in p.parts and "attempts" not in p.parts and p.stat().st_size < 8 * 1024**2]
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(candidates):
            zf.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)
