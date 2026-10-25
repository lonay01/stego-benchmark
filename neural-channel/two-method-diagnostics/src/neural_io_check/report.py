"""Auditable row tables and fixed, decoder-independent visual examples."""

import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import ATTACK_IDS, CONTROLS, METHODS, read, report_state, write


def table(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stream.getvalue(), encoding="utf-8")


def image_mean(rows, field):
    """Messages are repeated measurements, not additional independent images."""
    groups = {}
    for row in rows:
        if row.get(field) is not None:
            groups.setdefault(row["image_id"], []).append(row[field])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def gallery(root, method, cases):
    chosen = [c for c in cases if c["replicate"] == 1]  # Fixed before seeing BER.
    conditions = ("clean", "vae_1", "vae_2", "sdxl_0.15_r1", "ip2p_lighting_r1")
    for case in chosen:
        canvas = Image.new("RGB", (256 * len(conditions), 286), "white")
        draw = ImageDraw.Draw(canvas)
        for j, condition in enumerate(conditions):
            path = root / method / "images" / condition / (case["id"] + ".png")
            draw.text((j * 256 + 5, 5), condition, fill="black")
            if path.is_file():
                with Image.open(path) as im:
                    canvas.paste(im.resize((256, 256)), (j * 256, 28))
            else:
                draw.text((j * 256 + 5, 100), "Missing / blocked", fill="black")
        target = root / "gallery" / f"{method}_{case['image_id']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, quality=90)


def report(root, settings):
    root = Path(root)
    expected = settings["n_images"] * 5 * (len(CONTROLS) + len(ATTACK_IDS))
    lines = ["# InvisMark / ChunkySeal — neural IO diagnostic", "",
             "Native, five full messages per image. BER is a percentage in these tables.",
             "IO agreement is not a claim of high robustness or paper reproduction.", ""]
    summary, flat, states = [], [], {}
    for method in METHODS:
        rows = [read(p) for p in sorted((root / method / "checks").glob("*/*.json"))]
        state = report_state(rows, expected)
        states[method] = dict(state=state, rows=len(rows), expected=expected,
            blocked=sum(r["status"] == "blocked_generation" for r in rows))
        status = read(root / method / "status.json") if (root / method / "status.json").exists() else {}
        lines += [f"## {method}", "", f"**{state}** — {len(rows)}/{expected} committed checks.", ""]
        if status.get("reason"):
            lines += ["Last stage error: `" + status["reason"].replace("`", "'") + "`", ""]
        local = []
        for row in rows:
            score = row.get("official", {})
            direct = row.get("direct", {})
            floating = row.get("float_control", {}).get("score", {})
            flat_row = dict(method=method, image_id=row["image_id"], replicate=row["replicate"],
                condition=row["condition"], status=row["status"],
                information_bits=score.get("information_bits"),
                ber_percent=None if score.get("ber") is None else score["ber"] * 100,
                direct_ber_percent=None if direct.get("ber") is None else direct["ber"] * 100,
                float_ber_percent=None if floating.get("ber") is None else floating["ber"] * 100,
                exact_percent=None if score.get("exact") is None else int(score["exact"]) * 100,
                io_agrees=row.get("io_agrees"), repeat_equal=row.get("repeat_equal"),
                history_status=row.get("history", {}).get("status"),
                historical_bits_equal=row.get("history", {}).get("bits_equal"))
            local.append(flat_row)
            flat.append(flat_row)
        lines += ["| Condition | Measured | Blocked | BER % | Direct BER % | Float BER % | IO disagreements |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for condition in list(CONTROLS) + list(ATTACK_IDS):
            group = [r for r in local if r["condition"] == condition]
            row = dict(method=method, condition=condition, measured=sum(r["status"] == "measured" for r in group),
                blocked=sum(r["status"] == "blocked_generation" for r in group),
                ber_percent=image_mean(group, "ber_percent"), direct_ber_percent=image_mean(group, "direct_ber_percent"),
                float_ber_percent=image_mean(group, "float_ber_percent"),
                exact_percent=image_mean(group, "exact_percent"),
                io_disagreements=sum(r["io_agrees"] is False for r in group))
            summary.append(row)
            fmt = lambda value: "—" if value is None else f"{value:.4f}"
            lines.append(f"| {condition} | {row['measured']} | {row['blocked']} | {fmt(row['ber_percent'])} | "
                         f"{fmt(row['direct_ber_percent'])} | {fmt(row['float_ber_percent'])} | {row['io_disagreements']} |")
        lines += [""]
        cases_path = root / method / "native-cases.json"
        if cases_path.exists():
            gallery(root, method, read(cases_path))
    lines += ["## Interpretation", "",
        "- INCOMPLETE / TECHNICAL_ERROR / REVIEW_IO_OR_REPLAY: inspect logs and row evidence before a full run.",
        "- IO_AGREEMENT: all requested checks completed and decoder routes/repeats agreed. High attack BER remains valid evidence, not a trigger to tune the decoder.",
        "- IO_AGREEMENT_REPLAY_BYTES_DIFFER: routes agreed, but this is not a byte-exact replay of every historical generation. Inspect hashes before attributing old/new score differences.",
        "- Float vs PNG isolates serialization loss. Neither is selected as the winning official result.",
        "- Unwatermarked decoding near chance is an auxiliary negative control, not a calibrated FPR estimate.",
        "- Inspect gallery/*.jpg and original images/**/*.png: previews do not substitute for exact PNGs.",
        "- No edits to the bank, historical neural scores, checkpoint strengths or thresholds were requested by this notebook."]
    root.mkdir(parents=True, exist_ok=True)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table(root / "summary.csv", summary)
    table(root / "per_case.csv", flat)
    write(root / "status.json", states)
