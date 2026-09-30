"""Build a pinned, readable source archive and a four-cell Colab launcher."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "meta-recheck/dist"
BASE_SHA = "e45bade8d2431d88ae6346bcca19b9f954fa7979deb7c439d90e156ef39fec13"
ARCHIVE = "Four_Method_QUALITY_CHECK_source.zip"
NOTEBOOK = "Four_Method_QUALITY_CHECK.ipynb"


def build():
    base = (BASE / "Meta_SepWater_RoSteALS_CHECK_source.zip").read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA:
        raise ValueError("Frozen inference dependency changed")
    with zipfile.ZipFile(io.BytesIO(base)) as archive:
        contents = {n: archive.read(n) for n in archive.namelist() if not n.startswith("tests/")}
    for path in (ROOT / "wmbench_quality_review").glob("*.py"):
        contents["src/wmbench_quality_review/" + path.name] = path.read_bytes()
    for path in (ROOT / "tests").glob("*.py"):
        contents["tests/" + path.name] = path.read_bytes()
    for name in ("README.md", "VALIDATION.md"):
        contents[name] = (ROOT / name).read_bytes()
    contracts = json.loads(contents["source_contracts.json"])
    contracts.update(json.loads((ROOT / "editguard_sources.json").read_text("utf-8")))
    contents["source_contracts.json"] = json.dumps(contracts, indent=2).encode()
    contents["QUALITY_BASE.json"] = json.dumps({"sha256": BASE_SHA,
        "note": "Frozen inference src/registry/locks retained; new controller selects only four methods"}).encode()
    for name, content in contents.items():
        if name.endswith(".py"):
            ast.parse(content.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = stream.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    base_nb = json.loads((BASE / "Meta_SepWater_RoSteALS_CHECK.ipynb").read_text("utf-8"))
    code = ["".join(c["source"]) for c in base_nb["cells"] if c["cell_type"] == "code"]

    def transform(source):
        return (source.replace("Meta_SepWater_RoSteALS_CHECK", "Four_Method_QUALITY_CHECK")
                .replace("watermarking_meta_sepwater_rosteals_v1", "watermarking_four_method_quality_v1")
                .replace("watermarking_meta_recheck_scratch_v1", "watermarking_four_method_quality_scratch_v1")
                .replace("wmbench_meta_recheck.runner", "wmbench_quality_review.runner")
                .replace("meta_recheck_settings.json", "quality_settings.json")
                .replace(BASE_SHA, checksum))

    settings = transform(code[0]).replace(
        '["meta_hidden_revisited", "rosteals", "sepwater"]',
        '["rosteals", "sepwater", "distseal_posthoc_dcae", "editguard"]')
    settings = settings.replace("Only these three", "Only these four").replace(
        "50 Meta cases (two profiles) + 25 RoSteALS + 25 SepWater. Native only.",
        "200 cases: 5 images x 5 full messages x 8 fixed configurations. Native diagnostic only.")
    settings += '\nif os.name == "nt":\n    raise RuntimeError("Use Colab/Linux for this frozen GPU diagnostic; native Windows is not validated")\n'
    report = '''subprocess.run([str(CONTROLLER), "-m", "wmbench_quality_review.runner", "report",
                "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
print("Send this file:", OUTPUT_ROOT / "Four_Method_QUALITY_report.zip")
'''
    sections = [
        ("markdown", "# RoSteALS · SepWater · DistSeal · EditGuard\n\n"
         "Five historical images and five full messages. **200 cases, eight predefined configurations.** "
         "No changes to the old bank or production adapters. No attacks or matched PSNR.\n\n"
         "RoSteALS: crop256 / resize256. SepWater: crop224 / crop256 / resize256. "
         "DistSeal: selected RGB512 DC-AE. EditGuard: degrade / clean author checkpoints. "
         "Raw, clipped, integer and PNG decoding; same-latent AE controls and separate PSNR references.\n\n"
         "Use **A100 + high RAM**. Model environments run one at a time; results go to Drive. "
         "A six-hour budget is a limit, not an ETA. Read the supplied README before running. "
         "This release has CPU/static checks; its new GPU comparisons still need execution."),
        ("markdown", "## 1. Settings\n\nKeep N_IMAGES=5 and the four-method list. "
         "BANK_ROOT must point to the complete old native-resolution bank. OUTPUT_ROOT is separate."),
        ("code", settings),
        ("markdown", "## 2. Drive and readable source ZIP\n\nUpload the matching ZIP when prompted. "
         "It is cached on Drive; no Base64 code is embedded in this notebook."),
        ("code", transform(code[1])),
        ("markdown", "## 3. Run / resume\n\nRun cells 1–3 again after a disconnect. "
         "Completed records are checked and reused, including nonzero BER. Keep the same GPU/software."),
        ("code", transform(code[2])),
        ("markdown", "## 4. Report\n\nCan run after a partial run. "
         "Report generation does not load models; missing results are not counted as success."),
        ("code", report),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        cell = {"cell_type": kind, "id": f"quality-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            compile(source, f"cell_{index}", "exec")
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(payload)
    (dist / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"sha256": checksum, "base_sha256": BASE_SHA,
        "archive": ARCHIVE, "notebook": NOTEBOOK, "code_cells": 4,
        "python_modules": sum(n.endswith('.py') for n in contents)}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; source ZIP {len(payload):,} bytes; SHA256 {checksum}")


if __name__ == "__main__":
    build()
