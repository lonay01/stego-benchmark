"""Reproducible readable ZIP and four-cell Colab notebook; old release untouched."""

import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "author-reference/dist"
BASE_SHA = "1d5c1a7b641e2b0a01d3841d86cb2050a369c680940013cb6a14aebd37ea7f3a"
ARCHIVE = "Meta_SepWater_RoSteALS_CHECK_source.zip"
NOTEBOOK = "Meta_SepWater_RoSteALS_CHECK.ipynb"


def build():
    base = (BASE / "Embedding_Bank_AUTHOR_REFERENCE_source.zip").read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA:
        raise ValueError("Frozen reference dependency changed; do not rebuild old adapters")
    with zipfile.ZipFile(io.BytesIO(base)) as archive:
        contents = {n: archive.read(n) for n in archive.namelist() if not n.startswith("tests/")}
    for path in (ROOT / "wmbench_meta_recheck").glob("*.py"):
        contents["src/wmbench_meta_recheck/" + path.name] = path.read_bytes()
    for path in (ROOT / "tests").glob("*.py"):
        contents["tests/" + path.name] = path.read_bytes()
    for name in ("README.md", "VALIDATION.md", "meta_contract.json"):
        contents[name] = (ROOT / name).read_bytes()
    contents["RECHECK_BASE.json"] = json.dumps({"sha256": BASE_SHA, "note": "All base src/registry/locks bytes preserved"}).encode()
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    base_nb = json.loads((BASE / "Embedding_Bank_AUTHOR_REFERENCE.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in base_nb["cells"] if c["cell_type"] == "code"]

    def transform(source):
        return (source.replace("Embedding_Bank_AUTHOR_REFERENCE", "Meta_SepWater_RoSteALS_CHECK")
                .replace("watermarking_author_reference_v1", "watermarking_meta_sepwater_rosteals_v1")
                .replace("watermarking_author_reference_scratch_v1", "watermarking_meta_recheck_scratch_v1")
                .replace("wmbench_author_reference.runner", "wmbench_meta_recheck.runner")
                .replace("author_reference_settings.json", "meta_recheck_settings.json"))

    settings = transform(code[0]).replace(
        '["hidden_benchmark", "rosteals", "sepwater", "sepmark", "distseal_posthoc_dcae"]',
        '["meta_hidden_revisited", "rosteals", "sepwater"]')
    settings = settings.replace("Only these five diagnostic methods.", "Only these three diagnostic methods.")
    settings = settings.replace("125 historical cases + 50 new serializer-validation cases. No attacks or PSNR matching.",
                                "50 Meta cases (two profiles) + 25 RoSteALS + 25 SepWater. Native only.")
    bootstrap = transform(code[1]).replace(BASE_SHA, sha)
    bootstrap = bootstrap.replace("This diagnostic uses a fixed five-image replay and next-five holdout",
                                  "This diagnostic uses exactly the same five source images")
    result = '''subprocess.run([str(CONTROLLER), "-m", "wmbench_meta_recheck.runner", "report",
                "--settings", str(CONFIG)], env=ENV, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT / "_reports/REPORT.md").read_text(encoding="utf-8")))
print("Send this file:", OUTPUT_ROOT / "Meta_SepWater_RoSteALS_report.zip")
'''
    sections = [
        ("markdown", "# Meta Revisited HiDDeN · SepWater · RoSteALS\n\n"
         "A separate clean encode/decode diagnostic. **Old banks and production code are not changed.**\n\n"
         "Meta uses its official checkpoint and unmodified inference classes, with two predefined configurations: "
         "checkpoint 256²/no JND/strength 0.3 and official demo 512²/JND/strength 1.5. Both carry 48 bits. "
         "This is Meta's Revisited HiDDeN, not original 2018 HiDDeN.\n\n"
         "RoSteALS (256², raw100) and SepWater (224², raw100) replay the old five crops/messages with the frozen adapter "
         "and independently loaded upstream cores. The RoSteALS reference import-path defect is repaired.\n\n"
         "Five images × five messages × four rows/configurations = **100 cases**. No training, attacks, ECC or matched PSNR. "
         "Nonzero BER is a measured result, not a reason to retry.\n\n"
         "Use **A100 + high RAM**, as in the earlier run. One isolated environment at a time, fresh worker processes, "
         "automatic per-case saving to Drive. Upload the matching readable source ZIP when prompted. "
         "The six-hour budget is a safety limit, not an ETA."),
        ("markdown", "## 1. Settings\n\nKeep N_IMAGES=5. BANK_ROOT is the previous native-resolution canary, including its images, "
         "metadata and environment locks—not a report ZIP. OUTPUT_ROOT must be a new folder."),
        ("code", settings),
        ("markdown", "## 2. Drive and source ZIP\n\nThe ZIP is cached on Drive after the first upload; it contains readable Python, "
         "no encoded payload or model weights. Heavy dependencies stay outside the notebook kernel."),
        ("code", bootstrap),
        ("markdown", "## 3. Run\n\nAfter a disconnect run cells 1–3 with the same paths. Finished cases are verified and reused. "
         "Do not change GPU/software halfway through a paired comparison. Logs are saved under each method's _meta/logs."),
        ("code", transform(code[2])),
        ("markdown", "## 4. Report\n\nRun even after a partial run. The report does not load models. Send the review ZIP for analysis; "
         "small JSON records and logs are included, large tensors and PNGs stay on Drive."),
        ("code", result),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        item = {"cell_type": kind, "id": f"meta-recheck-{index:02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            compile(source, f"cell_{index}", "exec")
            item.update(outputs=[], execution_count=None)
        cells.append(item)
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / ARCHIVE).write_bytes(payload)
    (dist / NOTEBOOK).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dist / "build.json").write_text(json.dumps({"sha256": sha, "base_sha256": BASE_SHA, "archive": ARCHIVE,
        "notebook": NOTEBOOK, "python_modules": sum(n.endswith('.py') for n in contents), "code_cells": 4,
        "files": {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}, source ZIP {len(payload):,} bytes; SHA256 {sha}")


if __name__ == "__main__":
    build()
