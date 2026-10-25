"""Build a repair notebook without repacking the frozen scientific release."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
NAME = "COCO_Embedding_Bank_PRODUCTION_HIDDEN_FNNS_FIX"
SOURCE_SHA = "6d7ddfcd05abdaf46f97ab7b6b174bff098b3043edcf069db64041554263f7b5"


def main():
    notebook = json.loads((DIST / "COCO_Embedding_Bank_PRODUCTION_MAGICKWAND_FIX.ipynb").read_text("utf-8"))
    helper = (ROOT / "hotfix_hidden_fnns.py").read_bytes()
    checksum = hashlib.sha256(helper).hexdigest()
    archive = DIST / "COCO_Embedding_Bank_PRODUCTION_source.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SOURCE_SHA
    notebook["cells"][0]["source"] = [
        "# Production bank: Meta HiDDeN + FNNS repair\n\n",
        "This is the previous production notebook with a narrow repair layer for **two methods only**. ",
        "The frozen source ZIP, all 29 model contracts, messages and PSNR rules are unchanged.\n\n",
        "Default: continue the existing **100-image** bank; run only Meta HiDDeN and FNNS. ",
        "The other 27 methods are not prepared or executed. The final report includes all 29.\n\n",
        "Upload `hotfix_hidden_fnns.py` when setup asks (once; cached on Drive afterwards). ",
        "The original production source ZIP is reused from Drive. Stop the old runner first. ",
        "Use the same output directory, A100 and previous environment.\n\n",
        "Meta: ignore only the generated attenuations .pyc in identity comparison; verify actual source/weights. ",
        "FNNS: restore exact backup bytes, or archive a damaged image/message pair and rebuild its four modes ",
        "with the original seed and payload. No retry is triggered by BER. Old evidence stays in `_meta/repairs/`. ",
        "Local tests do not constitute a completed GPU run.\n",
    ]
    settings = "".join(notebook["cells"][2]["source"])
    settings = settings.replace("N_IMAGES = 5", "N_IMAGES = 100")
    settings = settings.replace('MODELS = "all"', 'MODELS = ["meta_hidden_revisited", "fnns_omniseal"]')
    settings = settings.replace("# All 29 reviewed entries; a subset is diagnostic only.",
                                "# Repair only these two; do not recompute the other 27.")
    notebook["cells"][2]["source"] = settings.splitlines(True)
    notebook["cells"][3]["source"] = ["## 2. Restore the unchanged release and the scoped repair helper\n\n",
        "Keep the existing OUTPUT_ROOT. No new trial is created and no checksums are disabled. ",
        "Previously fixed RoSteALS/SepWater setup remains in this notebook unchanged.\n"]
    bootstrap = "".join(notebook["cells"][4]["source"])
    bootstrap += '''
# Small readable Python repair helper; deliberately outside the frozen release.
HOTFIX_SHA256 = "__HELPER_SHA__"
HOTFIX_NAME = "hotfix_hidden_fnns.py"
cached_fix = OUTPUT_ROOT / "_release" / ("hotfix_" + HOTFIX_SHA256[:16]) / HOTFIX_NAME
fix_source = Path(HOTFIX_NAME)
if not fix_source.is_file():
    fix_source = cached_fix
if not fix_source.is_file() and IN_COLAB:
    from google.colab import files
    print("Upload hotfix_hidden_fnns.py from the supplied repair bundle.")
    files.upload()
    fix_source = Path(HOTFIX_NAME)
if not fix_source.is_file():
    raise FileNotFoundError("Place hotfix_hidden_fnns.py beside this notebook and rerun setup.")
fix_bytes = fix_source.read_bytes()
if hashlib.sha256(fix_bytes).hexdigest() != HOTFIX_SHA256:
    raise RuntimeError("Repair helper checksum differs; use the file supplied with this notebook.")
cached_fix.parent.mkdir(parents=True, exist_ok=True)
if not cached_fix.exists():
    cached_fix.write_bytes(fix_bytes)
if hashlib.sha256(cached_fix.read_bytes()).hexdigest() != HOTFIX_SHA256:
    raise RuntimeError("Repair helper did not persist correctly on Drive.")
HOTFIX = SCRATCH_ROOT / ("hotfix_" + HOTFIX_SHA256[:16]) / HOTFIX_NAME
HOTFIX.parent.mkdir(parents=True, exist_ok=True)
HOTFIX.write_bytes(fix_bytes)
print("Scoped repair ready:", MODELS, "Existing results:", OUTPUT_ROOT)
'''.replace("__HELPER_SHA__", checksum)
    notebook["cells"][4]["source"] = bootstrap.splitlines(True)
    run = "".join(notebook["cells"][6]["source"])
    run = run.replace('[str(CONTROLLER), "-m", "wmbench_bank.runner", "run",',
                      '[str(CONTROLLER), str(HOTFIX), "run",')
    notebook["cells"][6]["source"] = run.splitlines(True)
    report = "".join(notebook["cells"][8]["source"])
    report = report.replace('str(CONFIG)', 'str(REPORT_CONFIG)')
    report = '''# Read-only aggregation of all 29 methods, including the untouched 27.
REPORT_CONFIG = SCRATCH_ROOT / "production_report_all_settings.json"
REPORT_CONFIG.write_text(json.dumps({**SETTINGS, "models": "all"}, indent=2), encoding="utf-8")
''' + report
    notebook["cells"][8]["source"] = report.splitlines(True)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"repair_cell_{index}")
            cell.update(execution_count=None, outputs=[])
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    output = DIST / f"{NAME}.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (DIST / "hotfix_hidden_fnns.py").write_bytes(helper)
    with zipfile.ZipFile(DIST / f"{NAME}_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (output, DIST / "hotfix_hidden_fnns.py", archive, ROOT / "HOTFIX_HIDDEN_FNNS.md"):
            bundle.write(path, path.name)
    (DIST / "hidden_fnns_build.json").write_text(json.dumps({
        "fix": "hidden-fnns-storage-v1", "source_archive_sha256": SOURCE_SHA,
        "helper_sha256": checksum, "notebook": output.name,
        "default_images": 100, "selected_methods": ["meta_hidden_revisited", "fnns_omniseal"],
        "other_method_source_changes": 0, "gpu_execution": "not_executed",
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
