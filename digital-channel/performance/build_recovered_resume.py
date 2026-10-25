"""Package the explicitly approved four-pair replacement and safe fast resume."""

import ast
import hashlib
import json
import zipfile

import build_recovery_resume as base

NAME = "COCO_Digital_Channel_RECOVERED_FAST_RESUME"


def build():
    base.NAME = NAME
    base.build()
    target = base.DIST / (NAME + "_source.zip")
    with zipfile.ZipFile(target) as z:
        files = {n: z.read(n) for n in z.namelist()}
    manifest = json.loads(files["performance-release.json"])
    name = "src/stegastamp_loss_reencode.py"
    files[name] = (base.DIGITAL / "stegastamp_loss_reencode.py").read_bytes()
    manifest["files"][name] = hashlib.sha256(files[name]).hexdigest()
    manifest["execution_repair"] = "four-pair-approved-replacement-and-resume-v3"
    files["performance-release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(target, "w") as z:
        for name, data in sorted(files.items()):
            if name.endswith(".py"):
                ast.parse(data, filename=name)
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                       compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    notebook = base.DIST / (NAME + ".ipynb")
    nb = json.loads(notebook.read_text("utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        source = source.replace("performance_release_recovery_v2", "performance_release_recovered_v3")
        if "EXPECTED_SHA" in source or "SOURCE_SHA256" in source or "PAYLOAD_SHA256" in source:
            # Only the archive pin, never any model/input/result identity.
            import re
            source = re.sub(r'((?:(?:SOURCE|PAYLOAD)_SHA256|EXPECTED_SHA)\s*=\s*["\x27])[0-9a-f]{64}',
                            lambda m: m[1] + checksum, source)
        if cell.get("id") == "parallel-00":
            source = """# Digital channel — recovered bank and fast resume

Same 29 methods, full messages and 79 conditions. Four lost StegaStamp
image/message pairs were explicitly authorized for replacement on 2026-10-16.
Their Native and three Matched carriers and Clean controls are new measurements,
not byte-exact restoration. The previous records and files are archived separately.
No other method or experimental setting is changed.

Use the existing bank/output paths. Upload the matching source ZIP once if it is
not already in `_release` on Drive, then run all cells. The recovery cell verifies
the committed replacement and does not embed it again. A fresh VM may rebuild
environments; previously decoded evidence remains on Drive.
"""
        if cell.get("id") == "byte-recovery-note":
            source = """## Verify the approved four-pair replacement

This is specific to the October 16 recovery, not a generic permission to change
bank inputs. The sealed replacement, backup and original job stay under
`watermarking_stegastamp_recovery_20261016/approved_replacement`.
On a repeat run, only the replacement receipt and its files are checked.
"""
        if cell.get("id") == "byte-recovery-run":
            source = '''recovery_root = BANK_ROOT.parent / "watermarking_stegastamp_recovery_20261016"
recovery_job = recovery_root / "1792169597490590440_job.json"
recovery_work = recovery_root / "approved_replacement"
if not (recovery_work / "replacement_receipt.json").is_file():
    raise RuntimeError("Approved replacement receipt is missing. Stop and inspect recovery; do not regenerate blindly.")
subprocess.run([str(CONTROLLER), "-u", "-m", "stegastamp_loss_reencode", "run",
    "--job", str(recovery_job), "--work", str(recovery_work),
    "--results", str(OUTPUT_ROOT)], env=ENV, check=True)
print("Four replacement pairs verified; historical evidence retained.", flush=True)
'''
        if cell["cell_type"] == "code":
            ast.parse(source)
        cell["source"] = source.splitlines(True)
    notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (base.DIST / (NAME + "_SHA256.txt")).write_text(checksum + "  " + target.name + "\n")
    print("FINAL SHA256", checksum)


if __name__ == "__main__":
    build()
