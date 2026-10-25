"""Patch only setup/lease-overlay cells in the backed-up live notebook."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat

from build_lease_io_fix import DIST, NAME


def main():
    source = DIST / "COCO_Digital_Channel_before_LEASE_IO_FIX_20261018.ipynb"
    old = json.loads(source.read_text("utf-8"))
    new = copy.deepcopy(old)
    checksum = hashlib.sha256((DIST / (NAME + "_source.zip")).read_bytes()).hexdigest()
    changed = []
    for cell in new["cells"]:
        code = "".join(cell["source"])
        if cell["id"] == "parallel-01":
            assert '    drive.mount("/content/drive")' in code
            code = code.replace('    drive.mount("/content/drive")',
                '    drive.mount("/content/drive")\n'
                '    if not os.path.ismount("/content/drive"):\n'
                '        raise RuntimeError("Drive is not mounted; authorize it before continuing.")')
        elif cell["id"] == "psutil-fix-setup":
            assert "performance_release_report_io_v4" in code
            code = code.replace("# Report I/O repair v4:", "# Lease read / storage repair v5:")
            code = code.replace("COCO_Digital_Channel_REPORT_IO_FIX_source.zip", NAME + "_source.zip")
            code = code.replace("2198afa282de2c4c352700c44593d245adfa4e4e2d79194e3f0c6c205505a7f5", checksum)
            code = code.replace("performance_release_report_io_v4", "performance_release_lease_io_v5")
            code = code.replace("Report I/O v4 installed: SSD staging, verified Drive copies, previous reports archived. Ready to resume.",
                "Lease/storage v5 installed: bounded control-file reads, ownership checks retained. Same bank/results; ready to resume.")
        else:
            continue
        ast.parse(code)
        cell["source"] = code.splitlines(True)
        cell["outputs"], cell["execution_count"] = [], None
        changed.append(cell["id"])
    assert changed == ["parallel-01", "psutil-fix-setup"]
    assert len(new["cells"]) == len(old["cells"])
    for before, after in zip(old["cells"], new["cells"]):
        if before["id"] not in changed:
            assert before == after
    for cell in new["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    nbformat.validate(nbformat.from_dict(new))
    target = DIST / "COCO_Digital_Channel_LEASE_IO_FIX_CURRENT.ipynb"
    target.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(target)
    print("Only modified cells:", changed)


if __name__ == "__main__":
    main()
