"""Check the REAL pinned upstream namespace/package layout without GPU packages.

Pass a downloaded robustness ZIP and an output directory. This is a real source
layout regression test, not a claim of model construction or neural inference.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source_zip) as archive:
        for name in archive.namelist():
            target = (output / name).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Unsafe source ZIP")
        archive.extractall(output)
    outer = output / "robustness-8190fe329f5f072a06e3a2aea02bb5dda69aed9f/ImageNet-C"
    sys.path.insert(0, str(outer))
    broken = importlib.util.find_spec("imagenet_c")
    assert broken is not None and broken.origin is None
    sys.path.insert(0, str(outer / "imagenet_c"))
    fixed = importlib.util.find_spec("imagenet_c")
    expected = outer / "imagenet_c/imagenet_c/__init__.py"
    assert Path(fixed.origin).resolve() == expected.resolve()
    text = expected.read_text()
    assert "def corrupt(" in text and "corruption_dict" in text
    result = {"wrong_root_origin": broken.origin, "correct_root_origin": fixed.origin,
              "corrupt_api_defined": True, "scope": "Real pinned source discovery, not GPU/model execution"}
    (output / "layout_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
