"""Small command line interface, also called by the notebook without shell interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("inventory", "prepare", "run"):
        child = sub.add_parser(action)
        child.add_argument("--config", type=Path, required=True)
    child = sub.add_parser("import-data")
    child.add_argument("--images", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--captions", type=Path)
    child.add_argument("--legacy-manifest", type=Path)
    child = sub.add_parser("fixture")
    child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "import-data":
        from .data import import_dataset

        data = import_dataset(args.images, args.output, args.captions, args.legacy_manifest)
        print(f"Imported {len(data['images'])} verified PNG files: {data['content_hash']}")
    elif args.action == "fixture":
        from .data import synthetic_fixture

        synthetic_fixture(args.output)
    else:
        from .runner import Experiment

        experiment = Experiment(args.project, args.config)
        if args.action == "inventory":
            print(json.dumps(experiment.inventory(), ensure_ascii=False, indent=2))
        elif args.action == "prepare":
            experiment.prepare()
        else:
            experiment.run()


if __name__ == "__main__":
    main()
