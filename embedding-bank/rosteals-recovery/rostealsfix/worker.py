"""Only RoSteALS is accepted. Long work is gated by real runtime preflight."""

from pathlib import Path
import sys

from authornight.storage import EvidenceError, read


def main(job):
    if job["method"] != "rosteals":
        raise ValueError("This release runs RoSteALS only")
    phase = job["phase"]
    if phase == "import":
        from .migration import import_parent
        import_parent(job["settings"], job["code_hash"])
        return
    if phase == "prepare":
        from authornight.environment import REQUIREMENTS, prepare
        if "slackclient==2.9.3" not in REQUIREMENTS["rosteals"]:
            REQUIREMENTS["rosteals"].append("slackclient==2.9.3")
        prepare("rosteals", Path(job["project"]), Path(job["output"]), Path(job["scratch"]))
        return
    if phase == "imports":
        from .checks import import_and_attack_check
        import_and_attack_check(job)
        return
    from authorrecovery.worker import configure_gpu, preflight
    configure_gpu(job)
    root = Path(job["output"])
    if not read(root / "_meta/fix/import_check.json").get("passed"):
        raise EvidenceError("Complete the import/attack API check first")
    if phase == "preflight":
        preflight(job)
        return
    if not read(root / "_meta/recovery/rosteals_preflight.json").get("passed"):
        raise EvidenceError("Complete the full author preflight first")
    if phase == "coco":
        from authornight.latent import run_coco
        run_coco(job)
    elif phase == "corruptions":
        from authorrecovery.corruptions import run
        run(job)
    else:
        raise ValueError(f"Unknown phase: {phase}")


if __name__ == "__main__":
    main(read(sys.argv[1]))
