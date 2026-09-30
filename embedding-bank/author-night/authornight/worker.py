"""One process per stage; process exit releases all CUDA allocations."""

from pathlib import Path
import platform
import sys

from .storage import freeze, read


def main(job):
    root, scratch = Path(job["output"]), Path(job["scratch"])
    if job["phase"] == "prepare":
        from .environment import prepare
        prepare(job["method"], Path(job["project"]), root, scratch)
        return
    if job["phase"] == "inputs":
        from .environment import setup_inputs
        setup_inputs(job, root)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA GPU runtime. CPU fallback is disabled.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze(root / "_meta" / job["method"] / "execution.json", {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "cudnn": torch.backends.cudnn.version(),
        "tf32": False, "dtype": "float32", "cudnn_deterministic": True})
    if job["phase"] == "coco":
        if job["method"] == "distseal":
            from .distseal import run_coco
        else:
            from .latent import run_coco
        run_coco(job)
    elif job["phase"] == "corruptions":
        from .latent import run_corruptions
        run_corruptions(job)
    elif job["phase"] == "generation":
        from .distseal import run_generation
        run_generation(job)
    elif job["phase"] == "attacks":
        from .attacks import run
        run(job)
    elif job["phase"] == "inception":
        from .inception import run
        run(job)
    else:
        raise ValueError(job["phase"])


if __name__ == "__main__":
    main(read(sys.argv[1]))
