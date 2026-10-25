"""Pre-attack quality only, in a dedicated fixed metrics environment."""

from pathlib import Path
import time
import numpy as np
from PIL import Image
from wmbench.io import read_json, write_json, sha256
from wmbench_full_payload.storage import freeze, inside, EvidenceError


def run(job):
    import torch
    import lpips
    import importlib.metadata
    from skimage.metrics import structural_similarity
    root, bank = Path(job["results"]), Path(job["bank"])
    cases = read_json(root / job["model"]["id"] / "cases.json")
    device = job["settings"]["device"]
    metric = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    weights = {name: __import__("hashlib").sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
               for name, t in metric.state_dict().items()}
    freeze(root / "quality-identity.json", {
        "metric": "LPIPS AlexNet v0.1; RGB [-1,1]; spatial=False",
        "ssim": "skimage RGB channel average, gaussian sigma=1.5, population covariance, data_range=255",
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "lpips", "scikit-image")},
        "weights": weights})
    for index, case in enumerate(cases, 1):
        path = root / case["method"] / "quality" / (case["id"]+".json")
        if path.exists():
            if read_json(path)["case_key"] != case["key"]:
                raise EvidenceError("Quality record belongs to another carrier")
            continue
        if time.time() >= job["deadline"]:
            raise TimeoutError("Quality budget reached")
        x = np.array(Image.open(inside(bank, case["source_crop"])).convert("RGB"))
        y = np.array(Image.open(inside(bank, case["output"])).convert("RGB"))
        if x.shape != y.shape or sha256(inside(bank, case["output"])) != case["carrier_sha256"]:
            raise EvidenceError("Quality input drift")
        mse = np.mean((x.astype(float)-y.astype(float))**2)
        psnr = float(10*np.log10(255**2/mse)) if mse else "Infinity"
        if isinstance(psnr, float) and abs(psnr-float(case["psnr"])) > .001:
            raise EvidenceError("Fresh PSNR differs from bank PSNR")
        if psnr == "Infinity" and float(case["psnr"]) != float("inf"):
            raise EvidenceError("Fresh infinite PSNR differs from bank PSNR")
        tensors = [torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().to(device)/127.5-1 for a in (x, y)]
        with torch.inference_mode():
            perceptual = float(metric(*tensors).item())
        if not np.isfinite(perceptual):
            raise EvidenceError("Non-finite LPIPS is not a valid quality measurement")
        write_json(path, {"case_key": case["key"], "psnr": psnr,
                         "ssim": float(structural_similarity(x, y, data_range=255, channel_axis=2,
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                         "lpips": perceptual})
        if index % 10 == 0:
            print(f"QUALITY {case['method']}: {index}/{len(cases)}", flush=True)
