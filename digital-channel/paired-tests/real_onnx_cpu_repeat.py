"""Real unchanged ONNX checkpoint: repeated CPU extraction, not CUDA evidence."""

import gc
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import stegastamp_paired_hotfix as fix

weight, png = sys.argv[1:3]
with Image.open(png) as image:
    pixels = cv2.resize(np.asarray(image.convert("RGB")), (400, 400), interpolation=cv2.INTER_LINEAR)
inputs = {"image": pixels[None].astype(np.float32) / 255, "secret": np.zeros((1, 100), np.float32)}
rows = []
for count in (3, 1):
    session = ort.InferenceSession(weight, sess_options=fix.session_options(ort), providers=["CPUExecutionProvider"])
    for _ in range(count):
        result = session.run(["decoded"], inputs)[0]
        assert np.isfinite(result).all()
        bits = (result.reshape(-1) > .5).astype(int).tolist()
        assert len(bits) == 100
        rows.append(bits)
    del session
    gc.collect()
assert all(row == rows[0] for row in rows), "CPU extraction changed across repeats"
print(json.dumps({"status": "passed", "backend": "CPUExecutionProvider", "onnxruntime": ort.__version__,
                  "checkpoint_sha256": fix.cuda.checksum(weight), "png_sha256": fix.cuda.checksum(png),
                  "equal_extractions": 4, "sessions": 2, "bits": rows[0], "cuda_verified": False}))
