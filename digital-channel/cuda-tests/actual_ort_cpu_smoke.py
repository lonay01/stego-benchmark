"""Optional real-checkpoint CPU fallback rejection test (not a CUDA test)."""

import json
import sys

import onnxruntime as ort
import stegastamp_cuda_hotfix as fix

options = ort.SessionOptions()
options.intra_op_num_threads = 1
session = ort.InferenceSession(sys.argv[1], sess_options=options, providers=["CPUExecutionProvider"])
try:
    fix.require_cuda(session)
except RuntimeError as error:
    assert "CPU-only fallback is forbidden" in str(error)
    print(json.dumps({"status": "passed", "ort_version": ort.__version__,
                      "checkpoint_sha256": fix.checksum(sys.argv[1]),
                      "actual_providers": session.get_providers(), "correctly_rejected": str(error)}))
else:
    raise AssertionError("A real CPU-only session must not be accepted as CUDA")
