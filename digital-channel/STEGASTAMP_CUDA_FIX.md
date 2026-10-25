# StegaStamp: restore CUDA decoding

## Observed failure

The supplied `COCO_Digital_Channel_RESUME_FIX.ipynb` reused the completed earlier
methods, then stopped at StegaStamp Clean replay. Full prepare/decode logs for
session `1791203797899932084` contain:

```
libcublasLt.so.12: cannot open shared object file: No such file or directory
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*.
```

The corresponding bank metadata records CUDAExecutionProvider followed by
CPUExecutionProvider. Thus the worker's installed ONNX GPU wheel is not enough:
the session silently fell back to CPU because its isolated process could not
locate the shared libraries. CUDA/CPU outputs near a decision boundary need not
be bit-identical. A one-bit replay exception did not repair this runtime fault
and is superseded here, not expanded.

## Scope

- No edits to the frozen source archive, its identities, model files, messages,
  attacks, thresholds, geometry, quality settings or embedding bank.
- No changes to other methods. Their completed records remain reusable.
- The existing missing-cache cleanup fix remains in effect.
- StegaStamp alone receives the controller's existing CUDA12/cuDNN9 library paths
  through its child-process environment. Its pinned Python dependencies are not
  changed. The actual ONNX session must have CUDA first; automatic session-level
  CPU fallback is disabled. Normal CPU execution of individual unsupported graph
  operators is not forbidden (the original CUDA session also lists CPU).
- All selected StegaStamp Clean cases must exactly replay the original bank
  (`status`, `decoded_bits`, `ber`, `exact`) before any old record is withdrawn.
  Every decoding case also retains the original strict Clean check.
- Legacy StegaStamp digital records lacking actual-CUDA receipts are copied to
  a content-addressed backup on Drive, hash-checked and read back before their
  **exact active record files** are removed. No directory is deleted. This is
  backend-provenance correction, not discarding samples with poor BER.
- CUDA records, including partial records, resume normally. Old CPU records are
  never merged with CUDA rows. Missing attack ZIPs regenerate using the original
  deterministic checksums. Original Clean references stay untouched.

The observed run had 10 completed StegaStamp carriers. The repair discovers the
actual number of legacy records at runtime rather than hard-coding that count.
The backups, per-file receipts and old logs remain available for auditing.

## Run

Open `dist/COCO_Digital_Channel_STEGASTAMP_CUDA_FIX.ipynb` in Colab. Select the same
GPU as the bank (A100). Keep the existing bank/output directories and `N_IMAGES`.
Run all cells. Reuse the original source ZIP already saved on Drive; there is no
replacement archive. Do not delete results manually.

The previous repair helper cells remain byte-identical for provenance. Only the
run entry point changes; it does not call the old one-bit exception.

Look for `STEGASTAMP CUDA CLEAN ... exact replay`. If CUDA or strict replay still
fails, the notebook stops before moving existing records. Inspect
`stegastamp/cuda-preflight/` and its log. This repair does not claim that matching
provider names alone guarantees reproducibility across different GPUs/drivers.

Legacy backup: `stegastamp/_preserved_cpu_records/stegastamp-cuda-v1/`.
Execution receipts include actual providers/options and linked library hashes.

## Validation

Run `build_stegastamp_cuda_fix.py`, then `validate_stegastamp_cuda_fix.py` using the
existing diagnostic environment. The suite verifies notebook syntax, exact
scope of the notebook diff, original archive hashes, other-method behavior,
strict replay, backup readback, safe resume, missing libraries and CPU fallback.
CUDA inference itself cannot be tested on this local CPU-only machine. The
notebook performs that mandatory check on the user's GPU before accepting rows.

Reference: [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
ORT 1.20.x requires CUDA12/cuDNN9; its newer `preload_dlls` API is unavailable in
the pinned 1.20.2 wheel, so this fix supplies library paths before process start.
