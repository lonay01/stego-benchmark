# StegaStamp-only paired-control repair

## What the observed failure established

Session `1791216953124193256_cuda-preflight` successfully created an actual CUDA
session. Seven controls replayed the historical bank exactly; the eighth,
`000000000285_m02_matched_45`, differed at zero-based bit 24. Historical BER was
0.51; fresh BER was 0.50. This establishes a historical replay mismatch, **not**
its exact numerical cause. It is not an attack result. The earlier missing
CUDA-library problem had been resolved.

## Declared correction

This release changes **only StegaStamp's numerical execution and paired-control
policy**. It does not claim unchanged historical execution. The original model,
weights, input PNG hashes, payloads, image size, preprocessing, output threshold,
attacks and scientific scoring formulas remain unchanged. Other methods retain
their existing code and historical Clean checks.

The fixed CUDA profile requests TF32 off, `cudnn_conv_algo_search=HEURISTIC`, ORT
deterministic compute, sequential execution and one intra/inter-op CPU thread.
CUDA must actually load; CPU-only fallback is prohibited. These settings are
fixed before observing scores, not optimized for agreement with the bank.
The heuristic option avoids requesting the cuDNN frontend's `DEFAULT` fallback
plan mode, which the pinned upstream implementation warns can be very slow.
Provider settings are recorded, but empirical repeatability checks remain
mandatory: a numerical option alone is not an end-to-end determinism guarantee.

Every selected Clean carrier is decoded three times in the same session and
once in a newly created session. All four bitstrings must agree. The first
extraction is the paired control regardless of whether its BER is better or
worse than the historical one. There is no voting, threshold tuning or payload
feedback into the decoder. A poor but valid decoding remains a measured result.
Technical or repeatability failures remain failures and do not get scores.

Clean controls are committed only after all checks pass. A new decoding process
must replay those controls with the same runtime fingerprint before accepting
attacks. The fingerprint includes the numerical policy, actual providers and
options, ONNX hash/version, Python version, GPU model/driver, and CUDA-library
versions/hashes. Ephemeral filesystem paths are excluded. A change on resume
stops rather than mixing environments or replacing a committed control.

Historical bank Clean remains available in each case and the read-only bank.
`stegastamp/tables/stegastamp_bank_vs_paired_clean.csv` reports historical and
paired BER, their difference, exact-message rates at case level, and differing
bit positions. `STEGASTAMP_NUMERICAL_PROFILE.md` explains the numerical variant.
The original attack reports already pair each attack with its record's Clean;
they now use the new StegaStamp paired control. A difference from historical
Clean is never attributed to an attack. This distinction must be retained in a
paper; these rows are not bitwise reproduction measurements of the old bank run.

## Preservation and scope

Only selected legacy StegaStamp `records/*.json.gz` files are withdrawn. Before
withdrawal, each is copied byte-for-byte to a content-addressed backup, read
back, checksum-verified and accompanied by a receipt. Copies live in
`stegastamp/_preserved_previous_records/stegastamp-paired-cuda-v1/` on Drive.
No bank file, other-method record, or directory is deleted. Existing results
from this numerical profile, including partial ones, are reused unchanged.

Original source archive SHA-256:
`fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff`.
The notebook only adds an external readable repair module and switches its run
entry point. Older helper files are not edited or overwritten. They remain on
Drive, but their historical equality/one-bit gates are superseded for this method.

## Run and validation

Open `dist/COCO_Digital_Channel_STEGASTAMP_PAIRED_FIX.ipynb`. Keep the same bank,
output directory and image count. Select A100 and run all cells. The original
source ZIP is reused from Drive. Do not delete existing results manually.

`build_stegastamp_paired_fix.py` produces the notebook.
`validate_stegastamp_paired_fix.py` checks syntax, source immutability, other
methods' behavior, actual-checkpoint CPU inference, paired-control acceptance,
rejection of unstable or mixed-profile rows, backup readback and resume.
The test report separates CPU evidence from the mandatory, as-yet-unverified
Colab CUDA preflight. Setting deterministic options is not itself proof of
bitwise repeatability across all future GPUs/drivers.

Official references:

- [ONNX Runtime CUDA options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options): TF32 control and convolution search.
- [SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions): deterministic compute and execution/thread settings.
