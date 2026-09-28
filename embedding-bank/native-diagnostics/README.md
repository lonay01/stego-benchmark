# Five-method native-resolution diagnostics

A targeted recheck of **HiDDeN benchmark, RoSteALS, SepWater, SepMark and DistSeal
post-hoc DC-AE**, after the native-resolution canary. This does not change the
production bank, checkpoints, payloads, thresholds or protocol.

## Run

1. Upload `dist/Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY.ipynb` to Google Colab.
2. Select **A100 + high system RAM**, matching the completed canary. A different
   GPU may run, but is not a tested numerical-replay environment.
3. Run all four code cells. Mount Drive and upload the matching
   `Embedding_Bank_NATIVE_DIAGNOSTICS_RECOVERY_source.zip` when requested.
4. The default source is `MyDrive/watermarking_native_resolution_canary_v1`.
   It must contain actual historical images, metadata, locks and `trial.json`;
   the small review ZIP alone is not sufficient. No COCO re-download is used.
5. Results go to **`MyDrive/watermarking_native_diagnostics_v1`**. Read
   `_reports/REPORT.md`; share `native_diagnostics_report.zip` for review.

The source ZIP is cached on Drive after the first upload. After a disconnect,
run cells 1–3 again with the same paths, model selection and notebook version.
Completed stages, including bit errors, are reused. Only technical failures
retry, with their old records retained. Never run two controllers concurrently.

## Recovery from the September 22 diagnostic run

The original run measured SepWater, SepMark and DistSeal but failed before the
HiDDeN/RoSteALS cases at `inspect.getsourcefile()`. WIBE temporarily imports their
classes and removes the nested modules from `sys.modules` afterwards. Python's
inspection routine then reports these live Python classes as built-in classes.

The recovery checks the source paths retained by class methods instead. It does
not reimport the networks or change the module cache. Source lookup failures are
explicitly recorded in `audit.json`; evaluation-mode and checkpoint-coverage
checks remain mandatory and are not swallowed by this fix.

**Keep all five model IDs and the original output path.** The controller verifies
the completed three methods and skips their environments and inference. Only
incomplete cases are run. Do not delete Drive folders or change the serializer.
The new notebook needs its **new matching ZIP**, not the old source archive.

Recovery recognizes only the original source identity from release ZIP
`ba01ad3bd6c13131c8491031a5986cd35af0e987f9a37824604cd1b6506bbfc4`.
All source files outside the reviewed diagnostic patch must have the original
combined hash. Trial, model selection and policy must also match. The original
`diagnostic_identity.json`, `plan.json`, stage signatures and completed records
are retained. A separate `_meta/code_revisions/*.json` records the compatibility
decision, and newly executed stages record the current source digest. Previous
model error statuses are retained under `<method>/_meta/status_history/`; their
session-specific logs remain in place. An unrelated source or protocol change
still stops resume. There is no force-resume switch.

This corrects a diagnostic infrastructure failure, not bit errors in a research
model. The real HiDDeN/RoSteALS results still need the resumed Colab run.

Five historical images × five original messages × five methods = **125 cases**.
All cases are retained, including successful controls; this is not a search for
easier pictures or messages. Images are the original central crops: HiDDeN128,
RoSteALS256, SepWater224, SepMark256, DistSeal512. Native strength only.

## Checks

| Check | What it distinguishes |
|---|---|
| Historical PNG replay | Same saved image, blind decoder and full payload; compares decoded bits with the original run |
| Baseline re-embedding | Same crop, payload and original seed; compares pixels and decoded bits with the historical PNG |
| Raw floating-point image | Actual encoder/autoencoder output captured from the baseline forward pass, before output clipping |
| Clipped floating-point image | Range clipping independently of integer quantization |
| Round-to-even RGB8 in memory | Quantization independently of PNG IO |
| Round-to-even PNG | Exact pixel equality after PNG read; direct tensor decoder compared with the file API |
| Floor and half-up PNG | Two fixed serializer ablations, not per-message optimization |
| Clean image control | Same decoder on an unmarked crop; not an FPR estimate |
| Autoencoder-only control | RoSteALS/SepWater/DistSeal reconstruction using the **same sampled original latent** as the marked pass |
| SepWater author export | Signed tensor and float64 NumPy residual export following the pinned author `embed.py`, compared with the existing adapter |
| Model audit | Evaluation mode, normalization, source hashes, parameter dtypes and checkpoint coverage |

HiDDeN uses the existing third-party PyTorch benchmark checkpoint, **not** a
newly substituted original Torch7 model. Its decoder threshold remains0.5.
RoSteALS/SepWater/SepMark/DistSeal retain threshold0. RoSteALS inference keys
(`ae`, `control`, `decoder`) are checked against the loaded checkpoint because
the upstream loader uses `strict=False`. SepWater's existing strict inference
key validation is retained.

HiDDeN and RoSteALS also expose the WIBE wrapper return; it may already be
clipped. SepMark/DistSeal expose the tensor entering the common PNG serializer.
Hooks copy tensors only: no decode, sampling or reference-based decision runs
inside the original forward pass. Full useful payloads and original seed
derivation are checked against the historical records before any GPU work.

## Interpretation

The main table reports whole-message recovery and BER at each boundary, not a
single winner. `raw -> clipped` identifies clipping effects; `clipped -> RGB8`
identifies quantization effects. PNG is lossless with respect to supplied
integer pixels. If PNG/API differs from in-memory decoding, the report also
compares the same reloaded image through the direct decoder to isolate API
preprocessing from file IO.

All serializers are declared in advance and applied to every case. Do not choose
whichever works best per message, change thresholds using these five images,
shorten payloads, or rerun bit errors until successful. A serializer that helps
this diagnostic needs a separate validation set before production adoption.

Autoencoder controls report original→AE, original→marked and AE→marked PSNR.
These values are not additive; the reference remains the original crop.
Raw logits, wrong positions, clipping fractions, quantization deltas and
per-case comparisons remain in JSON records. Controls are excluded from the
main success table. Five messages per image are not five independent images.

`diagnostics_complete` means all requested observations exist, **not** zero BER
or a proven correct adapter. Reproducing errors supports reproducibility, not a
causal claim about the architecture. Raw output can be non-decodable even with
correct integration; compare an independent author reference before blaming a
checkpoint. This notebook neither trains nor repairs a model automatically.

## Safety, dependencies and cost

The old canary is read-only. Its hashes, model definitions, inference weights
and dependency identities are checked. Different weights or installed packages
stop a like-for-like replay. Runtime details are recorded separately; a GPU or
library-backend change may alter numerical results.

Each method runs in its own pinned Python environment and subprocess. Only one
is resident at a time. Verified completed scratch slots are removed; failed
slots remain for inspection. No result directory is deleted. The runner starts
another method only with >=30 GiB free scratch and monitors RAM/disk use. The
six-hour default is a safety ceiling, not a duration estimate.

Preparation can dominate elapsed time. Recovery skips preparation and inference
for verified completed methods, but still validates their saved evidence.
Network transfers, dependency setup and Drive speed prevent a reliable ETA.
This notebook does not purchase a runtime or keep Colab alive artificially.

Small arrays are retained on Drive so interrupted decoding need not redo
embedding. The review archive includes records, logs and environment locks,
but excludes `.npy`, PNGs and model weights. It is not a backup of the image bank.

## Development

`python build_notebook.py` builds a readable-source ZIP and a four-code-cell
notebook. It reuses the checksum-verified native-canary release as its base;
does not modify that release or any production adapter. `python -m pytest tests`
loads the exact shipped ZIP. See `VALIDATION.md` for actual test coverage and
the boundary between CPU validation and an unexecuted real-model GPU run.
