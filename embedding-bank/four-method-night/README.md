# Four-method overnight quality diagnostic

This is a separate recovery and expanded native-quality check. It does not alter
production adapters, previous images, messages, results or protocol documents.

## Run

1. Download both files from `dist`: `Four_Method_NIGHT_CHECK.ipynb` and
   `Four_Method_NIGHT_CHECK_source.zip`. Open the notebook in Colab.
2. Select **A100 + high system RAM**, matching the previous quality run. Do not
   run another GPU notebook concurrently. Native Windows is not supported here.
3. Keep `N_IMAGES = 20` and run all four cells. Authorize Google Drive and upload
   the matching source ZIP in cell 2. Once cell 3 is running, there are no further
   interactive prompts. The code is ordinary Python in the ZIP, not Base64.
4. Results go to `MyDrive/watermarking_four_method_night_v2`. The original native
   bank and `watermarking_four_method_quality_v1` are read-only inputs.
5. Send `Four_Method_NIGHT_report.zip` from the new folder. The report also lives
   at `_reports/REPORT.md`. Per-case logs and a `progress.json` are under each
   method's `_meta`; full tensors and PNGs remain on Drive.

After a session ends, select the same GPU and run the same cells again. Completed
sealed cases are verified and reused, even when BER is nonzero. An interrupted
case may need repeating; the other cases do not. Do not change `N_IMAGES`, source
release or data paths in an existing run. `MAX_HOURS` is only a per-invocation
budget and may be changed. It cannot extend a Colab session or ensure overnight
availability. Leave the computer powered and disable its ordinary sleep if
needed, but do not assume this prevents a server-side disconnect.

Linux GPU servers can use the same notebook with the three data directories
mounted under `/workspace/data`. Imports from this A100 run require a matching
recorded runtime. Mismatches are reported; there is no unsafe force-resume flag.

## Corrections

The previous SepWater `crop256` failed at `latent + residual`: the spatial sizes
were 64 and 56. This release keeps the selected checkpoint and its 224px API.
The comparison is now central **224** crop versus full RGB512 input resized to
**224** with Pillow bilinear interpolation. There is no padding, trainable change,
latent interpolation or silent claim that the old 256px experiment succeeded.

EditGuard's compile lock matched, but the installed package snapshot did not.
The only observed version difference was `platformdirs==4.11.12` versus `4.12.1`,
introduced through BasicSR's extra dependencies. Installation now constrains
these additional packages with the complete historical `installed.freeze.txt`.
The original lock, registry, weights and dependency identity gate are retained.
The previous Python patch version is requested when creating each model venv.
Both EditGuard checkpoint hashes are now tied to observed author downloads.
The clean checkpoint SHA-256 is
`b967b91b94a464ab73f82db63a57ee818e6fc5d99d7ae6aa094a1777cac977d5`;
this is a locally verified digest, not a checksum published by the authors.

## Matrix

| Method | Full information bits | Predeclared configurations |
|---|---:|---|
| SepWater | 100 | Historical crop224; full original resized to 224 |
| EditGuard copyright | 64 | Historical `degrade.pth`; author `clean.pth` |
| RoSteALS | 100 | Historical crop256; full original resized to 256 |
| DistSeal post-hoc DC-AE | 64 | Historical RGB512 configuration |

Default: **20 images × 5 messages × 7 configurations = 700 cases**. First five
images are historical replay; the next fifteen come from the same frozen
100-image manifest. No fresh COCO JPEGs are downloaded. Some additional images
were used for other methods in earlier diagnostics, so this is not claimed as
a universally unseen test set. Five messages do not create five independent
images.

The 100 successful, compatible v1 measurements are imported with their records,
hash-verified artifacts and execution provenance. Their images are not re-encoded.
The default queue therefore has 600 remaining new case measurements. Failed and
unattempted old rows are not imported. If the old quality folder is absent, all
700 are measured. The original native bank is always required. Additional tests
are not falsely marked as done on imported rows: their stage denominators remain
visible, and the new controls run on new cases.

The source ZIP pins the v1 release and preserves its inference files. RoSteALS
resize256 deliberately retains the exact prior direct-core/round-even path.
Its comparison with the historical API is a preprocessing-plus-export diagnostic,
not a pure causal crop experiment. No variant is selected based on favourable BER.

## Measurements and controls

- Full-message BER and exact recovery: raw float, clipped float, 8-bit round-even
  in memory, PNG reload, native saved PNG, and file-API/direct decoder agreement.
- Additional fixed serializers: floor and half-up, reported separately. The
  native saved PNG remains the main output; no winner selection.
- Lossless float32 `.npy` reload and bit agreement; PNG integer-pixel equality.
- Seed-reset re-embedding of message 1 for each newly measured image/profile,
  checking raw and PNG equality. This is a repeatability check, not a new sample.
- Same-latent autoencoder-only reconstruction for RoSteALS, SepWater and DistSeal.
  PSNR(source, AE), PSNR(source, marked), PSNR(AE, marked) are separately labelled.
- Clean-input and AE-only decoding controls. These are not calibrated FPR estimates.
- EditGuard decoder quantization is retained in the main path; decoding raw floats
  without that quantizer is an explicitly separate diagnostic control.
- Historical pixels/messages, source revisions, weight loading, exact dependency
  snapshots and runtime compatibility are checked. No answer bits enter decoder
  metadata.

The two cohorts are reported separately, with image-level aggregates in
`per_image.json`. A successful measurement can have nonzero BER. Completion,
consistency checks, reconstruction quality and payload recovery are distinct.
This diagnostic does not establish reproduction of the authors' paper datasets,
robustness to channel attacks, or architecture-level causal conclusions.

## Resource and failure handling

One model environment is active at a time. Completed scratch slots are removed;
Drive results are not. Start with at least 30 GiB free local SSD; the process guard
reserves 20 GiB and limits a slot to 100 GiB. Allow roughly 5–10 GiB free Drive
space for default-size diagnostic outputs and checkpoints; actual use varies.
The 12-hour budget is a ceiling, not a measured runtime prediction.

Every case is persisted immediately. A normal profile error does not suppress
the other profile or next method. Three identical consecutive technical errors
stop that profile to avoid wasting the night. Evidence/dependency mismatches
stop that method; they are not bypassed. Partial reports remain available.
Reporting reads small result records, not all float arrays or model weights.
Artifact checks occur on import/resume, not inside every report refresh.

The driver does not automatically release a paid Colab GPU when done. Check the
report and disconnect the runtime when you no longer need it.

## Development

`python build_notebook.py` creates deterministic artifacts and parses all Python
modules and notebook cells. `python -m pytest tests -q` runs CPU contracts and
orchestration smoke tests against the extracted release. `VALIDATION.md` records
what was actually executed. A synthetic test is not a pretrained GPU model run.
