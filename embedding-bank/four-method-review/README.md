# Four-method quality review

This diagnostic checks **RoSteALS, SepWater, DistSeal post-hoc DC-AE and EditGuard**.
It does not change the production adapters, old images, messages, thresholds,
payload lengths or matching rules. The three blocked entries are no longer in
the active benchmark. Old notebooks and audit records keep their original scope.

## Run in Colab

1. Download `dist/Four_Method_QUALITY_CHECK.ipynb` and its matching
   `dist/Four_Method_QUALITY_CHECK_source.zip`.
2. Upload the notebook into Colab. Choose **A100 and high system RAM**, as in the
   previous diagnostic. H100 is also possible for a separate new run; keep one
   GPU/software configuration throughout a paired run. This release is not
   validated on a low-RAM T4 or native Windows.
3. Run cells 1–4. Authorize Drive and upload the source ZIP when prompted.
   No model weights enter the notebook kernel. Four isolated environments run
   sequentially; successful scratch environments are removed after use.
4. After a disconnect, run the same cells with the same paths. Completed cases
   are hash-checked and reused, including cases with nonzero BER. Corrupt evidence
   stops the run rather than being overwritten. Technical errors can be retried.

Default input (read-only):
`/content/drive/MyDrive/watermarking_native_resolution_canary_v1`.
It must contain the complete old bank, originals, metadata, environment locks
and decoded messages, not just its report ZIP.

New output: `/content/drive/MyDrive/watermarking_four_method_quality_v1`.
Send **`Four_Method_QUALITY_report.zip`** from that folder after the run.
Intermediate logs and results live under `<method>/_meta` and `<method>/cases`.
Cell 4 can report a partial run without loading any model.

Keep at least **30 GiB free local scratch to begin each method**; the controller
stops below 20 GiB or above 100 GiB per model slot. Keep spare Drive capacity for
images, float arrays and retained author checkpoints. A six-hour session budget
is a safety limit, not a measured ETA or a Colab uptime guarantee. Watch actual
per-profile timing before estimating the full benchmark.

## Fixed matrix

All rows use the same first five historical COCO originals and five full messages
per image. This means five independent images, not 25 independent images.

| Method | Bits | Configurations, 25 cases each |
|---|---:|---|
| RoSteALS | 100 | Historical central 256 crop; full RGB512 original bilinearly resized to 256 |
| SepWater | 100 | Historical central 224 crop; central 256 crop; full RGB512 resized to 256 |
| DistSeal post-hoc DC-AE | 64 | Historical RGB512 carrier and selected DC-AE / watermark checkpoint |
| EditGuard copyright | 64 | Historical RGB512 + `degrade.pth`; same inputs and messages + author `clean.pth` |

Total: **8 configurations × 5 images × 5 messages = 200 cases**.
There are no attacks, training, ECC additions, payload shortening, alpha search
or selection of favourable messages. The 256px SepWater paths are diagnostic
native-core experiments, not a claim that the author's fixed-224 CLI uses them.
A geometry rejected by the checkpoint is a visible error, never silently resized.

For each case record full-message BER and exact recovery before clipping,
after clipping, after 8-bit rounding in memory, after PNG reload, and via the
file API where applicable. Check RGB/BGR conventions, eval mode, weight loading,
source revisions, complete payload length, historical replay and PNG integrity.
EditGuard retains its native decoder quantizer in the main path and also reports
an explicitly labelled no-quantizer control. Localization is not evaluated.

RoSteALS, SepWater and DistSeal also decode the **same original latent without
watermark**. Report PSNR(original, AE), PSNR(original, marked) and PSNR(AE, marked).
These are different comparisons, not additive components. The main quality
reference remains the original input. Clean-input/AE message scores are controls,
not estimates of a calibrated false-positive rate.

## What is already known

The earlier saved diagnostics already contain 25 same-latent AE controls for each
latent method. They were absent from the short summary, not absent from the run:

| Method | Original → AE, dB | Original → marked, dB | AE → marked, dB |
|---|---:|---:|---:|
| RoSteALS | 24.670 | 24.014 | 28.508 |
| SepWater | 24.548 | 23.635 | 27.853 |
| DistSeal post-hoc DC-AE | 22.571 | 22.120 | 28.659 |

Source: [saved native diagnostic report](https://drive.google.com/file/d/1a-w65dUww6Ew_1x55C0QQ6zojgP_Xc1l/view),
all `ae_png_control.json` records, reviewed 29 September 2026. Each number is
mean per-case PSNR. Reconstruction already accounts for substantial distortion,
but these numbers alone do not establish why or identify an implementation bug.
The new run repeats paired baselines in the same environment as the predefined
geometry/checkpoint comparisons; it is not a request to redo the entire bank.

The [pinned EditGuard README](https://github.com/xuanyuzhang21/EditGuard/blob/174778fb371640bb31b32fcb9f7bfd440c88766b/README.md)
explicitly distinguishes `clean.pth` (better fidelity) from `degrade.pth`.
The latter remains the historical baseline. The clean checkpoint is downloaded
from that author link, its first downloaded SHA-256 is recorded and pinned on
resume (the authors do not publish a checksum). It must pass a restricted tensor
load and strict complete state comparison. This is not a prevalidated new weight.

## Interpretation and next decision

`measured` means the check ran, not that BER is zero or the paper was reproduced.
Do not suppress any of the eight rows. Crop-vs-resize compares different image
content/frequency distributions; clean-vs-degrade compares different weights.
Neither is an isolated architecture comparison. If a variant is chosen after
this review, freeze the choice and validate on different images before the
100-image bank. Existing raw/PNG errors remain reported.

The source archive embeds the previously tested inference stack, pinned to
SHA-256 `e45bade8d2431d88ae6346bcca19b9f954fa7979deb7c439d90e156ef39fec13`.
It uses the old four model definitions solely to replay the old evidence.
Retired methods cannot be selected by this controller. Their frozen historical
support code is not an active benchmark invitation.

## Development checks

`python build_notebook.py` builds a deterministic readable ZIP and notebook;
all packaged Python and notebook code cells are parsed/compiled.
`python -m pytest tests -q` exercises the shipped archive, geometry, full-message
scoring, checkpoint loading, channel order, summary denominators and safe resume
with CPU test doubles. These tests are not substitutes for the four-model GPU
run. See `VALIDATION.md` for checks actually executed for this release.
