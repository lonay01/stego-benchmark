# RoSteALS: author-path recovery

This release fixes the observed `ImportError: cannot import name 'corrupt' from
'imagenet_c' (unknown location)`. The parent added `ImageNet-C` to `PYTHONPATH`,
which exposed an outer namespace directory. The actual package is nested at
`ImageNet-C/imagenet_c/imagenet_c/__init__.py`; its import root must be
`ImageNet-C/imagenet_c`. Both environment preparation and subsequent workers now
use that root, including the subprocess running the unmodified author CLI.

No upstream model/attack code, checkpoint, payload, seed or serializer is changed.
The previous release, results and production embedding bank remain untouched.
SepWater and DistSeal are not scheduled. This is not a claim of zero BER or an
exact reproduction of the paper's test dataset/distribution.

## Run in Colab

1. Download `dist/RoSteALS_AUTHOR_FIX.ipynb` and `dist/RoSteALS_AUTHOR_FIX_source.zip`.
2. Upload the notebook to Colab. Choose **A100 + high RAM**, matching the previous
   recorded A100-SXM4-80GB environment. A different GPU stops with a visible
   identity mismatch; do not bypass it or delete the old evidence.
3. Run all cells. Authorize Drive, then upload the **new source ZIP** when asked.
   Do not upload the old RoSteALS/SepWater source ZIP.
4. Leave default paths unchanged:
   - Read-only input: `MyDrive/watermarking_rosteals_sepwater_recovery_v1`
   - New results: `MyDrive/watermarking_rosteals_author_fix_v1`
   - Temporary environments: `/content/rosteals_author_fix_scratch_v1`
5. Wait for `IMPORT AND ATTACK CHECK PASSED: 70/70` and
   `PREFLIGHT PASSED: rosteals` before leaving it unattended.
6. Send `MyDrive/watermarking_rosteals_author_fix_v1/RoSteALS_AUTHOR_FIX_report.zip`.
   It is also downloaded by the final cell. Send it even if a stage failed.

The full previous folder is needed, not just a report archive. Inputs and
receipted RoSteALS/source downloads are copied and checksum-verified; SepWater
images and DistSeal weights are not copied. Allow at least 35 GiB free temporary
SSD and, as a planning allowance, about 10 GiB free Drive space. Actual artifact
sizes vary. No long inference is run in the notebook kernel.

`MAX_HOURS=8` is a session budget, not an ETA. After interruption, reconnect with
the same GPU and rerun the same notebook. Completed valid cases are reused;
corruption grids resume at missing/error conditions. Do not run two copies.
Logs and a 30-second controller heartbeat are written to the new Drive folder.
Colab uptime is not guaranteed. Stop/disconnect the runtime when finished to avoid
unnecessary GPU allocation.

## Fixed experiment

The new run inherits the same 100 original COCO JPEG IDs/order from the reviewed
recovery, not the earlier RGB512 centre crops. Author input size is 256×256.

| Stage | Work |
|---|---|
| Import/API check | Actual pinned package origin/hash; all author config imports, ECC clean roundtrip, all 14 × 5 synthetic corruption conditions |
| Author preflight | First image × 5 messages × 2 profiles; complete model construction/weights, PNG reload, repeatability and unmodified author CLI comparison |
| Native measurement | 100 images × 5 messages × raw100/BCH-text = 1,000 cases |
| Corruptions | 500 raw100 internal PNGs × 14 author test-pool operators × 5 severities = 35,000 tests |

Native measures include raw/clipped/internal-quantized/exported-PNG full-payload
BER, exact recovery, author word-accuracy threshold, legacy BCH text recovery,
PSNR, SSIM, LPIPS and SIFID. Source→AE, source→marked and AE→marked comparisons are
kept separate. Residual export returns to the original JPEG dimensions, exactly
as in the author inference script. Attacks use the internal floor PNG, not that
original-resolution export. This expanded grid differs from the paper's random
severity sampling. No training, alpha search, payload shortening or score-based
configuration selection is performed.

Raw100 means 100 independently generated bits. The separate text cohort uses
seven hex ASCII characters: 56 data positions, 28 bits of entropy, and the
author's BCH encoding to 100 positions. Legacy ECC outputs remain labelled;
SepWater's corrected BCH decoder is not silently substituted. Nonzero BER is
not a technical preflight failure. A failed import is not BER=0 or BER=1.

## Integrity and validation

The parent run is checksum-identified as
`31e475a2773d73beafdcfcfc7704fcee8069a0a0782b67552bad37d4e85e1565`, using source ZIP
`42ccd248d1764e78cc4a662626b07dab63778d83c56037bf47f60dbcca11c8dd`.
The new code hash is part of the new run identity. Original inputs, receipts,
source audits, environment pins and GPU identity are verified. No historical
result is relabelled as newly measured. The recorded parent had zero RoSteALS
observations; a parent with additional results is rejected for review.

`build_notebook.py` packages readable sources from the immutable parent ZIP.
The only change to inherited inference infrastructure is the two import-root
expressions in `authornight/environment.py`, documented in
`reference/import_fix.json`. A new controller/report limits execution to RoSteALS.

Run `python validate_release.py` for static checks and CPU unit/smoke tests.
`smoke_imports.py` checks the real pinned repository layout without importing
model packages. Full Linux/PyTorch 1.11/CUDA execution is not established by local
Windows tests: runtime import/attack and neural preflights must pass in Colab.
The additional local Python 3.9 environment attempt could not build the author's
`bchlib==0.14.0` because Windows C++ Build Tools are unavailable. It is not reported
as a successful full import smoke. Colab setup installs `build-essential`; the
runtime checks use the unchanged pinned Linux/BCH environment.

Upstream snapshots are unchanged:
- [RoSteALS](https://github.com/TuBui/RoSteALS/tree/8c09b022a362ffdf977a86f355e5af339793d9c9)
- [ImageNet-C](https://github.com/hendrycks/robustness/tree/8190fe329f5f072a06e3a2aea02bb5dda69aed9f/ImageNet-C/imagenet_c)

DistSeal continuation instructions remain in the previous recovery folder as
`DISTSEAL_RESUME.md` and in `../author-recovery/dist/DISTSEAL_RESUME.md` locally.
