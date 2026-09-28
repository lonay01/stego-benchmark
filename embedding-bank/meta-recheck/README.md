# Meta Revisited HiDDeN, SepWater and RoSteALS

A separate diagnostic before changing the production embedding bank. No old
result, checkpoint, payload or adapter is overwritten. This is not a new ranking
and not a reproduction of each paper's full evaluation.

## Run in Colab

1. Upload `dist/Meta_SepWater_RoSteALS_CHECK.ipynb`.
2. Select an A100 GPU and high system RAM, as in the previous diagnostics.
3. Run all four code cells. Mount Drive and upload the matching
   `Meta_SepWater_RoSteALS_CHECK_source.zip` when asked.
4. Keep `N_IMAGES=5`. Change paths only if your previous bank is elsewhere.
5. Send `Meta_SepWater_RoSteALS_report.zip` from the new output folder.

The notebook needs the existing native-resolution canary, not just its report:
`/content/drive/MyDrive/watermarking_native_resolution_canary_v1`. Its frozen
manifest, RGB512 originals, RoSteALS/SepWater PNGs, decoder records, checkpoint
identities and dependency locks must remain available. The reused reference
preflight requires ten manifest originals, although only the first five are
measured here. It will not silently download substitute COCO images.

New results: `/content/drive/MyDrive/watermarking_meta_sepwater_rosteals_v1`.
Temporary installations: `/content/watermarking_meta_recheck_scratch_v1`.
Final human-readable report: `_reports/REPORT.md` in the new result folder.
The matching source ZIP is cached under `_release` after the first upload.

After disconnection, run cells 1–3 again with the same settings. Completed cases,
including nonzero BER, are reused after SHA256 artifact validation. Technical
errors are archived before retry. Corrupted or incompatible evidence stops the
run rather than being overwritten. Keep the same GPU/runtime during each paired
comparison. The previous five-method reference run is read neither as an output
destination nor as a source of new measurements; the native bank remains read-only.

## Fixed test matrix

| Method/configuration | Carrier | Payload | Cases |
|---|---:|---:|---:|
| Meta Revisited HiDDeN, checkpoint configuration | Central 256×256 crop | 48 bits | 25 |
| Meta Revisited HiDDeN, official demo configuration | Original RGB512 | Same 48 bits | 25 |
| RoSteALS, historical checkpoint/config | Central 256×256 crop | Historical raw100 | 25 |
| SepWater, historical checkpoint/config | Central 224×224 crop | Historical raw100 | 25 |

Five images × five fixed random messages per row. These are 100 image/message/
configuration cases, not 100 independent images. No attacks, external alpha search,
matched PSNR, training, shorter messages, repetitions or BCH/ECC are added.
The five source images are the same as the historical canary, not a held-out test
for model selection. No claim is made that COCO images were absent from training.

### Meta: two configurations, not two checkpoints

The checkpoint is `hidden/ckpts/hidden_replicate.pth` from the official
[Meta repository](https://github.com/facebookresearch/stable_signature/tree/c91217c06e953a4eaf270ef9de4a3939bcea3366/hidden).
It is **Revisited HiDDeN**, a modified method, not the original 2018-author weights.
SHA256: `5a83d4d5c85f6d58abef7d1083d559dae1eb9171bf63ca0e0c13016fd4741443`.

- `checkpoint_256`: settings stored in the checkpoint: 256 pixels, 48 bits,
  no JND, image scaling 1, watermark scaling 0.3, no channel scaling.
- `demo_512`: settings in Meta's checked-in demo: 512 pixels, 48 bits, JND,
  image scaling 1, watermark scaling 1.5, no channel scaling.
- Both profiles are fixed before the test; do not pick the winner and hide the
  other. Differences cannot be attributed to resolution alone because JND and
  strength also differ. Neither is matched-quality against the other methods.
- Inputs come from existing RGB512 PNGs. The 256 input is a literal central crop,
  not a downsample. This differs from the demo's arbitrary-image resize.
- Encoder: 4 blocks/64 channels; decoder: 8 blocks/64 channels; ImageNet
  normalization; bipolar message; decoder threshold strictly greater than zero.
- Four unchanged inference classes are loaded from the hash-verified `models.py`.
  Unused ViT/DVMark classes, timm imports and augmentation/training modules are not
  executed. JND uses the unchanged author module. No network code is rewritten.
- The 5.5 MiB checkpoint is loaded with `weights_only=True` and only
  `argparse.Namespace` allowlisted for its training metadata. No unsafe fallback.
- Save raw float RGB; decode raw, clipped, rounded-in-memory, and reloaded PNG.
  PNG follows the demo's clip → round → ToPILImage sequence. Decode the clean
  source as a negative control, not as a calibrated detector/FPR measurement.

Meta uses its own isolated Python 3.10.12 / torch 2.5.1 / torchvision 0.20.1
environment. This is a compatibility runtime, not the authors' 2023 runtime.
Transitive dependencies are locked on the first preparation and stored on Drive;
subsequent runs reuse that lock and compare the installed inventory. This is
first-run resolution followed by locked resume, not an original-author lockfile.
All source Git blob hashes, checkpoint SHA256 and the full saved configuration
are retained. Only reviewed sources/weights are fetched; no diffusion model.

### RoSteALS and SepWater: repeat the problematic cases

The previous reference release is included as a checksum-pinned dependency.
Its adapters, registry, inference and scoring code are byte-for-byte preserved.
RoSteALS uses WIBE's pinned fork/distributed checkpoint, not a newly selected
author release. SepWater reuses the already audited checkpoint.

For each method: run the frozen adapter, reload a fresh upstream neural core in
a separate process, compare every inference weight/buffer, encode the same crop
and payload, and cross-decode both branches' float output and PNG files. Store
full bit vectors, logits, image differences, PSNR and per-case times.

The earlier RoSteALS error was in the diagnostic import path: WIBE's YAML uses
`RoSteALS.*`, requiring the parent `submodules` directory in addition to the legacy
`ldm` root. This release adds and checks those paths only for the fresh reference
process. It does not alter YAML targets, tensors, thresholds or checkpoint keys.

SepWater is rerun unchanged. Both methods receive the same arbitrary 100 bits,
not the authors' optional text+BCH payload. Source-defined residual PNG export
and the adapter's PNG export remain separate. `core_agreement` can coexist with
nonzero BER; `review_difference` requires investigation, not score tuning.

## Storage, resources and interpretation

Only one model environment is installed at a time. Successful scratch slots are
removed after all measurements are verified. Failed slots are retained for
diagnosis; they may consume disk until inspected or a fresh runtime is used.
Nothing under the old bank or result folder is removed. Start with at least
30 GiB free scratch; the controller stops below 20 GiB and caps each slot at
100 GiB. A100/high RAM is the recommended first-run environment, not a measured
minimum. The six-hour overall budget and 90-minute phase cap are safety limits,
not runtime estimates or guarantees of Colab availability.

Exact whole-message recovery and BER are both reported with their measured
denominators. Missing/error cases are not successes and not BER=0. Raw-to-PNG
degradation is visible separately. Twenty-five messages on five images are a
diagnostic, not evidence for architecture-level superiority or paper-level BER.

Full PNG/NPY evidence stays on Drive. The small review ZIP includes JSON, CSV,
logs, environment locks and reports, but not weights or source images. Report
generation reads small records; resume, not the report, validates large artifacts.

## Development

Build: `python build_notebook.py`. Run CPU tests: `python -m pytest tests -q`.
The builder refuses a changed previous-release ZIP. It compiles every packaged
Python module and all notebook cells. See `VALIDATION.md` for actual test scope.
Source files are readable; there is no Base64 code bundle and no automatic Git
push, production promotion, author contact or GPU purchase.
