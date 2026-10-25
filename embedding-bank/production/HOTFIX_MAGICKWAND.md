# RoSteALS / SepWater: notebook-only system-library repair

## Observed failure

The executed `COCO_Embedding_Bank_PRODUCTION.ipynb` supplied by the user contains
the same exception for both methods: `ImportError: MagickWand shared library not
found`. RoSteALS reaches it through `imagenet_c.corruptions`; SepWater through
`Noise_Layer.noise3`. Both import `wand.image` while constructing the author
model, even though this stage does not run attacks. The Python Wand packages
are already present; their native ImageMagick dependency is missing.

The previous `rosteals-recovery/build_notebook.py` (RoSteALS_AUTHOR_FIX) and
`author-recovery/build_notebook.py` (RoSteALS_SepWater_RECOVERY) installed
`libmagickwand-dev`, `build-essential` and `libgl1` in their Colab setup cells.
The production setup retained the compiler check but omitted the native image
libraries. This repair restores that setup, not another model implementation.

## Scope

Only setup cell 2 and its explanatory Markdown change. When RoSteALS or SepWater
is selected, the notebook checks loadable MagickWand and OpenGL libraries. In
Colab it installs only missing system packages, then checks again. Outside
Colab it provides Debian/Ubuntu installation instructions instead of silently
modifying the server. Missing dependencies fail before any model is dispatched.
No pip package, adapter, checkpoint, lock, payload, crop, serialization rule or
PSNR matching policy is changed by this repair.

The existing source ZIP remains byte-identical:

```text
6d7ddfcd05abdaf46f97ab7b6b174bff098b3043edcf069db64041554263f7b5
```

Consequently the release digest and scientific run identity remain unchanged.
The original ZIP can be reused from the Drive cache. No migration, deletion,
replacement of result records, or bypass of evidence validation is required.

## Resume

1. Open `dist/COCO_Embedding_Bank_PRODUCTION_MAGICKWAND_FIX.ipynb` in Colab.
2. Keep the previous A100 configuration, `N_IMAGES = 5`, `MODELS = "all"` and
   the existing `OUTPUT_ROOT` (`MyDrive/watermarking_production_bank_v1`).
3. Do not run the old and new notebooks against the bank simultaneously.
4. Run all cells. The setup prints a successful system-library preflight.
5. The controller validates completed cases and prints
   `REUSE: no embedding or decoding required` for the 27 completed methods.
   Only incomplete methods need model preparation and inference.
6. Inspect the final combined report before increasing N. Old unmatched targets
   and nonzero BER values are retained; this fix does not try to improve them.

## Executed checks

- Source ZIP checksum unchanged; all 107 packaged files unchanged.
- All four notebook code cells and 68 packaged Python modules parse; nbformat
  schema validation passes.
- Comparison against the supplied executed notebook confirms unchanged settings,
  execution and report cells. Setup changes are exactly the added preflight.
- Nine hotfix tests cover conditional installation, no-op for other methods,
  already-installed libraries, unloadable libraries, failed installation and
  server instructions. Package-manager calls in these tests are mocked.
- Fourteen bank regression tests pass, including actual Meta HiDDeN CPU smoke
  and a synthetic 27-complete/2-failed scheduler test. That test dispatches only
  RoSteALS/SepWater and verifies retained fixture PNG/status hashes.

These local checks do not constitute a new RoSteALS/SepWater GPU run. Real apt
installation and both author models must still be exercised in Colab. Their new
PSNR/BER are not known until that retry finishes. Machine-readable check output
is in `validation/local_checks.json`. The archived `VALIDATION.md` remains the
original release record deliberately, so its bytes do not invalidate resume.
