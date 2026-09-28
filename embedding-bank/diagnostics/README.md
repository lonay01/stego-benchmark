# Embedding-bank diagnostic recheck

This is a diagnostic branch, not a new benchmark protocol. It reads the existing
bank and writes to a separate folder. It does not modify production adapters,
replace encoded images, relax PSNR tolerances, sample easier messages, or train
models. A completed check may contain a nonzero BER.

## Run in Colab

1. Upload `dist/Embedding_Bank_DIAGNOSTICS_RESUMABLE.ipynb` to Colab.
2. Prefer the NVIDIA A100/high-RAM runtime used for the original bank. ChunkySeal
   is the demanding part; a T4 is not a compatibility guarantee. The diagnostic
   notebook does not reserve or pay for a runtime automatically.
3. Keep `N_IMAGES = 5`. Check `BANK_ROOT` points to the existing
   `watermarking_embedding_bank_v1`, including `_inputs` and all `_meta` files.
4. Run all four code cells. Mount Drive and upload the accompanying
   `Embedding_Bank_DIAGNOSTICS_source.zip` when asked. It contains readable Python
   source and pinned dependency locks, not Base64 or model weights.
5. Read `watermarking_embedding_diagnostics_v1/REPORT.md`. Share
   `diagnostics_report.zip` from the same directory for review.

The historical bank, diagnostic output, and temporary scratch must be three
separate, non-nested folders. The notebook checks this before preparing models.
Cloud Colab cannot access a local E: drive directly.

One method is prepared, checked, and unloaded at a time. Each has a separate
Python environment and process. Model environments, downloaded temporary model
weights, and source checkouts are removed from owned scratch directories after
each method. New author assets for previously blocked methods are retained in
the diagnostic Drive folder so they need not be downloaded again.

Allow approximately 20 GiB of free scratch reserve in addition to the active
model's files; the runtime monitor stops before this reserve is exhausted. The
per-model temporary cap is 150 GiB. Preparation has a 60-minute timeout and each
model's five-image diagnostic a 30-minute timeout. These are safety limits, not
predictions of completion time. A large part of first-run time may be dependency
installation and checkpoint transfer, not GPU inference.

## Exact checks

| Check | Inputs and measurements |
|---|---|
| FIN | Same 64-bit payload and HEAVY FED+INL weights. Author PIL Resize128, floating output, clipping, author PNG serializer; compare the current tensor-resize128 path and its residual lift to512. No-INL is a labelled ablation, not a replacement decoder. |
| HiDDeN benchmark | Same 30-bit payload and pinned PyTorch checkpoint. H/W read from checkpoint config. Author CenterCrop baseline, floating/clipped/PNG stages; WIBE resize baseline; current residual-to512 path. This is not a claim to test the original Torch7 release. |
| TrustMark-Q/B/P, ChunkySeal, RoSteALS, ARWGAN | Decode archived PNG; recalculate full-payload and common30 errors from prior decoder bits; re-embed the same source/message/seed; compare pixels and decode the new PNG. Also decode the original as a labelled clean control. |
| DistSeal | Same DC-AE and watermark weights. Save autoencoder-only reconstruction and watermarked reconstruction; measure original→AE, original→watermarked and AE→watermarked PSNR; decode AE-only as a control and marked output before/after PNG. |
| Matched PSNR | Inspect only historical `match_not_found_within_policy` records for the selected images across all32 catalog entries. Reproduce the old search and independently search the same scalar expression in SSE space; check floating-point neighbours. No decoder is used to choose alpha. |
| Missing methods | Download published candidates, record source/size/hash and tensor-name inventory, then run strict existing adapters when prerequisites are available. Never turn a successful download into a passed inference result. |

Five images are the first five entries of the frozen bank manifest, not a fresh
COCO sample. The notebook checks original hashes and reuses the exact full native
payload from the bank's deterministic mapping. For methods with historical
records, payload and common30 indices must agree with those records. Existing
model/checkpoint identities must match the historical bank before a recheck.

Both full-payload BER and common30 BER are reported, with wrong bit positions.
The expected answer is used by metric calculation, not passed into model tensor
decoders. The existing blind adapter API is retained. A clean-image coincidence
on five samples is a diagnostic observation, not an FPR estimate.

## Scalar PSNR policy stays unchanged

`uint8(round(clip(x + alpha * (y_native_uint8 - x), 0, 255)))`, NumPy float64
arithmetic, ties-to-even rounding, `0 <= alpha <= 4`, target39/42/45 dB, tolerance
±0.05 dB. The audit groups identical `(original pixel, integer residual)` pairs;
their weighted squared errors are exactly equivalent to evaluating the whole
image. It searches independently in squared-error space and tests adjacent
floating-point values. Every reported successful candidate is checked again on
the full image grid. No diagnostic candidate replaces a bank PNG.

An observed gap is reported as an observed numerical gap, not a proof that a
target is mathematically impossible. Reconstruction-only DistSeal images are
controls, not new references for matched quality. The three PSNRs are not additive.

## Previously blocked methods

- **EditGuard:** the author-linked `degrade.pth` can be downloaded automatically.
  The existing copyright64 adapter is then tested; this is not the localization
  experiment. [Author release](https://github.com/xuanyuzhang21/EditGuard).
- **SepWater:** the author folder publishes `last.ckpt` and `f4.ckpt`. The reviewed
  `last.ckpt` already contains ae/control/decoder weights; it is supplied to the
  existing strict full-model adapter. The redundant 756 MB `f4.ckpt` is not fetched.
  Missing inference keys or incompatible shapes remain failures. The extra AE
  file is not silently merged into a new checkpoint. [Author folder](https://drive.google.com/drive/folders/1NC5q8BsrUntZ6y_5o267GXnaowoNXp90).
- **OmniGuard:** the public `checkpoint.zip` is downloaded and inventoried. The
  copyright adapter is run only if a genuine matching copyright checkpoint is
  present at `author_assets/omniguard/omniguard-copyright-verified.pth` (or in the
  original bank's manual-assets directory, from which it is copied). Public
  localization weights are not relabelled as a copyright encoder/decoder.
  [Author release](https://github.com/xuanyuzhang21/OmniGuard).
- **Fang Screen:** needs an already installed, licensed MATLAB with Image
  Processing and Communications Toolboxes. No license purchase or Octave
  substitution is attempted. [Author code](https://github.com/FangHanNUS/Screen-Shooting-Resilient-Watermarking).
- **Camera-Shooting Instance:** remains integration-blocked. The published
  `premodel.zip` is on Baidu (password0518), and the existing catalog has no
  validated instance-level payload adapter. Installing dependencies alone does
  not implement segmentation/rectification and extraction.
  [Author code and download reference](https://github.com/gyz-star/Camera-Shooting-Resilient-Watermarking-on-Image-Instance-Level).

These last restrictions are explicit limitations, not completed tasks or model
quality findings. Source references were inspected on2026-09-20.

The public OmniGuard archive contains `checkpoint-175.pth`, TrustMark-Q encoder/
decoder files and two hybrid checkpoints. Inspection of `model_checkpoint_01500.pt`
found its `module.bm.*` component, but the author code initializes that component
from TrustMark-Q and freezes it. Its decoder namespace is `decoder.decoder.*`,
not the selected copyright architecture's `decoder.backbone.*`. Renaming those
keys and relabelling the result as a new trained OmniGuard copyright release is
not a valid fix. The diagnostic inventories these files without that substitution.

HiDDeN's WIBE512 wrapper clamps its result internally. Its `bank512_float` label
means before PNG serialization, not before that internal clamp; the separate
checkpoint-native stage is the genuinely unclipped tensor test.

FIN baselines follow the pinned [encode.py](https://github.com/QQiuyp/FIN/blob/f77222ffe9089a5a204e0fd37bd1e33a905be1ed/encode.py),
[decode.py](https://github.com/QQiuyp/FIN/blob/f77222ffe9089a5a204e0fd37bd1e33a905be1ed/decode.py),
and [image loader](https://github.com/QQiuyp/FIN/blob/f77222ffe9089a5a204e0fd37bd1e33a905be1ed/utils/datasets.py).
HiDDeN's CenterCrop follows the pinned
[validation loader](https://github.com/ando-khachatryan/HiDDeN/blob/556f76dd0602e4351ed4c19bd2ee87d50ef3c0de/utils.py).

## Resume and integrity

- Results are saved after every diagnostic stage and image. Measured poor BER is
  kept; it is not retried until a favourable result appears.
- Technical errors are retained in history and may be retried on a later run.
- Completed cases are reused only if their signatures and evidence hashes agree.
- Original image hashes, payloads, source code, model registry and policy are
  fixed. Changing the scope/code requires a new diagnostic output folder.
- Dependency locks and runtime/weight identities are archived before scratch
  cleanup. A Colab reset can require reinstallation, but not redoing completed
  cases. Do not run two diagnostic controllers on the same output folder.
- A model's `completed` status means the diagnostic finished. Inspect exact
  recovery counts and stage errors before deciding on a production change.

## Local development

The controller needs Python3.10+, NumPy, Pillow, psutil, gdown and uv. Model
dependencies are installed from the existing per-model registries/locks.

```bash
export PYTHONPATH="$PWD/src:$PWD/diagnostics"
python -m pytest diagnostics/tests -q
python diagnostics/build_notebook.py
python -m bank_diagnostics.controller run --settings diagnostic_settings.json
```

Tests use deterministic synthetic fixtures and small tensor stubs; they are not
evidence of a successful five-image GPU run. Any real-checkpoint smoke evidence
is labelled separately in `dist/VALIDATION.md`. The production code remains unchanged.
