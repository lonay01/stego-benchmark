# COCO neural channel

Evaluate the existing 29-method production embedding bank. No encoder runs here;
no message, checkpoint, native geometry or Matched Quality setting is changed.
The digital run has its own processes, output and scratch directories and is not
modified by this notebook.

## Files and launch

Use `dist/COCO_Neural_Channel_RESUMABLE.ipynb` together with the matching
`dist/COCO_Neural_Channel_RESUMABLE_source.zip`. The ZIP contains readable Python
sources, frozen decoder dependencies, tests and protocol snapshots. There is no
Base64 payload in a code cell, no `git clone main` at runtime and no dependency on
GitHub authentication. The notebook checks the exact ZIP checksum.

1. Open the notebook in Colab. Select **A100, high RAM**, preferably the same
   A100 runtime as the reviewed bank. A T4 is not the recommended target for this
   full canary. The current Windows machine cannot run all 29 environments.
2. Leave `N_IMAGES = 5`, `MODELS = "all"`, `GPU_BATCH_SIZE = 4`. Mount Drive and
   point `BANK_ROOT` to the completed `watermarking_production_bank_v1` folder.
3. Run cells in order. Upload the source ZIP when prompted. The notebook keeps
   a copy on Drive, so subsequent sessions need no manual upload. Dependencies
   and checkpoint downloads are visible in the setup/stage logs.
4. Review `_reports/REPORT.md`, coverage and per-method Clean controls. A
   successful notebook run is not a claim of good watermark robustness: a
   measured high BER is a valid result. Safety blocks remain visible.
5. Increase **only `N_IMAGES` to 100**. Keep source release, bank, output and
   batch size unchanged. The original five images form the same prefix; their
   completed results are reused. Do not copy results into a different protocol.

A separate Linux NVIDIA server uses the same notebook. Set the three filesystem
paths to mounted persistent storage and local SSD. Changing hardware or the
numerical environment may fail the strict execution/Clean replay checks; it is
not automatically equivalent just because the CUDA code runs. Prefer the same
GPU type for canary and full run. This version targets CUDA 12 / PyTorch 2.5.1
hardware such as A100/H100, not unvalidated newer architectures.

Do not start this notebook in the **same Colab runtime** as an ongoing digital
run. A different Colab session or server is needed for actual simultaneous
experiments; GPU availability and service limits are not guaranteed.

## Implemented protocol

The source is `docs/neural-channel-protocol.md` from local Git commit
`bce02b45d9a1146164248dd5c736cd09dc343672`. GitHub connector access returned 404
and direct Git access failed DNS during preparation on 7 October 2026. Therefore
this release identifies and bundles that committed protocol; it does **not**
claim that a newer remote revision was checked. Its file hash is in the release.

- Same saved carriers and five full-native-payload messages as digital.
- Native and Matched 39/42/45 dB remain separate. Unmatched pairs have no carrier
  and are counted as missing coverage, not silently generated or re-matched.
- **21 conditions:** Clean; SDXL VAE posterior mode, one/two passes; SDXL
  img2img strengths .15/.30/.50 with three repeats each; InstructPix2Pix golden
  hour and watercolor instructions, three repeats each; central-25% SDXL
  inpainting, three repeats.
- Pinned checkpoint revisions, exact prompts, 20 requested steps, scheduler,
  precision, mask and guidance settings are in `src/wmbench_neural/spec.py` and
  `reference/neural-channel-protocol.md`. Executed timesteps are recorded.
- COCO captions: minimum annotation ID, fixed once for all 100 images. The
  official annotation JSON and hash are retained. Download uses the valid-TLS
  COCO S3 endpoint; certificate checks are not disabled.
- Seeds depend on COCO ID, message replicate and attack condition, not method,
  embedding mode, N, batching order or restart. Each item gets a separate CPU
  generator. No seed is selected using a decoder result.
- Standalone VAE uses FP32 and quantizes between its two passes. Diffusion uses
  FP16 with the checkpoint's required VAE upcasting. Extra SDXL watermarking is
  off; the shipped safety checker is retained. Safety placeholders are not
  decoded. Inpainting output is not pasted onto the original background.

### Native resolution and controls

Clean reads the saved **native-size** PNG, without resizing. Other conditions
use the fixed, recorded transform `native -> bicubic 512 -> attack -> RGB8 PNG
-> bicubic native -> unchanged decoder`. No encoder is rerun at 512. The
auxiliary `resize_only` control follows that resize round trip without a neural
model, so resizing loss can be seen separately. It is excluded from the
21-condition matrix and four-family macro. The protocol leaves interpolation
unspecified; this release fixes Pillow **10.4.0 BICUBIC** before measurement.
Both resized attack inputs and native-size decoder inputs are prepared in the
common environment, not in the method-specific Pillow versions. The latter PNGs
are included in each temporary ZIP when their size differs from 512.

All methods retain the reviewed production implementations, including the
author RoSteALS/SepWater fixes, Meta HiDDeN/FNNS helpers and DistSeal post-hoc.
Strict Clean replay is checked **before** neural generation and again on decoder
loading. StegaStamp uses the already audited paired CUDA profile: first fresh
Clean, three repeats in one ONNX session and one in a new session, no voting.
Historical bank Clean is preserved and reported separately. Other methods do
not receive a relaxed Clean tolerance.

## Parallelism and memory

- The expensive neural pipelines process fixed-width GPU batches. Default: four
  images per call, with deterministic duplicate padding of incomplete batches.
  The padding outputs are discarded, never scored. Batch width is frozen in the
  run identity; changing N does not change it.
- One CPU reader prefetches the next input group. Two bounded CPU workers encode
  PNGs, calculate PSNR/SSIM and save/check ZIPs on Drive while the next GPU batch
  runs. At most two groups wait for those workers.
- Byte-verified carrier PNGs are cached once on the local SSD for all attack
  stages of the method; the code does not reread every carrier from Drive 20
  times. Execution manifests are referenced by hash rather than duplicated in
  every task's JSON.
- Decoder installation (CPU/network) overlaps GPU embedding-quality measurement.
  Each watermark decoder keeps its own original environment. No incompatible
  packages are installed into a shared watermark environment.
- One heavy neural pipeline occupies the GPU at a time. Its process exits before
  the method decoder starts. The pipeline stages are VAE, SDXL img2img,
  InstructPix2Pix and inpainting; each is decoded and cleaned up before the next.
  This avoids keeping every method or several large diffusion models resident.
- Decoders remain at their reviewed single-image API. Unvalidated decoder
  batching, shared-model threads and mixed CUDA processes are not enabled just
  to make a GPU-utilization number larger.

Actual utilization and throughput must be measured on the canary. There is no
claim that this saturates every GPU or makes a million generations cheap.
Batch timings, actual steps, stage wall times and resource telemetry are saved.
Amortized attack seconds are batch time divided by fixed batch width and are
not single-image latency; padding and real item counts are both recorded.
The implementation follows the Diffusers
[batch/independent-generator interface](https://huggingface.co/docs/diffusers/main/using-diffusers/batched_inference)
and validates the actual installed **0.32.2** call signatures in its CPU tests.

If four images do not fit, OOM stops safely; no hidden batch-size, precision,
step-count or offloading change is made. Start a **new** canary output with
`GPU_BATCH_SIZE = 2` (or 1). Pick a successful profile before the full run.

## Storage, restart and deletion

Default durable root: `MyDrive/watermarking_neural_channel_v1`.
Default local scratch: `/content/wmbench_neural_scratch`. Keep at least 60 GiB
free before a method, with additional room for cached SDXL/IP2P weights.

The user explicitly chose temporary image retention on 7 October 2026. This
overrides permanent attacked-image retention in protocol section 7:

1. Neural outputs are generated in RAM and saved as temporary **Drive ZIPs**,
   grouped by method/attack/COCO ID with mode folders inside. This deliberately
   uses Drive while outputs are expensive and not yet decoded, so a Colab reset
   does not discard a completed generation stage.
2. Each ZIP and manifest is read back and hashed. Each valid PNG is restored,
   transformed to decoder dimensions, decoded and scored against the full
   message. Results are independently checksummed and read back on Drive.
3. Only after every item in that ZIP has a durable terminal record is the exact
   owned ZIP deleted. A deletion receipt and every generated PNG hash remain.
   No original bank file or old experiment is deleted.

Temporary Drive usage is bounded by **one method and one pipeline stage**, not
all methods. At N=100, the largest stage has up to 18,000 PNGs (9 SDXL repeats ×
2,000 eligible carriers), plus native-size decoder PNGs for smaller carriers.
At 500 KiB per 512 PNG the first part is about 8.6 GiB, but actual sizes
vary; leave tens of GiB free on Drive. Download caches/environments use local
SSD. Google Drive mount capacity reporting is not a reliable quota check.

After a disconnect, rerun the same cells. Complete results are never retried to
improve BER. Existing undecoded ZIPs are reused after verification. A partial
generation group may be repeated with identical seeds; conflicting bytes cause
a stop, not silent replacement. Completed results also survive an empty local
scratch directory. No promise of bit-identical output across different GPU
types or software stacks is made.

Deleting intermediate images makes exact visual reinspection impossible without
regeneration. Hashes/configurations remain, but hashes are not a substitute for
the image bytes. Keep selected qualitative examples before deletion only under
a separately declared, decoder-independent selection policy; this release does
not pick favourable examples.

## Results

`<method>/records/*.json.gz` contains full decoded bits, status, paired Clean,
resize-only control, seeds, PNG/decoder-input hashes and per-task metadata.
`<method>/tables/` contains:

- `reference_and_clean.csv`: historical and freshly replayed Clean, resize
  control, capacity/interface, actual bpp, native dimensions and embedding
  PSNR/SSIM/LPIPS. BER here is a fraction.
- `per_attack.csv.gz`: condition/repeat, BER/exact, paired degradation,
  input/output PSNR/SSIM/LPIPS, inpainting inside/outside-mask MAE and cost.
- `attack_summary.csv`: average repeats/messages within each image, then images.
- `macro_summary.csv`: equal weights for four families; Clean excluded. Missing
  family observations prevent a complete macro score for that image.
- `matching_coverage.csv`: planned/saved/unmatched before neural attacks.

Aggregate BER/exact values are percentages; differences are percentage points.
`attack_psnr_finite` aggregates finite PSNR observations only and reports their
image count; exact infinite values are preserved in the per-attack table.
95% intervals use 2,000 image-cluster bootstrap samples with a fixed seed.
`_reports/pairwise_matched.csv` uses common image/message pairs and all repeats.
Native/post-ECC interfaces and message lengths are disclosed, not pooled into a
capacity-weighted winner. Reference papers are archived as context; missing
verified numeric author results are labelled, never filled with invented values.
Stage wall times and observed resource peaks are in `_reports/performance.csv`.
Do not sum concurrent prepare/quality stage times as elapsed wall-clock duration.

## Validation and limits

See `VALIDATION.md` and `dist/validation-results.json`. Static checks and local
CPU contract/integration tests do not establish CUDA model success. The first
real five-image run is the end-to-end GPU acceptance test. This release does
not launch a paid experiment automatically.
