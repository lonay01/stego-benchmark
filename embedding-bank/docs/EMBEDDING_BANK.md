# Reusable COCO embedding bank

This stage prepares watermarked images once. The same saved PNGs can later be
used for digital attacks, neural transformations and screen–camera experiments.
It does **not** run attacks, generate calibration images, train models or claim
that the complete benchmark has finished.

The requested catalogue contains **32 post-hoc methods and variants**. Inclusion
in the catalogue is not proof that the required weights, software or adapter are
available. Missing prerequisites are reported explicitly; no other method or
checkpoint is substituted to fill a folder.

## 1. Inputs and messages

- Use the fixed historical COCO100 corpus: RGB PNGs, **512 × 512**, with its
  original manifest and image hashes. Do not reconstruct missing originals from
  newly downloaded JPEGs and treat them as the same corpus.
- `N_IMAGES = 100` prepares the full bank. Start with `N_IMAGES = 5` to check the
  complete embedding workflow on the fixed first five images.
- Assign **one deterministic, balanced 30-bit message per image**: 15 zeros and
  15 ones. Store the assignment; use the same message for that image across all
  methods and modes. This is the newly agreed one-message design, not the older
  five-messages-per-image experiment.
- Where a model accepts more than 30 bits, use the registered common-bit
  positions and deterministic filler. Preserve the complete native payload,
  common positions and any reserved bits in metadata. Do not overwrite
  method-specific detection positions or silently truncate a message.
- Record raw capacity, useful payload length and ECC separately. Fang retains
  its published CRC/BCH coding; it is not a raw30 neural watermark.

The five-image run is a prefix of the 100-image run. Changing only `N_IMAGES`
from 5 to 100 must not change the original files, messages or scientific
configuration of the first five images. All 100 originals and their manifest
are frozen when the bank is created, even for a five-image run.

## 2. Embedding and quality modes

For each method–image pair, embed once and save the **Native RGB uint8 PNG**.
Construct the three matched variants from that saved Native image, not from a
second stochastic embedding call or an unsaved floating-point tensor.

| Folder | Rule |
| --- | --- |
| `native/` | Configured method and checkpoint; no external strength matching |
| `matched_39/` | Target PSNR 39 dB, tolerance ±0.05 dB |
| `matched_42/` | Target PSNR 42 dB, tolerance ±0.05 dB |
| `matched_45/` | Target PSNR 45 dB, tolerance ±0.05 dB |

For original pixels `x` and saved Native pixels `y`, both in `[0,255]`:

```text
matched(alpha) = round_ties_to_even(clip(x + alpha * (y - x), 0, 255))
0 <= alpha <= 4
PSNR = 10 * log10(255^2 / mean((x - matched(alpha))^2))
```

Use **one scalar alpha for the whole image**. There is no per-pixel correction,
decoder-guided search, seed selection or message substitution. Check the achieved
PSNR after PNG save/reload. Keep the measured PSNR, alpha and matching status.

If a target cannot be reached within the fixed search policy and tolerance,
record it as unavailable. Do not publish the nearest image as a successful
matched result. Native can still be valid even when a matched mode is unavailable
or clean message extraction is imperfect. Record clean extraction separately;
do not keep trying until a desirable bit accuracy appears.

The optional `identity_check` decodes saved images without an attack and is
enabled by default. Disabling it skips an important integration check; it does
not change embedding or matching and must not be described as verified clean
recovery.

A technical decoding exception is retried on the next run using the same fixed
seed, without re-embedding the Native image. The previous diagnostic is archived.
A successfully measured nonzero BER is never retried to obtain a better message.

## 3. Strict method allowlist

Only these 32 entries belong to this bank:

| Group | Methods / variants |
| --- | --- |
| Existing digital baselines (15) | TrustMark-Q, WAM, VideoSeal, PixelSeal, ChunkySeal, VINE-B, VINE-R, RoSteALS, InvisMark, RivaGAN benchmark, DWT-DCT-SVD, MBRS, CIN, HiDDeN benchmark, EditGuard |
| Additional post-hoc methods (6) | ARWGAN, SSL Watermarking multibit, MaskWM, Robust-Wide, SepMark, OmniGuard |
| Physical specialists (6) | StegaStamp ONNX, PIMoG, SSRH, SepWater, Fang Screen, Camera-Shooting Instance |
| New additions (5) | TrustMark-B, TrustMark-P, FIN, FNNS from OmniSealBench, DistSeal post-hoc DC-AE |

TrustMark variants and benchmark-specific implementations retain their explicit
labels. The nine generation-time methods are outside this stage. Previously
removed methods are not brought back through an old registry's default list.

### Known prerequisites and verification limits

| Entry | Requirement or limitation |
| --- | --- |
| StegaStamp ONNX | Uses the WIBE raw100 ONNX conversion, not a claim of numerical equivalence to every TensorFlow implementation. CUDA inference requires the compatible ONNX Runtime libraries. |
| PIMoG / SSRH | The original networks operate at 128 pixels in BGR. The documented adapter lifts their residual to the 512-pixel carrier. This adaptation stays fixed in all modes. |
| SepWater | Supply the full matching author checkpoint; required encoder, autoencoder and decoder tensors must exist. A missing checkpoint is a blocker. |
| Fang Screen | Requires licensed MATLAB with Image Processing and Communications Toolboxes. MATLAB Runtime or Octave is not a verified substitute. Feature-poor images can fail the original embedding algorithm. |
| Camera-Shooting Instance | The current inventory has no verified runnable embedding/payload contract. It remains blocked until integration and checkpoint validation are completed. |
| EditGuard / OmniGuard | Require the correct copyright-watermark checkpoints; localization-only weights are not interchangeable. |
| New additions | Public source and checkpoints do not replace a canary of the exact selected implementation, payload mapping and target environment. |

Historical real CPU interface checks exist for StegaStamp ONNX, PIMoG and SSRH on
one author-provided image. They do not establish COCO100 coverage or validate the
production GPU environments. Technical failures are neither successful outputs
nor evidence that an algorithm cannot carry the message.

## 4. Output layout

The method is the top-level image folder, followed by the quality mode:

```text
watermarking_embedding_bank_v1/
├── _inputs/                 # Exact originals, manifest and message assignment
│   └── images/              # <12-digit COCO ID>.png
├── bank.json                # Frozen release, model registry and scientific policy
├── _reports/                # Coverage tables and report
├── trustmark_q/
│   ├── native/
│   ├── matched_39/
│   ├── matched_42/
│   ├── matched_45/
│   └── _meta/               # records/, keys/, executions/, logs/
├── wam/
│   ├── native/
│   ├── matched_39/
│   ├── matched_42/
│   └── matched_45/
└── ...
```

Image names identify their COCO source. Metadata links each file to its original,
message, adapter sidecar, model revision, checkpoint hashes and measured quality.
An empty or incomplete mode folder is not a successful result; consult the
coverage report. Per-method records are stored under
`_meta/records/<image_id>_<mode>.json`; decoder sidecars are under
`_meta/keys/<image_id>/metadata.json`. Execution records and logs are retained
alongside them. Start with `_reports/REPORT.md` and its accompanying CSV tables
when reviewing a run.

With 100 images, one message and all four modes available, the upper count is
**12,800 marked PNGs**. At five images it is **640 PNGs**. Model weights, Python
environments and download caches are separate from this persistent image bank.
Size the output and scratch disks separately, using the five-image run to measure
actual file sizes.

## 5. Windows or Colab

### Windows with the Samsung SSD

Use a dedicated directory, not the drive root:

```text
OUTPUT_ROOT = E:/watermarking_embedding_bank_v1
```

Set `SCRATCH_ROOT` to a separate local working directory with sufficient free
space for the active model's source, environment and weights. Persistent results
belong under `OUTPUT_ROOT`; temporary installations do not need to be copied
into the image folders. Keep the SSD connected during writing.

Native Windows is not a blanket compatibility guarantee for all upstream code.
Some WIBE integrations use symbolic links or dependencies requiring compilation;
Windows permissions, toolchains and available GPU wheels must be checked. Fang
can use a locally installed licensed MATLAB. Unsupported environments must fail
with a reason rather than fall back silently to another implementation.

### Colab Pro+

Download both `COCO100_Embedding_Bank_RESUMABLE.ipynb` and
`COCO100_Embedding_Bank_source.zip` from the same release. Open the notebook in
Colab and upload the ZIP when cell 2 asks for it. Alternatively, put the ZIP on
Drive and set `SOURCE_ARCHIVE` to that path. No Base64 source is embedded in the
notebook. The ZIP checksum pins the exact readable Python files used by the run.

Colab with an NVIDIA GPU is the preferred first target for the Python model
environments. Mount Drive and use:

```text
OUTPUT_ROOT = /content/drive/MyDrive/watermarking_embedding_bank_v1
SCRATCH_ROOT = /content/watermarking_embedding_scratch
```

A cloud Colab runtime **cannot write directly to the Samsung SSD connected to
the Windows computer**. Save the durable bank to Drive, then copy it to
`E:/watermarking_embedding_bank_v1`. Local `/content` storage is temporary and
must not be the only location of completed results. Colab does not supply a
licensed MATLAB installation for Fang.

Run from the project environment after preparing the input corpus and settings:

```bash
python -m pip install uv==0.8.15 .
python -m wmbench.embedding_bank run --config settings.json
```

Example `settings.json` for a first local canary; replace the input and scratch
paths with real directories. `coco_root` points to the existing corpus root with
`test/` and `manifest.csv`, not a newly selected COCO subset.

```json
{
  "output_root": "E:/watermarking_embedding_bank_v1",
  "scratch_root": "D:/watermarking_embedding_scratch",
  "coco_root": "D:/watermarking_inputs/coco100",
  "manual_assets_root": "D:/watermarking_inputs/manual_assets",
  "n_images": 5,
  "models": "all",
  "device": "cuda",
  "identity_check": true,
  "cleanup": true,
  "max_hours": 12,
  "prepare_minutes": 60,
  "model_minutes": 180,
  "min_free_disk_gib": 20,
  "min_free_ram_gib": 2,
  "max_slot_gib": 150
}
```

Use `device: "cpu"` for a CPU-capable local canary; not every model is guaranteed
to support CPU inference. `models` can be an explicit list of allowlisted IDs
for diagnosis, but a subset run is not a completed 32-method bank. Time, RAM and
disk limits are execution safeguards, not reductions of the scientific task.
With `cleanup` enabled, the controller can reclaim its managed scratch slot
between models; the persistent bank is not a cleanup target.

The notebook exposes the same operation with editable paths and `N_IMAGES`
(the JSON key is `n_images`).
Begin at five images, inspect coverage and errors, resolve prerequisites, then
increase to 100 without changing the scientific settings.

## 6. Execution, resume and transfer

- Process methods sequentially in isolated environments. A worker holds only
  its own model; incompatible model dependencies are not installed together in
  the notebook kernel.
- Commit records per image and mode. Resume checks the specification and file
  hashes, not merely whether a PNG filename exists. A partially written or
  corrupted file is not complete.
- Valid saved Native images are reused when producing or resuming matched
  modes. A stopped job does not require rerunning the entire model.
- Runtime and source provenance are recorded. Moving files for storage is not
  permission to silently merge newly computed outputs from changed checkpoints,
  payload mappings or numerical settings.
- Resume is pinned to this source release. Changing source code or the scientific
  policy requires a new bank or an explicitly audited migration. The controller
  hotfixes accept exactly the known
  previous source fingerprints listed in `compatibility/embedding_bank_hotfixes.json`,
  and only when the dataset, complete registry and policy still match. It keeps
  `bank.json`, image records and PNGs unchanged and writes a separate audit receipt
  under `_reports/source_compatibility/`. It is not a general hash-check bypass.
  The FNNS/DistSeal loader fix additionally requires no existing model identity,
  image records or PNGs for these two methods before the first migration. Failed
  initialization logs are allowed. The receipt permits subsequent resumes of
  outputs created after the fix. Results from other methods are retained.
  Changing only
  the image count, storage paths or runtime does not change saved PNG identities.
  Checkpoint hashes are frozen per method; new runtime/library details remain
  separate execution records. Cross-hardware numerical equivalence is not assumed.
- Logs and the final report distinguish completed, failed, blocked and
  unreachable matched jobs. The expected denominator remains the requested
  allowlist × images × modes; missing prerequisites do not turn a partial bank
  into an all-32 success.
- Transfer the **entire bank directory**, including metadata, hidden files,
  keys, originals, messages and reports. The PNG folders alone are insufficient
  for reproducible extraction. Keep secret keys out of public research releases
  unless sharing them is an explicit part of the release design.

This stage ends with a reusable, audited input bank and an honest completeness
report. Channel-specific experiments consume these saved images independently;
one channel's attacked output is not the next channel's input.
