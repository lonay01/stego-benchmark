# RoSteALS, SepWater and DistSeal: author-path overnight check

This experiment asks a narrow question: do the released checkpoints behave
correctly when used through their authors' inference paths? It does not change
the embedding bank, retrain a model, shorten messages or select a configuration
because it happens to decode better on these images.

## Run in Colab

1. Upload `Three_Method_AUTHOR_NIGHT.ipynb` to Colab. Select **A100 and high RAM**.
   Keep the same GPU type when resuming this output folder.
2. Run the settings and setup cells. Authorize Google Drive. Upload
   `Three_Method_AUTHOR_NIGHT_source.zip` when requested. This is readable Python,
   not embedded Base64. The ZIP is cached on Drive for the next session.
3. Check that `BANK_ROOT` points to `watermarking_native_resolution_canary_v1`.
   Only its image-ID manifest is read. The selected original COCO JPEGs are
   downloaded into the **new** output folder; the old 512px crops are not reused.
4. Run the queue cell, then the report cell. `MAX_HOURS = 8` is a wall-clock
   budget from queue start, including setup; it is **not an ETA**. Do not run
   another copy against the same output directory.
5. Before leaving, wait for Drive authorization, source upload and the first
   saved results. Keep the browser/computer awake if required by your Colab plan.
   No keep-alive bypass is included. Colab can still terminate the runtime.

The complete dependency stacks are isolated: Python 3.9 / PyTorch 1.11+cu113 for
RoSteALS, Python 3.10 / PyTorch 2.0.1 for SepWater, and Python 3.10 / PyTorch 2.7.1
for DistSeal. Only one model worker uses the GPU at a time. All stacks fit on an
ordinary large Colab temporary disk; allow at least 60 GiB free before setup.
The persistent cache holds source snapshots and weights, not virtual environments.
Allow roughly **120 GiB free on Drive** if you intend to finish the 50,000 paired
DistSeal images plus the other artifacts. This is storage planning, not a measured
archive size. The first eight hours may produce much less.

The report is written to:

```
MyDrive/watermarking_author_reproduction_v1/_reports/report.md
MyDrive/watermarking_author_reproduction_v1/_reports/summary.json
MyDrive/watermarking_author_reproduction_v1/Three_Method_AUTHOR_NIGHT_report.zip
```

Send the last ZIP for analysis. The notebook can rebuild it from a partial run.

## Fixed experiment

The default cohort is the first 100 IDs in the existing COCO manifest. Five
independently seeded messages are used per image. This is a new experiment:
original JPEG geometry and newly declared messages differ from the old cropped
canary. Its results must not overwrite, or be presented as paired replays of,
those earlier observations.

| Path | Input to the network | Message | Planned COCO cases |
|---|---|---|---:|
| RoSteALS raw | Author resize to 256×256 | Full random 100 bits | 500 |
| RoSteALS text | Same | Seven ASCII characters, author BCH encoding to 100 bits | 500 |
| SepWater raw | Released author resize to 224×224 | Full random 100 bits | 500 |
| SepWater text | Same | Seven ASCII characters, author BCH encoding to 100 bits | 500 |
| DistSeal post-hoc DC-AE | Author notebook resize to 512×512 | Full random 64 bits | 500 |

### RoSteALS

The network, full configuration, checkpoint, BCH helper and export expressions
come from [the author repository](https://github.com/TuBui/RoSteALS).
The primary exported PNG uses the author's residual-upscaling path: compute the
marked image internally at 256px, resize its residual to the original COCO size,
add it to the original RGB image, clip, and convert to uint8.

The batch loop calls the author's encoder, latent control and decoder directly.
For five fixed text cases it also runs **the unmodified `inference.py` in a fresh
process** and compares both exported pixels and reported raw bit accuracy.
The source script is unchanged; a `runpy` launcher applies the same FP32,
TF32-disabled, deterministic-cuDNN policy as the batch worker. Otherwise older
PyTorch defaults could create a numerical mismatch unrelated to the adapter.
The CLI decodes the internal floating-point image. We additionally decode the
saved, reloaded original-resolution PNG using the author's input transform.
These are separate columns, not interchangeable results.

The released evaluation code's three comparisons are retained: source → AE,
source → marked, and AE → marked. We report PSNR, SSIM, LPIPS and SIFID, with
internal-size and original-resolution export measurements separated. RoSteALS'
published word accuracy uses a BER threshold; zero-error message recovery is a
stricter metric. Both are stored and clearly named.

The extended corruption sweep uses the author's 14 test-pool operators at all
five severities: 35,000 tests. This expands the released evaluator's random
severity draw and starts from saved internal PNGs, not raw tensors. It is
**not an exact replication of the paper's corruption distribution**. The known
ECC branch in the released evaluator decodes `stego` instead of its perturbed
copy; this notebook does not use that branch as evidence of noisy ECC recovery.

### SepWater

The [released repository](https://github.com/CVhnu/sepwater) and
[published checkpoint folder](https://drive.google.com/drive/folders/1NC5q8BsrUntZ6y_5o267GXnaowoNXp90)
provide an inference configuration at 224px. Its secret-control weights produce
56×56 latent residuals. A 256px carrier needs 64×64 latents. Changing only the
input size is therefore not a valid 256px checkpoint.

No verified matching 256px watermark checkpoint/configuration pair was located.
The other published `f4.ckpt` is downloaded and its tensor groups audited; it is
not silently substituted. The [paper](https://doi.org/10.1109/TCSVT.2026.3670224)
describes a 256px setup. Its reported quality is not a 224px checkpoint reference;
no corresponding published 224px result was established.

The full released configuration and `last.ckpt` are used without deleting the
loss/noise modules. Export follows `embed.py` exactly, including float64 addition
of the original RGB image and the final uint8 cast. Five cases are independently
checked through unmodified `embed.py` **and** `extract.py`. The latter reports
the 56 text-prefix bits, whereas this report also measures the complete 100-bit
network payload. Do not equate these two metrics.
Text messages are seven deterministic hexadecimal characters (56 encoded data
positions, not 56 bits of entropy). The independent raw100 cohort supplies
uniform full-length random payloads; the text cohort tests author CLI/ECC behavior.

The released ECC helper is tested with 0, 1, 3, 5 and 6 injected data-bit errors.
It is kept unchanged even if error correction fails; full payload BER remains
available. Its modern `bch.decode` call is not the same as applying `bch.correct`.
The extended synthetic noise sweep uses all nine released validation operators
at five severities: 22,500 tests. These are not real screen-camera photographs.

### DistSeal on COCO

This follows the author's [`posthoc_dcae.ipynb`](https://github.com/facebookresearch/distseal/blob/d0a71cd272bb29e3b49cdc965bbaf4d24f2684ac/notebooks/posthoc_dcae.ipynb):
the released diffusers DC-AE, latent embedder, scaling 0.5, and 64-bit detector.
Each source is encoded once per message; unmarked and marked reconstructions
come from the same latent. Source → AE quantifies reconstruction loss independently
of the watermark. The primary PNG uses the floor conversion in the author's
`save_img`; round-to-even and half-up are diagnostics, never selected winners.

One checkpoint loader compatibility issue is handled explicitly: if
`args.attenuation=None` fails at `.lower()`, only that value is normalized to
`"none"`. The original error, change, checkpoint hash and exact loaded inference
tensor checks are recorded. No architecture or learned weight is changed.

### DistSeal on author-domain generated images

The separate generation path uses the released **base UViT-H and native DC-AE**,
512px output, 30 DPM-Solver steps, no classifier-free guidance (scale 1), and
latent scale 0.2889. The watermark is inserted into the sampled latent before
decoding; generated RGB is never re-encoded. Each saved pair shares the exact
same latent, with and without watermark. The released fixed 64-bit signature
is used, as in the author's evaluation path.

The target is 50 images for each of 1,000 ImageNet classes: **50,000 pairs**.
Class-conditional generation does not require downloading ImageNet photographs.
Balanced prefixes visit every class before starting the next repetition.
Independent batch seeds enable resumption; this is not bit-identical to the
author's eight-GPU RNG order. Generation is batched; AE decoding is one image at
a time to bound memory. The author's generation serializer is retained exactly:
`clamp(127.5 * signed_image + 128, 0, 255).to(uint8)`.

The released `scripts/attacks.yaml` contains 86 parameterized distortions; with
identity there are 87 conditions: flip, rotation, resize, crop, perspective,
brightness, contrast, saturation, grayscale, hue, JPEG, blur and the combined
JPEG/crop/brightness attack. They are evaluated on both COCO and generated PNGs.
Operators are loaded from a pinned OmniSealBench checkout without its unrelated
audio/video runners. The composite image attack's class bodies are unchanged.
Random geometry is seeded per sample and condition. Category-wise BER and macro
averages are reported with observed coverage, not silently filled missing values.

**ImageNet validation is not available, so FID against ImageNet is unavailable.**
COCO is not substituted. Inception Score is computed separately for marked and
unmarked generated PNGs using [torch-fidelity 0.3.0](https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html)
and ten splits, once at least
1,000 samples exist and time remains. It needs no real-image reference. The
sample count and partial-coverage flag accompany every value; an IS on a partial
prefix is not compared numerically as if it were the paper's 50k evaluation.
It targets the post-hoc DC-AE branch only, not the paper's RAR, distillation or
training experiments. It must not be called a complete numerical reproduction
of every table in [the DistSeal paper](https://arxiv.org/abs/2601.16140).

## What happens during the night

The three COCO paths have priority. Then the queue alternates an initial balanced
generation prefix and the corruption checks. Remaining time goes to generation,
with a final decoding window reserved. Individual stages have time ceilings so
one slow method cannot consume the entire session. Setup has a 30-minute ceiling
per environment. A stage can be partial even if its process exits normally.

The controller prints a heartbeat every 30 seconds: stage, elapsed time, free
temporary SSD and GPU use. Full subprocess output is written immediately to Drive.
It stores one record/artifact archive per COCO image/message/profile, and per
small generated batch. An interrupted incomplete case is repeated; completed,
hash-verified cases are not. No historical result or model weight is deleted.

Run the same notebook again to continue. Keep `N_COCO`, `GEN_PER_CLASS`, batch
size, source ZIP, output directory and GPU type unchanged. `MAX_HOURS` may change.
Scientific setting changes require a new output folder. A checksum or identity
mismatch stops reuse instead of combining incompatible measurements.

The full 50k generation and its 4.35 million distortion evaluations are **not
promised to finish in eight hours**. The report gives completed/planned cases,
technical failures and unfinished work separately. For a bounded pilot choose
`GEN_PER_CLASS=1` (1,000 generated images) before the first run, in a different
output directory if a 50-per-class run already exists.

## Verification and repository layout

```
authornight/          controller, author-path drivers, metrics, resume, report
locks/               resolved Linux dependency sets for all three environments
reference/           small pinned upstream excerpts used in CPU contract tests
tests/               static, synthetic, quantization, resume and attack tests
validation/          local test evidence; no fabricated model results
build_notebook.py    deterministic readable source ZIP + notebook builder
```

The release has CPU unit/smoke tests and Linux dependency-resolution checks.
`validation/sepwater_cpu.json` audits the actual pretrained secret-control weights.
These checks **do not establish successful end-to-end CUDA inference**, runtime,
model BER, or agreement with papers. Those must come from the saved overnight run.
Exact GPU/Python/package/source/checkpoint identities are recorded with the results.
The external author checkpoints use PyTorch serialization; load only the declared
official assets, never untrusted replacement files.

Source revisions are pinned in `authornight/spec.py`. RoSteALS and DistSeal
reference excerpts retain upstream copyright/license files. OmniSealBench image
operators are under its separate license, included in `reference/omni/LICENSE.md`.
The source bundle is not a relicensing or redistribution of pretrained weights.
