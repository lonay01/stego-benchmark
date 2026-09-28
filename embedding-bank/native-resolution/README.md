# Native-resolution embedding canary

Check whether the selected implementations can embed and recover their complete
useful messages at their declared carrier resolutions. This is an integration
and clean-channel test, not a robustness benchmark or an architecture ranking.

## Run in Colab

1. Upload `dist/Embedding_Bank_NATIVE_RESOLUTION_CANARY.ipynb` to Colab.
2. Select a GPU runtime. A100 + high system RAM matches the previous test setup.
3. Keep `N_IMAGES = 5` and `MODELS = "all"`. Set `BANK_ROOT` to the existing bank
   containing `_inputs/manifest.json`, `legacy_manifest.csv` and its 100 originals.
4. Run the four code cells. Mount Drive, then upload the matching
   `Embedding_Bank_NATIVE_RESOLUTION_CANARY_source.zip` when asked.
5. Inspect `_reports/REPORT.md` and `models.csv`. Send
   `native_resolution_canary_report.zip` for review. It contains records/logs,
   not model weights or the image bank.

The default result location is
`MyDrive/watermarking_native_resolution_canary_v1`. The old bank is read-only.
Five images and five messages per image give 25 trials per method: 725 marked
PNGs for the 29 runnable configurations. Three exclusions are reported separately.

## What is fixed

- **Sources:** the same first five IDs of the frozen historical COCO100 corpus.
  COCO is not inherently 512x512; our historical preprocessed PNGs are.
- **Carrier:** exact central pixel crop of each frozen RGB512 image. No resizing
  or resampling in this step. For even size S the box is
  `((512-S)/2, (512-S)/2, (512+S)/2, (512+S)/2)`.
- **Message:** five reproducible random full-length useful messages per image.
  SHAKE256, seed 20260921; the length-independent stream is shared across models.
  Required reserved/detection positions are preserved, and are not counted as
  useful information. No common30 subset, repetitions, or additional ECC.
- **Embedding:** Native strength for the frozen selected implementation. No
  external alpha, matched PSNR, attack, or residual upscaling to a 512 carrier.
  Official internal resizing, channel order, normalizations and thresholds stay.
- **Checkpoint:** the same source revisions and weights as the reviewed bank.
  Dependencies are isolated per method. Downloaded weight hashes, installed
  dependency identities, geometry observations and runtime details are saved.

"Default" means the **selected checkpoint/configuration**, not every variant of
an architecture. DWT and SSL are configurable: this trial uses the original
selected settings of **32** and **30** useful bits. The preceding 4096/2048 trial
tested structural limits; it must not be mixed with this trial.

## Carriers

| Carrier | Selected methods |
|---|---|
| 128x128 | FIN (HEAVY), HiDDeN benchmark, ARWGAN, PIMoG, SSRH, MBRS (30-bit checkpoint), CIN |
| 224x224 | SepWater |
| 256x256 | RoSteALS, VINE-B, VINE-R, MaskWM, SepMark |
| 400x400 | StegaStamp ONNX benchmark |
| 512x512 | TrustMark-Q/B/P, WAM, VideoSeal, PixelSeal, ChunkySeal, InvisMark, RivaGAN benchmark, DWT-DCT-SVD, SSL Watermarking, Robust-Wide, EditGuard, FNNS OmniSeal, DistSeal post-hoc DC-AE |

512 is the selected carrier for flexible APIs, **not** an assertion that their
encoder was trained at 512. Internal model resizing remains in those APIs. For
example, a wrapper may embed a small residual back into an arbitrary carrier as
part of the author's method. We remove only our external small-to-512 transfer.
Runtime geometry fields, where exposed, are checked before the first image;
every emitted PNG is checked against the declared dimensions.

Excluded: **OmniGuard** (no verified matching copyright checkpoint), **Fang
Screen** (licensed MATLAB/toolboxes), **Camera-Shooting Instance** (unverified
adapter/checkpoint). They are not scored as decoding failures or counted as passes.

## Boundaries and measurements

For each image/message pair:

1. Validate source hash; freeze the central crop, crop box and output hash.
2. Embed the complete payload once using the selected checkpoint.
3. If a compatible image-tensor boundary is exposed, decode the encoder/wrapper
   return before our PNG serialization. The reference message is withheld.
4. Save RGB8 PNG, reload it, and blind-decode it with the same model and keys.
5. Compare the complete useful message: BER, bit accuracy and exact recovery.
   Raw-interface BER and reserved-bit errors are recorded separately.
6. Record PSNR of the saved marked PNG against the same unmarked crop, embedding
   and decoding time, payload size, bits/pixel, technical errors and provenance.

The saved PNG check is mandatory for every runnable method. Before-PNG checks
are observations of the exposed boundary: some author wrappers quantize
internally already. A file-only API is marked `unavailable`, never counted as a
float-before-save pass. A failed observer is reported separately from PNG BER.
FIN keeps INL in its decoder; StegaStamp uses the audited raw100 ONNX graph,
not an unverified TensorFlow replacement. Their native bridges have no external
residual lifting. Other algorithms and thresholds are unchanged.

## Reading the table

`png_exact_canary` means all 25 PNG messages decoded exactly, with no before-PNG
diagnostic error. `completed_with_bit_errors_or_rejections` means the code ran
but some useful information was lost. `error`, `blocked`, `pending` and
`excluded` are different states. Do not interpret a finished cell as all models
passing. Review per-image errors as well as averages; five messages from the
same crop are not five independent images.

Do not silently shorten messages, change weights, search for favourable crops,
or retry measured bit errors to improve this table. Different payloads and
carrier sizes compare configured systems; they do not identify architectural
causes. Crops from one COCO parent also contain different amounts of scene
content. Architecture claims require controlled follow-up experiments.

## Resume, disk use and isolation

After a disconnect, run cells 1–3 with the **same source ZIP and paths**. Valid
images and decoding measurements are reused. Only technical failures retry;
their old records are archived. Corrupt or changed sealed artifacts stop the
run instead of being silently accepted or replaced. A source/policy/weight
identity change requires a new result directory. Increasing N from 5 to 100
under the same frozen identity extends the stable image prefix, after review.

The controller runs in a small Python3.10 environment. Model environments are
installed and used **one at a time**, in subprocesses. Successful scratch
environments, downloads and caches are removed only inside an ownership-checked
slot. Durable results/locks are not removed. Failed slots stay available for
inspection. The runner stops before another model when <30 GiB SSD is free;
the process guard also watches remaining disk/RAM. It does not reserve cloud
resources or keep Colab alive artificially.

Source ZIP, originals, reports and environment locks survive on Drive; local
venvs and weights may need to be downloaded again after runtime deletion.
The report is built from small JSON records without model initialization.

## Layout

```text
watermarking_native_resolution_canary_v1/
  trial.json
  _inputs/{manifest.json,contracts.json,images/,crops/<size>/}
  <method>/
    native/<coco_id>_m01.png
    _meta/{records/,attempts/,logs/,environment/,executions/}
  _reports/{REPORT.md,models.csv,results.csv,summary.json}
```

## Development

From this directory, run `python build_notebook.py` and `python -m pytest tests`.
Tests load the exact bundled source ZIP, not a different development tree.
`smoke_cpu.py` runs real FIN or DWT locally on a clearly labelled author-example
input; it is **not** the historical COCO canary. See `VALIDATION.md` for evidence.
The existing `../full-payload` modules provide shared storage, dependency repair,
asset provenance and process management; deployment bundles those dependencies.
The production bank and earlier full-payload notebook are not modified.
