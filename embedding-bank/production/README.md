# Production embedding bank and clean decoding

This release combines the reviewed existing-image configurations in one
resumable notebook. It does not train models or run digital, neural, or physical
attacks. Do not use the old common30 notebook for this experiment.

## Run in Colab

1. Download both files in `dist`: `COCO_Embedding_Bank_PRODUCTION.ipynb` and
   `COCO_Embedding_Bank_PRODUCTION_source.zip`. Upload the notebook to Colab.
2. Select A100 with high RAM, the platform used for the author-path diagnostics.
   Keep at least 35 GiB free temporary SSD before setup. One model runs at a time;
   weights and dependencies are not installed together. H100 is not claimed as
   verified for the old RoSteALS PyTorch 1.11/CUDA 11.3 stack.
3. Leave `N_IMAGES = 5`, `MODELS = "all"`. Set `BANK_ROOT` to the old bank with
   `_inputs/manifest.json`, `_inputs/legacy_manifest.csv` and the 100 frozen PNGs.
   Nothing in that bank is modified. Missing originals cause an explicit error,
   not a download of a different COCO corpus.
4. Run all four code cells. Authorize Drive and upload the matching ZIP once.
   It is cached in the output directory for subsequent sessions. No other
   interactive prompts are expected during model execution.
5. Inspect `_reports/REPORT.md`, `clean_summary.csv`, `clean_results.csv` and
   `summary.json`. The final cell also creates `production_bank_report.zip`.
6. After reviewing the five-image test, change **only `N_IMAGES = 100`** and run
   the cells again. Keep the same notebook/source ZIP and output directory.
   Previously completed PNGs and clean decoding results are reused.

Default persistent output: `MyDrive/watermarking_production_bank_v1`.
Temporary scratch: `/content/wmbench_production_scratch`. A cloud runtime cannot
write to the Samsung SSD attached to your PC. Afterward, copy the whole bank to
`E:/watermarking_production_bank_v1`, including metadata, originals and keys.

The same release works on a Linux x86-64 NVIDIA server with CUDA-capable driver,
Git and `build-essential`; edit the three directory paths. No GPU is rented or
purchased by this notebook. Native Windows and Mac GPU execution of all 29
methods are not supported by this release. They can store and inspect results.

## Frozen configurations

Each method uses the full variable payload of its **selected configuration**,
not a claimed theoretical maximum. Five SHAKE256 messages per image use seed
20260921 and the same assignment rule as the native canary. Reserved positions
remain fixed; no short-message repetition or extra ECC is added. DWT uses 32
bits and SSL uses 30, not the earlier structural-limit experiment.

Small inputs are central pixel crops from the frozen RGB512 source, with no
external resize. Internal author preprocessing stays in the selected API.

| Carrier | Methods |
|---|---|
| 128 × 128 | FIN HEAVY, ARWGAN, PIMoG, SSRH, MBRS, CIN |
| 224 × 224 | SepWater, released checkpoint; no invented 256px weights |
| 256 × 256 | Meta Revisited HiDDeN, RoSteALS, VINE-B, VINE-R, MaskWM, SepMark |
| 400 × 400 | StegaStamp ONNX |
| 512 × 512 | TrustMark-Q/B/P, WAM, VideoSeal, PixelSeal, ChunkySeal, InvisMark, RivaGAN benchmark, DWT-DCT-SVD, SSL, Robust-Wide, EditGuard, FNNS OmniSeal, DistSeal post-hoc DC-AE |

Important decisions carried forward:

- Meta HiDDeN uses `checkpoint_256`, 48 bits, scale 0.3, no JND. The old third-party
  HiDDeN is not selected or renamed.
- RoSteALS uses the full author configuration and checkpoint from AUTHOR_FIX,
  including its corrected import roots and locked Python 3.9 environment. It
  retains author residual export, with the selected 256px crop as its carrier.
- SepWater uses the reviewed full author configuration at 224px, raw100 and its
  author PNG export. The separate BCH-text diagnostic is not the bank payload.
- DistSeal uses the audited author post-hoc DC-AE path, scale 0.5, 64 assigned
  random bits and author floor serialization. Reconstruction-only PNGs and
  source-to-reconstruction PSNR are stored as separate controls. No UViT
  generation, ImageNet download, or FID campaign runs here.
- EditGuard uses the previously selected copyright `degrade.pth` checkpoint.
  The diagnostic `clean.pth` is not selected as a test-set winner. BasicSR's
  dependency installation is constrained by the frozen environment lock.
- FIN retains its inverse INL decoder. FNNS retains 32 variable message bits
  alongside its internal detection pattern. InvisMark retains six reserved
  positions, with 94 variable bits scored.
- SepMark retains the reviewed serializer, not the best of floor/round/half-up.
  Nonzero clean BER remains a valid observation.

The author paths were previously tested on original-size COCO export as well.
This bank supplies the agreed central native-size crops and new common message
assignments; it does **not** copy those older measurements or promise identical
PSNR/BER. The integration of these paths is what the first five-image run checks.

## Modes and scoring

Embed once per image/message, save and reload the Native RGB8 PNG. Construct
matched candidates from this exact saved image using the existing scalar-only
policy: alpha in [0, 4], 64 bisection iterations, targets 39/42/45 dB and tolerance
0.05 dB. PSNR references the same unmarked crop. There is no pixel-wise correction
or decoder feedback. This specifically names the saved-uint8 residual boundary;
it is not matching an unsaved float encoder tensor.

An unmatched target produces a diagnostic record, not a falsely labelled PNG.
Matched images with decoding errors remain in the bank. Clean decoding runs
after carrier preparation for each method, before that model is unloaded. The
final cell aggregates all methods without loading them again.

Score complete variable messages. Save expected/decoded bits, error counts,
BER (fraction and percent explicitly labelled), exact recovery, embedding PSNR,
timings and coverage. All saved modes have their own clean result for future
paired attack comparisons. Reference messages are withheld from the decoder.
PSNR is source-to-marked, including any autoencoder reconstruction distortion.
SSIM/LPIPS, author-paper comparison tables and attack deltas are later reporting
stages, not measurements claimed by this bank builder.

## Output layout

```text
watermarking_production_bank_v1/
  trial.json
  _inputs/                 # 100 originals, manifest, crops, payload contracts
  _release/                # Exact source archive for reconnects
  _author/                 # Author weight/config audits and reusable downloads
  _assets/                 # Audited manual assets where needed
  _reports/                # REPORT.md, clean_results.csv, clean_summary.csv, summary.json
  trustmark_q/
    native/000000000139_m01.png
    matched_39/000000000139_m01.png
    matched_42/000000000139_m01.png
    matched_45/000000000139_m01.png
    _meta/                 # Assignments, sidecars, records, errors, progress, logs
  ...
```

Up to **580 × N** marked PNGs: 2,900 at N=5 or 58,000 at N=100, plus source,
crop and DistSeal control files. Actual disk use is measured by the canary;
source archives, model weights and environments are additional storage.

## Resume and interpretation

The entire 100-image manifest is frozen on first setup. N and storage locations
are not scientific identities; messages do not depend on execution order.
The source release, policies, models, checkpoints and resolved environment locks
remain fixed. A changed scientific configuration requires a new output root.
All newly computed executions record runtime details; moving to another GPU
does not imply bitwise numerical equivalence. Compare a fixed subset before
claiming cross-hardware equivalence.

Records are committed per image and mode. Hash conflicts stop reuse rather
than overwrite evidence. Decoder exceptions can be retried without re-embedding;
measured bit errors and matching misses are not retried for a better score.
Three consecutive technical failures stop that method; other methods continue.
Incomplete cases after interruption can be resumed. Do not run two controllers
against the same output folder.

`MAX_HOURS` is a session budget, not a time estimate or a workaround for Colab
limits. The notebook does not prevent Google from ending the runtime. Scratch
cleanup is restricted to owned model installations/caches; output PNGs, metadata
and author evidence remain on persistent storage. Reconnection may reinstall
unfinished model environments. Reporting reads small records, not all model
weights; artifact validation takes place on execution/resume. Reporting uses
one consolidated index per method, refreshed when that method finishes or stops.
It does not repeatedly scan all methods' thousands of small Drive files. During
an active method, consult `_meta/progress.json` and its log. A hard runtime loss
can leave the report index behind the durable task records; resuming rebuilds
it. To rebuild reports without inference, run the controller's `reindex` action
instead of `report`. This is the explicitly slower record scan, not a GPU task.

Completion, clean accuracy and target coverage are separate report fields.
Do not present a technically completed bank as error-free decoding or a completed
robustness benchmark. Never delete failed cases to make the summary look complete.

## Development

`python build_notebook.py` creates a deterministic source archive and notebook,
parsing every packaged Python module and code cell. `python validate_release.py`
checks the actual extracted package and exercises CPU contract/resume tests.
See `VALIDATION.md` for executed checks and remaining GPU verification.
