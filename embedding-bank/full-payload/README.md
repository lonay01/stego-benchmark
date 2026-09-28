# Full-payload embedding canary

This is a **new diagnostic release**, separate from the common30 embedding bank.
It checks all 32 catalogue entries on the same first **five historical COCO
images**, with **five full-length random messages per image**. Native and scalar
Matched Quality at 39, 42 and 45 dB are checked by decoding the saved PNGs.
There are no digital, neural or physical channel attacks in this notebook.

## Run in Colab

1. Download both files from `dist/`: `Embedding_Bank_FULL_PAYLOAD_CANARY.ipynb`
   and `Embedding_Bank_FULL_PAYLOAD_CANARY_source.zip`.
2. Open the notebook in Colab. Start with the A100/high-RAM runtime used for the
   previous diagnostic checks. One model runs at a time; fitting every upstream
   model into T4 VRAM is not guaranteed.
3. Leave `N_IMAGES = 5`, `MODELS = "all"`. Set `BANK_ROOT` to the old bank which
   contains `_inputs/manifest.json`, `_inputs/legacy_manifest.csv` and all 100
   original PNGs. Old encoded images and old messages are not reused.
4. Run the four code cells in order. Mount Drive and upload the matching ZIP
   when requested. There is no Base64 source in the notebook.
5. Read `_reports/REPORT.md` and send `full_payload_canary_report.zip` for review.

Default durable output: `MyDrive/watermarking_full_payload_canary_v1`.
Default temporary installations: `/content/watermarking_full_payload_scratch_v1`.
The old bank is read-only. Never point the new output directory at the old one.
No paid runtime is reserved or purchased automatically.

To resume, reconnect the GPU, reopen the same notebook/ZIP and run the cells
again with the same paths. Completed, hash-verified measurements are reused,
including imperfect recovery and failed scalar matching. Technical exceptions
can be retried on the identical message, with the previous error retained.
Temporary environments may need rebuilding after a Colab reset; saved results
on Drive do not. `MAX_HOURS` is a safety ceiling, not a guaranteed session length.

## What full capacity means here

The test uses every variable message position of the selected checkpoint and
interface. It does not retrain models, repeat a short message, add external ECC,
or pretend every configuration has the same payload. Thirty-bit checkpoints
still carry 30 bits: increasing their decoder size requires a different model.

| Special case | Tested contract |
|---|---|
| FNNS | 32 information bits in 48 positions; preserve 16 detection bits |
| InvisMark | 94 information bits in 100 positions; preserve the six registered reserved positions |
| Fang Screen | 30 information bits; retain native CRC6/BCH coding and report ECC rejection separately |
| DWT-DCT-SVD | 4096 raw bits at RGB512, Haar LL256, block4, one active chroma plane; replaces the old arbitrary 32-bit configuration in this canary only |
| SSL Watermarking | 2048 bits for the selected 2048-dimensional orthogonal carrier space; assert the loaded dimension and key shape; replaces the old 30-bit setting in this canary only |

DWT's limit follows the number of visited blocks in the
[author implementation](https://github.com/ShieldMnt/invisible-watermark/blob/main/imwatermark/dwtDctSvd.py).
SSL's carrier generator requires `k <= d` in the
[pinned author source](https://github.com/facebookresearch/ssl_watermarking/blob/3e65334bd28ecbc66756f0179def9dc4c88a6acc/utils.py).
These are **structural interface limits, not measured error-free capacities**.
A poor BER at the limit is an observation, not permission to shorten messages
until a desirable result appears. All other lengths come from the pinned
checkpoint registry, with runtime shape/decoded-length validation. Unknown
capacity stays unresolved. No claim about a global architectural maximum is made.

`_inputs/payload_contracts.json` records all lengths, message indices, reserved
positions, coding and bits per pixel. `SHAKE256(seed, COCO ID, replicate)` generates
a stable MSB-first stream. Equal-length methods receive identical messages;
shorter ones use a prefix, but **every information bit is scored**. The same
message is reused across modes. Selection of N or execution order never changes it.

## Checks and interpretation

- Embed once per image–message pair; save the Native RGB512 PNG and decoder keys.
- Decode the saved Native PNG. Expected messages are removed from the metadata
  passed to blind decoders; only shape/configuration/key information remains.
- Derive Matched Quality from the saved Native uint8 residual: one scalar alpha
  in `[0,4]`, 64 bisection iterations, rounding to nearest/ties-to-even, tolerance
  ±0.05 dB. No per-pixel correction or decoder-guided search. Verify PSNR again
  after saving/reloading. An out-of-tolerance candidate is not published as matched.
- Decode every available matched PNG; do not filter cases by Native success.
- Observe pre-serialization float decoding where the adapter exposes a compatible
  tensor boundary. File/PIL-only boundaries are explicitly unavailable, not fake
  "before PNG" tests obtained by reading the PNG back.
- For FIN and HiDDeN, repeat the author's-resolution / PNG / residual-to-512
  diagnostic chain for message replicate 1 on each image. DistSeal additionally
  has its autoencoder-only reconstruction control. These are labelled auxiliary
  diagnostics; they do not choose the bank's output or alter its quality budget.
- Score BER over all information positions and report reserved-bit errors
  separately. Native ECC rejection is a failed message, with no invented BER.
  A process crash or malformed bit string is a technical error, not BER=1.

This canary reports PSNR and recovery, not the final protocol's complete
SSIM/LPIPS, confidence-interval or attack evaluation. Exact recovery is sensitive
to payload length; do not rank unequal-capacity methods by exact recovery alone.

`native_exact_canary` means all 25 Native messages were recovered exactly and
the four mode tasks and required auxiliary diagnostics reached a terminal status.
It does **not** mean every matched
target exists or decodes perfectly. Inspect per-mode rows in `results.csv`.
The report retains denominators for all 32 methods even if prerequisites block some.

## Known blockers and dependency repairs

- Camera-Shooting Instance: no verified embedding/payload adapter in this release.
- Fang Screen: needs an existing licensed MATLAB plus Image Processing and
  Communications Toolboxes; normal Colab does not supply them.
- OmniGuard: needs a verified 100-bit copyright checkpoint. Localization weights
  are not substituted, and an unrelated public ZIP is not advertised as a fix.
- EditGuard and SepWater: reuse author weights from the old diagnostic folders
  when available, or download the previously audited releases. Verify hashes.
  EditGuard keeps the BasicSR compatibility import; SepWater pins setuptools
  with `pkg_resources`. These repair imports, not model arithmetic.
- ChunkySeal preparation allows three hours, including its large download.
  This is a ceiling, not a speed estimate. Successful model scratch is cleaned
  after its worker exits; failed scratch is retained for a same-runtime retry.

Every method has an isolated environment, preserved dependency lock, pinned
source, checkpoint inventory and execution log. The controller keeps only small
metadata in memory. It monitors RAM, local disk and wall time, with a 20-GiB
scratch reserve and a 150-GiB per-model cap. Logs and completed image records
are written to Drive, not held until all methods finish.
The controller stops before starting another model if free scratch falls below
30 GiB, rather than filling the disk with repeated failed installations.

## Files and provenance

```text
watermarking_full_payload_canary_v1/
  trial.json
  _inputs/                         # Exact original corpus + capacity contracts
  _assets/                         # Audited manually supplied/downloaded weights
  _reports/REPORT.md
  _reports/models.csv
  _reports/modes.csv                # Exact recovery / BER / failures by mode
  _reports/results.csv
  <method>/native/<COCO_ID>_m01.png
  <method>/matched_39/<COCO_ID>_m01.png
  <method>/matched_42/...
  <method>/matched_45/...
  <method>/_meta/records/           # Per-mode encoding and decoding seals
  <method>/_meta/attempts/          # Metadata, private keys, preserved attempts
  <method>/_meta/diagnostics/       # Resolution/serialization/autoencoder controls
  <method>/_meta/logs/
  <method>/_meta/environment/
```

At N=5 the upper bound is **3,200 marked PNGs**, plus small diagnostic artifacts.
Old production Python files, registry and notebook are unchanged. The new source
ZIP contains their exact snapshot plus the canary package and the existing
diagnostic stage helpers. The old common30 scorer is not used by this runner.

The source ZIP is content-pinned in the notebook. Model/checkpoint identities
and dependency locks are frozen on first use. Changed source, messages or payload
contracts require a new output directory, not weakened resume checks. Runtime
hardware/OS details are recorded separately; cross-hardware numeric equivalence
is not claimed. Keep the entire result tree when copying it to the Samsung SSD.

## Before the 100-image stage

Review all Native failures, unmatched targets and blocked models first. Do not
automatically launch 100 images when the queue reaches its end. The ID/seed design
supports a stable 5-to-100 prefix, but the production migration is a separate
decision after this canary. A changed adapter/configuration needs a new versioned
bank. Five successful images cannot guarantee error-free recovery on all 100.

## Developer validation

Run `python -m pytest embedding-bank/full-payload/tests -q` from the repository
root. The test fixture builds and unpacks the exact ZIP into a temporary directory
and adds its `src` to `PYTHONPATH`. Validate notebook schema with `nbformat` and AST
parse all code cells and bundled Python files. The local test suite checks IO,
full-message scoring, resumed work, corruption detection and scalar matching;
contract stubs are not evidence of pretrained-model quality. See `VALIDATION.md`
for the checks actually executed for this release.
