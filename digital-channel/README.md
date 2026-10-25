# Digital channel: saved carriers, temporary attacks

This notebook consumes the completed 29-method **production** embedding bank.
It does not embed messages, regenerate missing carriers, change payload lengths,
change native geometry, retune alpha or repair the bank. The bank is read-only.
The original production source ZIP is shipped byte-for-byte internally, with its
file inventory checked before execution. Only the already reviewed Meta source
loader is reused; FNNS carrier-repair hooks are **not installed** in this runner.

## Start in Colab

1. Download both files from `dist`: `COCO_Digital_Channel_RESUMABLE.ipynb` and
   `COCO_Digital_Channel_RESUMABLE_source.zip`.
2. Upload the notebook to Colab. Select an NVIDIA GPU and high RAM. The reviewed
   production bank used an A100; retaining that runtime is the lowest-risk first
   test. A different GPU is not silently assumed numerically equivalent: fresh
   Clean outputs must equal the saved controls. No claim that all 29 were tested
   on a T4/H100 is made here.
3. Keep `N_IMAGES = 5`, `MODELS = "all"`. Check `BANK_ROOT` points to
   `MyDrive/watermarking_production_bank_v1`. It needs the **whole** bank, including
   `trial.json`, `_inputs`, per-method `_meta`, and `_author/_meta`, not only PNGs.
4. Run the cells in order. Upload the source ZIP when asked. A hash-checked copy
   stays in the result folder; reconnecting does not require uploading it again.
5. Review `_reports/REPORT.md`, coverage and per-method tables. Only then change
   `N_IMAGES` to `100` and rerun with the **same source ZIP and output directory**.
   Existing image/message/attack assignments and measurements remain identical.

For a Linux NVIDIA server, place the notebook and ZIP together, mount/copy the
whole bank to persistent storage and change `BASE`/`SCRATCH_ROOT` in the settings
cell. `/workspace/data` is only a default path, not proof of persistent storage.
Native Windows is supported for offline tests, not for the 29-model GPU runner.
Do not start two runners against the same output. Colab disconnects and paid
runtime limits still apply; this notebook does not bypass them.

## Storage and execution

One method at a time:

1. Verify its bank records, full message assignments, PNG/crop/key hashes and
   native geometry. Unmatched PSNR targets stay in coverage; no image is invented.
2. Generate all **pending** attacked carriers for that method, in ZIP packages
   containing `method/mode/family/condition/carrier.png`. Each ZIP holds the 78
   attacked variants of one saved carrier and an internal manifest. ZIP uses
   STORE: PNG is already compressed. All attacks start from the saved carrier.
3. Finish and verify this saving stage before loading a decoder. Prefer local
   SSD if a conservative full-method estimate plus a 60-GiB environment reserve
   fits. Otherwise use `OUTPUT_ROOT/_temporary`. Later packages can fall back to
   Drive if local free space falls. Drive quota is **not** inferred from the
   mounted filesystem's free-space number: leave enough quota or the write will
   fail safely. At 100 images a 512-pixel method has at most 156,000 attacked PNGs;
   a conservative uncompressed-size bound is about 120 GiB. Actual PNG sizes vary.
4. Measure source-crop-to-carrier PSNR/SSIM/LPIPS in a separate metrics process.
   Then prepare the method's original isolated Python environment, compare its
   checkpoint/source/dependency identity with the bank, and decode saved PNGs.
   Clean is replayed and compared to the bank before each pending carrier's attacks.
5. Save full expected/recovered messages, statuses, error positions, parameters,
   pixel hashes, execution metadata and timings to Drive. Checkpoint every carrier,
   every error, and after a decode when 60 seconds have elapsed since the previous
   checkpoint. Progress is printed every ten attacks. This avoids writing a small
   Drive file after every fast inference. Valid measurements, even high BER, are reused.
6. Read back complete result records, write tables and a commit receipt, and
   delete **only verified, owned temporary attack ZIPs**. The original bank,
   messages, records, logs and manifests are not deleted. Model subprocess exit
   releases VRAM; owned model installations are cleared between methods.

An interrupted method keeps its packages. A lost Colab SSD loses only temporary
data; deterministic regeneration must reproduce existing package hashes. An
unexpected evidence change stops the run rather than silently overwriting it.
The first failing method stops scheduling so failed caches do not accumulate on
Drive. Rerun after resolving a technical failure: already measured attacks are
skipped, and previous failed attempts remain recorded. Clean discrepancies stop
that method and appear under `clean-drift/`; never treat them as channel damage.

Results still need persistent disk space (full bit strings are retained). The
bulk attacked PNGs do not accumulate across methods. Author checkpoint caches
for RoSteALS/SepWater/DistSeal may remain under `_runtime/_author`; they are not
the attack-image bank and are not deleted by attack cleanup.

## Frozen digital protocol

The exact repository protocol and baseline policy are bundled under `reference/`.
The local repository copy was inspected; GitHub DNS/authentication prevented a
fresh remote verification during construction. Their content hashes are recorded.
This implements Native and matched 39/42/45 dB **as already saved**, five full
messages per image, with each method's reviewed native crop dimensions.

| Family | Levels | Repeats |
|---|---|---:|
| Clean | Existing saved PNG | 1 |
| JPEG | Q 90, 70, 50, 30 | 1 |
| Gaussian noise | Sigma .005, .01, .02, .04 in [0,1] | 5 |
| Gaussian blur | Sigma .5, 1, 1.5, 2 pixels | 1 |
| Brightness | .70, .85, 1.15, 1.30 | 1 |
| Crop and restore | Retained area .90, .75, .50 | 5 |
| Resize and restore | Side scale .75, .50, .25 | 1 |
| Rotation | −15, −10, −5, −2, +2, +5, +10, +15 degrees | 1 |
| Compound | The four protocol levels, unchanged | 5 |

Total: **79 conditions**, 78 new PNGs per available carrier. Compound order is
brightness → rotation → resize/restore → Gaussian noise → JPEG.

Implementation choices not specified numerically in the prose are fixed in
`spec.py`: bicubic resampling; fixed rotation canvas with black fill; Gaussian
blur with reflect boundary and truncate 3; clip and round-to-even RGB8 after
each operation; JPEG 4:2:0, no optimize/progressive. JPEG is decoded and stored
losslessly as PNG for identical file-based interfaces. Crop uses sqrt(area) on
each side, rounding dimensions; actual area and crop box are logged. No attack
output is PSNR-matched again.

Seeds depend on image ID, message replicate and attack condition, **not** method,
mode, N or processing order. Noise uses a shared 512×512×3 PCG64 float64 field,
centrally cropped at native size without resizing, preserving sigma and shared
pixel noise where geometries overlap. Crop location and compound parameters are
shared as normalized variables; integer crop boxes naturally vary with geometry.

## Tables

`<method>/records/*.json.gz` is the durable, auditable source of truth.
`<method>/packages/*.json` retains every attack parameter and saved PNG checksum
even after its disposable ZIP is gone. Per-method `tables/` contains:

- `per_attack.csv.gz`: every planned eligible condition, including missing/error/
  rejected rows, BER, exact recovery, extraction time and paired clean changes.
- `attack_summary.csv`: separate severity and mode; repeats then messages within
  each image, then equal image weight; paired image bootstrap, 2,000 resamples.
- `macro_summary.csv`: eight families with equal weights, levels equal within
  family; Clean excluded. Macro metrics are withheld per image when their
  required observations are incomplete, and contributing image counts are shown.
- `reference_and_clean.csv`: full capacity/interface, actual crop/bpp, saved
  quality, clean decoding, matching coverage, and author-reference availability.
  Published numbers are **not verified** in this source bundle and remain so,
  rather than inventing a paper PSNR/BER or claiming author reproduction.
- `_reports/pairwise_matched.csv`: common image/message pairs meeting the target
  for both methods and fully decoded repeats, with sample sizes and image-cluster
  intervals. This is explicitly a surviving-subset comparison; retain the coverage
  table alongside it. Different capacities/interfaces are not an equal-payload ranking.

BER/exact values in CSV are percentages; delta BER and exact loss are percentage
points. JSON BER remains a fraction, matching the production bank. Paired deltas
use the same mode/carrier's clean result, not Native for all targets and not a
paper number. Clean errors and negative deltas are retained. ECC rejection means
unsuccessful exact recovery; no returned complete message means unavailable BER.
Technical errors are not assigned fake BER. Five images are a canary, not a
statistically persuasive robustness experiment. CI estimation at N=5 is limited.

## Development

Build: `python build_notebook.py`. Offline/CPU checks:
`python validate_release.py`. The validator unpacks the exact distributed source
to a temporary directory, parses every Python module and notebook code cell,
validates notebook structure, and runs tests. Optional real Meta checkpoint smoke
uses `WM_DIGITAL_META_FIXTURE` pointing to a directory with `models.py`,
`attenuations.py`, `hidden_replicate.pth` and `demo.png`. No weights are shipped.
See `VALIDATION.md` for what was actually executed and what still requires Colab.
