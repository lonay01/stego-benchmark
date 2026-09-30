# DistSeal: saved state and continuation

Recorded from `Three_Method_AUTHOR_NIGHT_report.zip`, inspected 2026-09-30.
These are the last reviewed counts, not a live progress display.

## Where the experiment lives

- Google Drive: `MyDrive/watermarking_author_reproduction_v1`
- Colab path: `/content/drive/MyDrive/watermarking_author_reproduction_v1`
- Plan: `plan.json`
- Report: `_reports/report.md` and `_reports/summary.json`
- Small report archive: `Three_Method_AUTHOR_NIGHT_report.zip`
- COCO records: `distseal/coco/records/`
- Generated batch records: `distseal/generated/records/`
- Generation artifacts and attack records: under `distseal/`; each record lists
  its exact artifact paths and hashes. Do not rename or move individual files.
- Cached original source ZIP: `_release/Three_Method_AUTHOR_NIGHT_source.zip`
- Local original notebook:
  `D:\GIT\stego-benchmark\embedding-bank\author-night\dist\Three_Method_AUTHOR_NIGHT.ipynb`

Frozen identity:
`6b1ba1f5878b65736fce691053685c079300448ada3f1ea3120599bc758469d8`

Original source ZIP SHA256:
`9f186b48fe81c1443b6db48d91a7c0b828b2ebe2b9246b732b59bb08ea0e8f61`

## What is complete

| Work | Last reviewed coverage |
|---|---:|
| COCO native encode/PNG/decode | 500/500 (100 images × 5 messages) |
| COCO attack conditions | 43,500/43,500 |
| Generated pairs | 15,156/50,000 |
| Generated images with complete 87-condition grids | 2,964/50,000 |
| Generated distortion evaluations | 257,868/4,350,000 |
| Inception Score | measured on the partial 15,156-image set |
| FID against ImageNet validation | unavailable; dataset not supplied |

COCO PNG BER: 0.20625%; exact 442/500; source-to-PNG PSNR 24.35254 dB.
COCO source-to-unmarked-AE PSNR: 24.93458 dB (reconstruction control).
Generated PNG BER: 0.209076%; exact 13,383/15,156; paired PSNR 31.16076 dB.
IS marked: 96.35082; unmarked: 97.64151, both partial n=15,156.
Do not present these as completed 50k measurements or all variants of the paper.

## How to resume only DistSeal later

1. Open the **original** `Three_Method_AUTHOR_NIGHT.ipynb`, not the two-method
   recovery notebook. Select the same A100-SXM4-80GB/high-RAM environment.
2. Keep these scientific settings unchanged:
   `N_COCO=100`, `GEN_PER_CLASS=50`, `GEN_BATCH_SIZE=4`, original output path,
   original source ZIP. `MAX_HOURS` may be changed (for example, 8).
3. Run the original settings and setup cells. The correct source ZIP is cached
   on Drive. If prompted, supply the original 3-method ZIP with the SHA above.
   Do not supply the recovery ZIP, replace plan.json, or disable identity checks.
4. Replace **only the queue cell** with the following. This changes scheduling,
   not model settings or the frozen scientific plan:

```python
import authornight.runner as queue
from authornight.report import report_bundle

queue.STAGES = [
    ("distseal", "generation", 9999, {}),
    ("distseal", "attacks", 45, {"cohort": "generated"}),
    ("distseal", "inception", 30, {}),
]
try:
    SUMMARY = queue.run(SETTINGS, EXPECTED_SHA)
finally:
    REPORT_ZIP = report_bundle(OUTPUT_ROOT)
    print("Saved report:", REPORT_ZIP)
```

The old runner reserves 75 minutes for attacks/IS. Completed batches and grids
are verified and reused. No other model is scheduled, and finished COCO work is
not rerun. Generation may finish before the time limit; that is normal.

5. Once generation reaches 50,000, spend the next sessions on remaining attacks
   by replacing `queue.STAGES` above with:

```python
queue.STAGES = [
    ("distseal", "attacks", max(1, int(MAX_HOURS * 60) - 30), {"cohort": "generated"}),
    ("distseal", "inception", 30, {}),
]
```

6. Run the original final report cell and check completed/planned counts. An exit
   code of zero alone is not proof that 50k generation or all 4.35 million tests
   finished. Recompute/report IS on the final full cohort, not the earlier prefix.

Allow storage for the full target: the original protocol budgeted about 120 GiB
free Drive capacity for paired images and artifacts (planning allowance, not a
measured size). If the runtime changes, restore the recorded GPU/framework rather
than bypassing provenance checks. Never run two sessions writing this same folder.

The RoSteALS/SepWater recovery writes elsewhere and does not modify this run.
