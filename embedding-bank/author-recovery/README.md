# RoSteALS and SepWater: targeted author-path recovery

This notebook continues the reviewed three-method author-path experiment without
rerunning DistSeal or changing the production embedding bank. It is not a new
training run and does not select messages, weights, serializers or image sizes
to improve scores on the test set.

## Run

1. Upload `dist/RoSteALS_SepWater_RECOVERY.ipynb` to Colab.
2. Select **A100 with high RAM**, matching the previous A100-SXM4-80GB execution.
   The recorded GPU/framework identity is checked; do not bypass a mismatch.
3. Run all cells. Authorize Google Drive, then upload the matching
   `RoSteALS_SepWater_RECOVERY_source.zip` when asked. The ZIP is cached on Drive.
4. Keep `PARENT_ROOT` and `OUTPUT_ROOT` separate. Default locations:

   - Read-only parent: `MyDrive/watermarking_author_reproduction_v1`
   - New results: `MyDrive/watermarking_rosteals_sepwater_recovery_v1`
   - Temporary environments: `/content/author_recovery_scratch_v1`

5. Allow at least 35 GiB free temporary SSD and approximately 10 GiB free Drive
   space as a planning allowance for copied SepWater artifacts and new RoSteALS
   artifacts. Weight/image sizes determine actual usage; these are not measurements.
6. `MAX_HOURS=8` limits a session, not the total time required. Reconnect and run
   the same notebook again to finish partial work. Do not run concurrent copies.
   Normal Colab session limits still apply; no keep-alive workaround is used.
7. Send `OUTPUT_ROOT/RoSteALS_SepWater_RECOVERY_report.zip` for analysis.

The first import streams and hashes saved artifacts; it can take several minutes
on Drive. Progress is printed every 25 records and the controller has a 30-second
heartbeat. GPU preflight failures appear in the report and block that model's
long stages. Failure of one method does not suppress the other's CPU/GPU checks.

## What changes

### RoSteALS

The old run stopped at `from slack import WebClient`. The recovery adds
`slackclient==2.9.3`, the version in the pinned author's
`dependencies/requirements.txt`. All previous RoSteALS dependency pins remain
fixed. No Slack token is requested and no notification is sent.

The full author configuration, pretrained inference tensors, 256px preprocessing,
100-bit decoder and original-resolution residual PNG export are unchanged.
The first COCO image is tested with five raw100 and five BCH messages. This also
executes the unmodified author CLI in a separate process and checks its pixels
and bit accuracy against the batch path. Missing/failed checks prevent the long
run. BER itself is not a preflight pass criterion.

Then all 100 original COCO images × 5 messages × 2 profiles are measured. The
author's 14 test-pool corruption operators are evaluated at all five severities
on the raw100 internal PNGs: 35,000 conditions. This expanded fixed grid is not
the paper's random severity sampling. Native FP32 inference only; no matched PSNR.

### SepWater

The completed 500 raw100 and 500 text-profile observations, their images and the
176 existing corruption grids are copied with checksum/provenance validation.
They are not silently relabelled as new inference results. Model execution and
package metadata are retained so further attacks cannot silently use a new stack.
Ten fixed replay cases (five images × raw100/BCH) must reproduce the old exported
pixels and decoded bits before the remaining GPU attacks run.

The released checkpoint/configuration remains **224×224**. This is not a claimed
reproduction of the paper's 256px experiment. The remaining author validation
pool uses nine operators × five severities × 500 raw100 cases = 22,500 conditions.
Attacks start from the same internal floor PNGs as before, not the original-size
residual export. Their seeds, strengths and operators are unchanged.

The BCH correction is a separate, explicitly labelled CPU postprocessing path:

1. Read the already recorded 100 predicted bits.
2. Remove only the four packet-padding bits; split seven data bytes and five ECC
   storage bytes, using BCH polynomial 137 and correction radius 5.
3. Call `bch.decode(data, ecc)` **and** `bch.correct(data, ecc)`.
4. Report exact 56-position data recovery, rejected packets and accepted-but-wrong
   packets separately. The true message is used only for scoring, never decoding.

The legacy author result, full-payload BER and zero-error 100-bit recovery remain
unchanged. Seven hexadecimal characters occupy 56 data positions but contain only
28 bits of entropy. This text diagnostic does not establish recovery of arbitrary
56 random useful bits and is not compared to the independent uniform raw100 cohort.
More than five errors can cause rejection or miscorrection; no success guarantee
is claimed outside the code's correction radius.

## Local verification

The corrected decoder is tested with 1,200 randomized 0–5-error cases plus 100
single-position tests. A read-only CPU replay on the actual prior report recovered
496/500 PNG text messages (legacy helper: 381/500); four packets were rejected.
For internal float decoding it recovered 498/500. This changes postprocessing,
not model BER, network weights or saved images. See `validation/sepwater_ecc_replay.json`.

CPU tests cover imports, evidence tampering, interrupted copies, ECC, incomplete
coverage and notebook/source packaging. Linux locks are resolved ahead of use.
Full end-to-end CUDA execution of this recovery is **not claimed as locally
tested**; runtime preflights establish that on the actual Colab environment.

## Reuse and integrity

The accepted parent identity is
`6b1ba1f5878b65736fce691053685c079300448ada3f1ea3120599bc758469d8`.
Its source ZIP has SHA256
`9f186b48fe81c1443b6db48d91a7c0b828b2ebe2b9246b732b59bb08ea0e8f61`.
Other runs are rejected. New record identities include the recovery source hash;
imported rows preserve the parent record hash/path/time. No scientific fingerprint
is bypassed or overwritten. Large model caches are copied only for the two methods;
DistSeal weights and generated pairs are not copied.

Code inherited from that exact source archive is bundled as readable `authornight/`.
Two batch-driver changes allow a separate worker entrypoint and a one-image smoke
limit; no upstream author inference script is edited. `authorrecovery/` implements
the explicit migration, BCH replay, preflights, queue, resumable attack grids and
report. Failed attack conditions are retried on the next run; completed conditions
are reused. All previous failed logs are retained, with latest phase status shown.

Scientific upstream references:

- [RoSteALS, pinned source](https://github.com/TuBui/RoSteALS/tree/8c09b022a362ffdf977a86f355e5af339793d9c9)
- [SepWater, pinned source](https://github.com/CVhnu/sepwater/tree/4b99a4d3a99f504a09e5e743cffe745b8856e36a)
- [BCH library](https://github.com/jkent/python-bchlib)

## DistSeal handoff

See `DISTSEAL_RESUME.md`. Setup also writes that instruction file to the new
Drive output directory. Do not run DistSeal through the recovery worker.
