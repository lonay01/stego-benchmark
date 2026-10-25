# Digital channel: bounded parallel decoding and fast resume

This is an execution overlay on the reviewed digital-channel release. It does
not replace adapters, weights, payloads, geometry, attack parameters, seeds,
PNG serialization, scoring, aggregation or bootstrap rules. The original ZIP
is copied byte-for-byte into the distribution. Its release identity is retained;
the additional execution code has a separate SHA-256 inventory.

## Start in Colab

Download both files from `digital-channel/dist`:

- `COCO_Digital_Channel_PARALLEL_FAST_RESUME.ipynb`
- `COCO_Digital_Channel_PARALLEL_FAST_RESUME_source.zip`

Stop the old runner **before** starting this one. Do not run two notebooks
against the same output. The current cloud process is not changed by this release.
Use the same GPU type and the same `BANK_ROOT`, `OUTPUT_ROOT`, `SCRATCH_ROOT` as
before. The archive is uploaded once and retained in `OUTPUT_ROOT/_release`.
`N_IMAGES = 100` continues the current full prefix; use 5 for a canary. Increasing
from 5 to 100 requires validation of the larger bank catalogue; it does not
re-embed images or discard the earlier digital results.

Start with `DECODE_WORKERS = 2`, `ATTACK_WORKERS = 2`, `FULL_AUDIT = False`.
On A100 80 GB, try four decode workers **only after** checking two-worker logs
and throughput. Changing concurrency does not change which cases are scored.
One model environment is installed at a time. Each worker loads its own copy
of that model; memory is not shared. StegaStamp remains serial and uses its
existing fixed-CUDA/paired-Clean repair, unchanged.

Workers are admitted one at a time after warm-up, using measured per-process
GPU/RAM and remaining memory with headroom. Unknown GPU memory means sequential
execution. Warm-up is not a worst-case memory proof: a later OOM stops the run,
keeps evidence, and must be resumed with fewer workers. CPU attack creation is
parallelized across cases. All packages for the method are committed before
decoding starts, as in the existing experiment.

The old VideoSeal telemetry examined for this change (7 October 2026, A100
80 GB) showed about 12% average GPU utilization after warm-up and less than
0.9 GiB used VRAM. That motivates testing two and then four workers, **not** a
guarantee of 2x/4x speedup for every method. CPU, Drive I/O and report generation
can be bottlenecks. Per-decode timings under concurrent load are not equivalent
to isolated latency; keep execution logs and compare end-to-end throughput.

## Numerical and data safeguards

- Workers own disjoint whole-image shards, including every message and mode
  belonging to an image. They have separate temporary PNGs and execution logs.
- A fixed Clean, JPEG, Gaussian-noise and rotation probe is run once as a serial
  reference, then by every additional worker. Bits, status, BER and exactness
  must agree. Probes are diagnostics, not extra measurements or best-of-N
  selection. The unchanged Clean replay gate still runs for every pending case.
- The probes do not prove every GPU kernel deterministic under all loads. The
  purpose is to detect immediate execution drift without changing the model.
- Carrier, decoder metadata and source-crop hashes are checked before pending
  cases enter the decoder. ZIP and member hashes are still checked on use.
- Original atomic checkpoints and readback are retained. Valid high-BER or
  rejected results are not retried to improve a score. Technical errors stop.
- Before disposable attack ZIPs are removed, the original raw-result checksum,
  completeness, owner and ZIP-content checks are applied. Embedding-bank PNGs
  and result records are never deleted by this overlay. Existing StegaStamp
  legacy migration retains its original verified-backup rule.

## Fast resume is a stated trust policy, not a full disk audit

### October 16 resume/recovery update

`COCO_Digital_Channel_RECOVERY_FAST_RESUME.ipynb` uses a new, separately named
source ZIP and source directory. The original production and digital release
files remain byte-identical, so existing measurements retain their identity.

Completed methods acquire a small selection-bound receipt after the existing
completion gate succeeds. Later starts read that receipt, compare the same raw
result size/mtime inventory and table fingerprints, and check the catalogue
fingerprints. They do not deserialize or rewrite the large case catalogue or
rebuild reports. An unchanged catalogue is no longer rewritten on ordinary
resume either. The first upgrade of each old receipt still validates its catalogue.

First-time bank audits now save one checked image at a time. If a later image
is damaged or the VM expires, previously checked images can be reused under the
same immutable-input policy. Pending inputs are still hashed before decoding;
`FULL_AUDIT=True` ignores both the compact completion shortcut and image caches.
Neither shortcut claims to detect external edits preserving size and mtime.

The included one-time StegaStamp recovery is restricted to four observed empty
Native PNGs. It uses the original model and messages in an isolated environment.
Replacement requires the original PNG SHA256 and the saved blind-decoded bits
to match for every candidate. Empty originals and recovery evidence are retained
separately. Existing records, hashes, matched images and digital scores are not
rewritten. A nonempty changed file or any mismatch stops the repair.

The embedding bank is treated as immutable experiment input. A checksummed
catalogue reuses previous bank validation and revalidates its contracts and
case keys without opening every already-processed PNG. Result indexes are
checksummed gzip snapshots, one per source image. A single directory listing
compares raw result size/mtime; changed/new records are read and checksummed.
Completed methods additionally require an unchanged case list, result inventory
and report fingerprints. They skip environment installation, decoding and
report rebuilding entirely.

**The first migration must read the old result records once.** Progress is
printed every 50 records; the compact indexes are saved on Drive. This can still
take time. Later sessions reuse them and read only missing/changed records.
A raw checkpoint committed just before a crash but absent from the compact
index is discovered and verified on the next run. Previous evidence is kept.

Fast mode cannot detect an external edit deliberately preserving size and
mtime, nor re-audit already-completed bank PNGs. After moving/repairing storage,
editing the bank, or for a final archival audit, set `FULL_AUDIT = True`. This
rehashes the bank and rereads result records, so it is intentionally slower.
Corrupt cache snapshots fail closed: preserve them for investigation or run a
full audit to regenerate derived caches from raw evidence.

The current-session report is `_reports/FAST_REPORT.md`. A historical canary is
not labeled complete for 100 images. Original method tables remain under
`<method>/tables`. Pairwise report generation occurs only after all selected
methods have finished and is cached until their evidence changes.

## Disconnects

### Approved four-pair recovery (2026-10-16)

`COCO_Digital_Channel_RECOVERED_FAST_RESUME` is the follow-up to the failed
byte-exact recovery attempt. With explicit user approval, only four StegaStamp
image/message pairs were re-embedded once using the frozen production functions.
All four modes and their Clean controls are replaced together. The transactional
recovery archives every previous affected file and record before writing any
replacement, and records new hashes and execution provenance. It does not choose
candidates by BER. This is **new measurement evidence**, not exact restoration.
The notebook requires the committed receipt and verifies it on restart.

New completed receipts store report content hashes as well as file timestamps.
When Google Drive settles a timestamp after upload, identical report bytes remain
reusable. Changed content still invalidates the shortcut. Old receipts lacking
these hashes do not gain retrospective trust: they take the ordinary verified
result/report path once before acquiring a new receipt. This first migration may
still be slow; subsequent starts need no repeated report generation.

The notebook launches a detached controller with stdout written to a Drive log,
not a notebook pipe. Closing the tab or interrupting the log-viewing cell does
not intentionally kill it. Re-running Start attaches to the existing verified
PID instead of launching another copy. Use the explicit stop cell to stop it.
The existing controller lease also prevents concurrent writers.

This cannot keep a terminated Colab VM alive or guarantee an overnight session.
After a VM reset, rerun the setup/start cells with the same paths. Wait at least
three minutes for the old remote lease to expire. Environments and missing SSD
attack ZIPs may need rebuilding; durable decoded results do not.

## Validation

### Report publication repair (2026-10-17)

`COCO_Digital_Channel_REPORT_IO_FIX` changes report storage only. The original
numerical report implementation runs on local SSD, using the validated cached
results and local copies of quality records. It does not change models, attacks,
payloads, matching, metrics or previously measured raw records.

Six completed tables are checked (including gzip EOF checks and SHA256), copied
into an immutable `_report_versions/<version>` on Drive, and read back. Existing
canonical tables are backed up there before publication. Canonical copies use
verified direct writes, not a streaming gzip `.partial` rename on Drive. The
publication marker and completed-method receipt follow successful readback of
the entire set. Multi-file publication is **not atomic**: do not read a partially
published set as completed. After interruption, the same input/result inventory
can reuse the staged version without rerunning metrics or model inference.
Previous reports and decoded evidence are retained; only disposable attack ZIPs
continue to follow the existing cleanup policy.

Run `build_notebook.py`, then `validate.py` in this directory. Validation extracts
the exact distributed archive, checks all source inventories and notebook cells,
runs existing digital-channel tests and new cache/sharding/CPU smoke tests.
No local CUDA model test or Colab speedup is claimed. See the generated
`dist/parallel-validation.json` for actual test counts and limitations.
