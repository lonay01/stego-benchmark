# Controller lease read recovery

The October 18 run stopped while decoding EditGuard because Colab could not
read `_reports/active_controller.json`. The file remained present through the
Drive API. This establishes a filesystem/API visibility mismatch, not its exact
cause. The old reader allowed only 0.8 seconds of backoff for missing-file reads.

This execution-only repair retries transient missing/transport errors and partial
JSON with 39 seconds of backoff in total. It never substitutes cached ownership,
recreates a missing lease, or accepts another controller's token. A persistent
failure still stops the run. Retry backoff cannot bound a kernel-blocked FUSE read
or keep a terminated Colab VM alive. Acquisition, remote expiry and single-writer
checks remain those of the original controller; Drive is not a distributed lock.

Before starting, a storage preflight requires the existing embedding bank and
result folders. Paths under `/content/drive` require an actual mount. A uniquely
named probe is written, synced, read back, and removed; failed probes remain for
diagnosis. No bank image, result, ownership file or scientific setting is changed
by this preflight. It is an immediate filesystem check, not a cloud durability
guarantee.

Use the same bank/output paths, image count and GPU type. Run the new notebook's
cells in order and authorize Drive if prompted. Never run the old and new
controllers simultaneously. Completed method receipts are reused; partially
decoded cases resume from valid checkpoints. Missing disposable attack ZIPs on a
new VM may have to be recreated, but measured rows are not intentionally rerun.

The previous release ZIP is unchanged. The new ZIP has a separate hash inventory
and source directory. CPU tests inject temporary absence, persistent absence,
transport errors, permission errors, corrupt JSON and competing owners. They do
not claim to reproduce Google's FUSE implementation or test all models on GPU.
