# Recovery of the 2026-09-20 diagnostic run

Use `dist/Embedding_Bank_DIAGNOSTICS_RECOVERY.ipynb` with
`dist/Embedding_Bank_DIAGNOSTICS_RECOVERY_source.zip`. Select the same A100/high-RAM
Colab runtime, keep five images, and run all cells. Upload the matching ZIP when
asked. Do not run the original diagnostic notebook in parallel.

Only EditGuard, SepWater and ChunkySeal are retried. The old eight completed
checks and the 45-case scalar audit are not repeated or relabelled. The image
bank and old diagnostic directory are read-only. New results are stored in
`MyDrive/watermarking_embedding_diagnostics_recovery_v1`.

## Confirmed causes

- EditGuard: `ModuleNotFoundError: No module named 'basicsr'` in author
  `Inv_arch.py`. BasicSR1.4.2 is installed after Torch with build isolation
  disabled. Its removed torchvision import is redirected to the same tensor
  `rgb_to_grayscale` implementation. Model arithmetic and weights are unchanged.
- SepWater: Lightning1.8.6 imports `pkg_resources`, but the installed
  setuptools84 no longer provides it. The recovery pins setuptools70.3.0.
- ChunkySeal: almost 60minutes were spent downloading a 12.4GiB archive and
  initializing it. Our 60-minute preparation ceiling killed the process before
  work began. Recovery allows up to three hours for its preparation, bounded by
  the overall session budget. It does not claim faster network transfer.

The two already downloaded author checkpoints are copied from the old diagnostic
asset directory with checksum verification. The original copies are unchanged.
ChunkySeal may need downloading again because the old notebook already cleaned
its failed scratch. This cannot recover data that was deleted by that run.
Recovery retains failed scratch so a retry in the same Colab runtime need not
repeat successful downloads/installations. A runtime reset still loses scratch;
finished recovery image cases remain on Drive.

All changes are confined to diagnostic packaging and its isolated workers.
The production source, registry, checkpoint choice, payloads and scalar PSNR policy
are unchanged. New dependency environments and logs are saved in the recovery
folder. Read the original `watermarking_embedding_diagnostics_v1/REPORT.md` for
the eight completed methods, and the recovery `REPORT.md` for these three.

CPU/import checks are not a substitute for the complete Linux/A100 five-image run.
Any later strict checkpoint or forward-pass failure remains visible in the logs;
no random-weight, missing-key or favourable-message fallback is allowed.
