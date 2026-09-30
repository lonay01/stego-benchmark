# Validation record

This release is a diagnostic, not a completed new GPU experiment.

- Historical native diagnostic records were read from the existing Drive report.
  Archive SHA-256: `925bb32e1cd98e0b7a9eed2bad4c3b616f54f5ce1f142c9c93fc0964752d1666`.
  All 75 AE-control records were present, 25 each for RoSteALS, SepWater and DistSeal.
- The pinned EditGuard README and inference source were inspected. Its clean and
  degraded-condition checkpoints are distinct author configurations.
- The build parses every Python source in the archive and compiles all four
  notebook code cells. **13 diagnostic tests passed**. They include a CPU smoke
  through the actual EditGuard adapter with a synthetic neural core, fixed
  geometry checks, same-latent control, full-payload assignment, read-only source
  validation, resume and artifact-corruption rejection. This does not test the
  pretrained EditGuard network.
- **74 active-code regression tests passed**, including the real Meta CPU smoke
  using the official local fixtures. No optional test was skipped in that run.
- Full GPU inference for the new geometry controls and EditGuard clean checkpoint
  has not been run on this Windows CPU-only host. No paper-equivalence or final
  100-image readiness claim is made.

Historical releases are intentionally not rebuilt or migrated by this change.
