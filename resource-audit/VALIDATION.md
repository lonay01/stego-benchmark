# Validation — 25 October 2026

- Python 3.12.14, Windows; audit code uses Python 3.10+ standard-library APIs.
- 25 automated tests passed, including static compilation of every code cell,
  nbformat schema validation and sequential execution of all notebook code cells
  with a temporary local dataset instead of a mounted Drive.
- All source files compile with `compileall`.
- A separate smoke run parsed 30 real ChunkySeal stage timings, one SDXL telemetry
  file and one RoSteALS author-run outcome, read from the connected Google Drive.
  It produced the HTML report, CSV tables and JSON provenance without parse errors.
  This deliberately small snapshot is **not** a complete audit of the user's history
  and must not be presented as the total resource expenditure.
- Original benchmark releases, experiment images, scientific results and running
  Colab processes were not modified or restarted.

Not validated here: interactive Google OAuth in a fresh Colab, a full filesystem
scan of the mounted Drive, and actual Colab billing. Those require opening the
delivered notebook, mounting Drive, and reviewing the generated report/coverage.
