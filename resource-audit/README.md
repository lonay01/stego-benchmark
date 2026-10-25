# Benchmark resource audit

Read-only accounting of saved experiment logs. This is separate from the embedding,
digital and neural runners; it does not change a scientific release or resume state.

## Run

Open `dist/BENCHMARK_Resource_Audit.ipynb` in a **separate CPU Colab runtime**, mount
Drive and run all cells. Do not change or stop the existing GPU notebook. No package
installation, model weights or GitHub access is required. The notebook contains the
readable Python source, not an encoded archive.

`BASE` defaults to `/content/drive/MyDrive`. Known historical experiment folders are
checked explicitly. Use `EXTRA_ROOTS` for moved or additional runs. Optional date
filters use Moscow calendar dates; `END_DATE` is inclusive. By default all available
history is considered, rather than assuming a month of continuous computation.

Outputs go to a unique folder under `watermarking_resource_audit`:

- `REPORT.html`: supervisor-facing Russian report; print to PDF in a browser.
- `attempts.csv` / `attempts.json`: attempts, timestamps, stage wall time, outcomes,
  sampled resource measurements and source paths.
- `methods.csv`, `daily.csv`: per-method accounting and non-overlapping daily coverage.
- `summary.json`, `issues.json`, `evidence_manifest.json`: accounting, omissions,
  snapshot hashes and cache provenance.
- A ZIP beside the report folder, assembled by the final notebook cell.

Inspect the result before sharing. Files may include experiment names and paths, but
the audit does not read or export message bits, images, model weights or free-text logs.

## Definitions and limits

**Recorded stage hours** sum the saved `wall_seconds` of dated attempts, clipped to
the selected period. Concurrent stages are additive here. Setup, I/O and waiting
inside a monitored process are included. A stage returning zero is not proof that a
method passed its scientific checks or completed a 100-image experiment.

**Calendar coverage** merges overlapping recorded intervals across the selected runs.
For attempts lacking a terminal record it adds only adjacent telemetry samples no more
than 30 seconds apart. It never extrapolates the last sample to the present. This is
elapsed-time coverage, not the sum of machine-hours on multiple independent servers.

Dates come from the recorder's `attempt = time.time_ns()`, not the model-session
filename or a file modification date. Legacy author-run `.outcome.json` files use the
timestamp in their explicit `stage` field and recorded `seconds`. Undated timings are
retained in the ledger but excluded from date-filtered totals. Unknown schemas are
reported, not guessed. Copied attempts are deduplicated by attempt ID; conflicting
terminal records are excluded with a warning. The format assumes time.time_ns() IDs
are unique across the project's runs, as in the current recorders.

GPU memory/utilization are sampled **device-wide**, not attributed to one process.
The utilization column is an unweighted mean of available integer-percent device
samples within an attempt, not a time integral or utilization of the whole month.
RAM is process-tree RSS; temporary slot size is neither total Drive storage nor GPU
memory. Peaks span the full overlapping attempt, even if date filters clip its duration.
Concurrent device readings are never added into an alleged total GPU cost.

No monetary cost, Colab compute-unit consumption, FLOPs, continuously allocated GPU
hours, or training-hours are inferred. Billing evidence is needed for actual spending.
Most recorded work here is benchmarking pretrained watermarking methods.

The scanner checks fixed metadata directories only, not millions of images/result
records. Root and per-method `timings`, `logs`, `_logs`, and `_meta/{timings,logs}` are
supported. Nested arbitrary legacy layouts, ZIP archives, lost/overwritten logs, manual
development and gaps outside monitored stages are not reconstructed. The report is
evidence available to this scanner, not a complete claim about all work performed.
Old method directories are intentionally included in a history audit; their presence
does not reintroduce excluded methods into the production benchmark.

Limits: 20,000 files, 2 GiB of newly read content, 30 minutes, 64 MiB per telemetry file
and 2 MiB per timing file by default. A triggered limit yields an explicitly partial
report. FUSE operations themselves may block; the time limit is checked between reads,
not a hard cancellation of an operating-system call. Live files are read to a snapshot
size, with changes/partial records flagged. Retry later for a stable final snapshot.

Repeated runs cache parsed data by path, size and mtime. Cache hits are labelled and
reuse the **previous** SHA-256; they are not a fresh content verification. Set
`USE_METADATA_CACHE = False` to re-read and rehash. Cache writes and report writes occur
only under the separate audit output folder. No source file is modified or deleted.

## Local use and tests

Python 3.10+; the audit itself uses only the standard library. In Python:

```python
from pathlib import Path
from resource_audit import choose_roots, run_audit

base = Path("E:/saved_experiments")
summary = run_audit(choose_roots(base), Path("E:/benchmark_audit"))
```

For development only, install `pytest` and `nbformat`, then run:

```text
python build_notebook.py
python -m pytest tests -q
```

Tests cover overlap, clock identity, date clipping, missing final records, telemetry
gaps, malformed files, cache invalidation, copied/conflicting attempts, file limits,
source immutability, HTML/CSV escaping and execution of the notebook's embedded audit
code. Colab OAuth and access to the complete mounted Drive still require a user run.
