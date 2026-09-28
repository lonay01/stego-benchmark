# Validation scope

Local validation on 2026-09-27: **26 tests passed** on Windows, Python 3.12,
PyTorch 2.5.1 and torchvision 0.20.1, CPU only. Six of these checks used locally
available, pinned upstream source files; those optional checks are skipped when
the fixtures are not supplied. No pretrained model performance was measured.

The checks cover:

- Syntax parsing of all 64 packaged Python files and compilation of all four
  notebook code cells; notebook schema validation.
- Byte-for-byte preservation of the frozen base modules, dependency locks and
  registry files; archive checksum verification.
- Synthetic input-bank preflight for all 175 cases, separated historical and
  next-five cohorts, unchanged input-bank contents and protected output paths.
- Resume, artifact-tamper rejection, complete nonzero-BER records and report
  generation in a separate process.
- Independent decoder cross-checks on synthetic tensors and rejection of
  changed inference weights or modified upstream source.
- Execution of the five selected upstream export functions on CPU fixtures.
- Fresh loading of the actual upstream HiDDeN classes from a **synthetic random
  checkpoint**, with matching weights and forward outputs. This is not the
  trained-checkpoint experiment.

This release requires a real-weight GPU run in Colab. Local validation is limited
to static checks, source-contract review, synthetic CPU inference and persistence
tests. Synthetic checks are not performance results for the five research models.

Validation is performed against the generated source ZIP, including unchanged
modules from the pinned native-diagnostics recovery release. Notebook code cells
are compiled without executing Drive authorization, package downloads or GPU jobs.

Do not interpret `checks_complete` as error-free watermark recovery. Completion,
upstream agreement, PNG transport and full-payload BER are separate outcomes.
