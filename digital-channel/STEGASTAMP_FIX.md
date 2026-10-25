# StegaStamp Clean replay repair

Use `dist/COCO_Digital_Channel_STEGASTAMP_FIX.ipynb`. Run its cells in order
with the original bank/output paths and `N_IMAGES = 5`. The original source ZIP
is reused from Drive; no new ZIP is required. Stop any previous runner first.

## Observed failure

The supplied notebook completed 19 methods and committed two StegaStamp
carriers before stopping on `000000000285_m01_matched_42`.
The saved Drive diagnostic reports one changed decoded bit, at zero-based
index 51: historical BER 0.48, fresh BER 0.47. Both extractions are non-exact.
Source, dependency and artifact checks passed before this comparison.

The exact carrier was downloaded read-only and its SHA256 verified:
`920c02620674607f39b9a20d4f4a851599ac38b01450c13dbadf19343084dcc1`.
Eight local CPU extractions with the original ONNX graph all reproduced the
fresh 100-bit result. Inspection of an in-memory diagnostic graph showed
`Sigmoid -> Round`; bit 51 had probability approximately 0.5013 on CPU.
This is consistent with numerical/backend replay drift, but it does not prove
the precise cause of the historical CUDA difference. CPU ONNX Runtime 1.20.1
was used only for this diagnostic, not substituted into the benchmark.

## Exact scope

Only the StegaStamp Matched Quality Clean comparison receives a bounded
exception. The PNG hash and original model/dependency checks must still pass.
Both results must be measured, non-exact, internally consistent 100-bit
extractions differing at exactly one position. Two extra decodes must match
the first fresh result exactly. The first result is kept whether its BER is
better or worse. No retries select a winner and no decoded bits are corrected.

The historical bank stays unchanged. A receipt retains old/fresh results,
both verification decodes and execution metadata under
`stegastamp/clean-replay/`. The result record references that receipt. The
existing report code already computes attack deltas against `rows.clean`,
so the current first-fresh Clean extraction is the paired reference.
If a partial case already contains a different Clean result, it still stops
to avoid combining attacks measured against two different controls.

Native drift, larger differences, unstable repeated decoding, changed hashes,
dependency/weight changes and all other methods retain hard-stop behavior.
This is an explicit replay-policy exception, **not** a claim of bitwise
reproduction of historical Clean. The 47% BER remains a poor measured Clean
result; this repair does not improve watermark quality.

## Preservation and validation

The original source ZIP remains byte-identical:
`fbaec01ed86f1b0293250e9cde03aca8d1a9a63a0c0a2c69b761b0d1d74577ff`.
The repair is a separate, hash-checked readable Python module. One comparison
block is replaced in memory for the isolated StegaStamp decode process only.
The notebook changes one original run entry point and adds the repair cell;
the other original cells are unchanged. Completed records remain reusable.

Validation: 29 existing regression tests with the repair installed and 14
repair tests passed. These include the real Meta CPU fixture, all 79-condition
mock-decoder commit/resume flow, unchanged bank hashes, failure safeguards,
and executable notebook setup. All seven notebook code cells parse; the
notebook schema validates. See `dist/stegastamp-fix-validation.json`.
The repaired end-to-end StegaStamp run on A100 has **not** been executed locally.

Build: `python build_stegastamp_fix.py`.
Validate with the existing diagnostic dependencies:
`python validate_stegastamp_fix.py`.
