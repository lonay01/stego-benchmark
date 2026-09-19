# Embedding bank: validation record

Scope: the dedicated September 18 embedding-bank release, not the previous
neural-channel experiment. No full COCO100 run was launched during development.

## Automated checks

The original release passed 43 offline unit tests in the local CPU validation environment:

- 16 bank tests: deterministic messages, reserved positions, 100-file input
  freeze, exact scalar pixels, PSNR tolerance, resume, corruption and relocation.
- 15 recovery tests: actual checkpoint identity versus transport archives,
  narrow InvisMark log exclusions, task signatures, deterministic retries,
  partial-key preservation, and no retries of successfully measured poor BER.
- 12 additional-adapter tests: pinned registry/dependency contracts, source compilation,
  light controller imports, payload mapping, blind decoder interfaces and mocked
  control flow. These do **not** validate real new-model checkpoints.

Ruff checks and formatting were applied to the new source and tests. Every code
cell in the delivered notebook is compiled. The separate source ZIP is checksum-checked,
its Python source parsed, and its 32-entry registry imported in a separate
process after extraction to a different directory. No Torch import is required
by that controller/package check.

Commands from the source directory, with `src` on `PYTHONPATH`:

```bash
python -m unittest discover -s tests -p 'test*bank*.py' -v
python scripts/build_embedding_bank.py
python scripts/check_embedding_bank_release.py
```

The development CPU environment used Python 3.12.14, NumPy 1.26.4,
Pillow 10.4.0, Torch 2.5.1+cpu and ONNX Runtime 1.20.1. Production model
environments retain their own registered dependency versions; those are not
silently replaced with this test environment.

## Controller hotfix validation

The updated suite has 56 tests: the original 43 plus 13 regressions for the unsupported Camera
entry, continuation through all five following methods, bounded worker-error
logs, audited code-only resume and rejection of scientific changes. The actual
notebook ZIP-extraction code is tested with valid, mismatched, path-traversing and
locally modified source bundles. These checks do not load GPU models.

The old source was extracted from the user-supplied executed notebook. Its source
ZIP hash is `9736c8bfb5a32d59c9f622fd61eff2537bb02a623107c96870a3d78277897aac`.
Only `src/wmbench/embedding_bank.py` changed among runtime source files. The full
registry is unchanged. AST comparison confirmed that message construction, task
signatures, image processing, decoder checks and model identity normalization
remain unchanged. Payload comparisons for all 31 payload-capable entries on five
test IDs were identical (155 comparisons).

## Real cached-model smoke

DWT-DCT-SVD, PIMoG, SSRH and StegaStamp ONNX were run on one cached natural
RGB512 image, each in an independent subprocess. Network connections were
forbidden in the children. No new dependencies or checkpoints were downloaded.

| Method | Native BER30 | Matched39 BER30 | Matched42 BER30 | Matched45 BER30 |
|---|---:|---:|---:|---:|
| DWT-DCT-SVD | 0 | 0 | 0 | 0.0667 |
| PIMoG | 0 | 0 | 0 | 0.0333 |
| SSRH | 0 | 0.1667 | 0.3333 | 0.4333 |
| StegaStamp ONNX | 0 | 0.2333 | 0.2667 | 0.3333 |

All 16 PNGs were saved; all matched PSNR values were within 0.05 dB of their
targets. A second invocation preserved the files byte-for-byte and added zero
embedding or decoding calls. BER increases were retained, not repaired.

The input's SHA256 was
`9b0d5ba68cff1bcaeffd4f0dca3bd54b875072b13d314aa08bd4487fecf074d5`.
This test does not establish membership in the frozen historical COCO100 corpus.
Its table is an interface check on one image, not a model ranking.

Reproduction script: `scripts/smoke_embedding_bank.py`. It requires an existing
local source/weight cache and refuses network connections. Original local evidence
is under `validation/embedding_bank_cpu_20260918_183445/`.
The controller hotfix was also checked with the same four cached implementations
and the same fixture: all four interface/resume checks passed. Local evidence is
under `validation/embedding_bank_cpu_20260919_194354/`. Validation image banks and
model weights are not included in the public source bundle.

## Not established by these checks

- All 32 models running successfully on Colab, Windows or any particular GPU.
- Real pretrained inference for the five newly added adapters.
- Availability of EditGuard, OmniGuard and SepWater manual checkpoints.
- Licensed MATLAB/toolboxes for Fang or a working Camera-Shooting Instance adapter.
- Detection calibration, robustness under attacks, or research conclusions from COCO100.

These limits are why the distributed notebook starts at five images and reports
incomplete methods explicitly. A catalogue count is never used as a success count.
