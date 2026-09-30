# Validation record

Validated locally on 2026-09-29 using Windows, Python 3.12.14 and PyTorch
2.5.1+cpu. CUDA was not available. These checks do not replace the Colab run.

## Regression and smoke tests

Command from the repository root:

```text
python -m pytest embedding-bank/four-method-night/tests -q
```

Result: **22 passed in 176.72 seconds**. The tests cover:

- The observed EditGuard dependency drift: platformdirs 4.11.12 to 4.12.1,
  with 82 packages in each recorded snapshot.
- Complete-snapshot constraints alongside the unchanged original compile lock,
  and preservation of the dependency identity gate.
- Recovery after a failed environment audit without replacing historical
  snapshots or losing the failed audit.
- SepWater's 224px file API, including decoder geometry; no 256px bypass.
- Real adapter boundaries with explicitly synthetic CPU networks: full payload,
  blind decoder metadata, PNG and float round-trips, fixed serializers and
  seed-reset repeatability. These are not pretrained-model performance tests.
- Read-only historical inputs, corrupted-artifact rejection, compatible v1
  imports, nonzero-BER reuse, changed-source rejection and restart identity.
- Independent profile failures, method-level evidence stops, continued queue
  execution and lightweight partial reports.
- Packaged CLI startup, notebook validation and static source checks.

The builder parses all 81 Python files in the release and compiles all four
code cells. The notebook passes nbformat schema validation. Existing v1
inference modules, registries and locks are checked byte-for-byte against the
frozen source ZIP. The notebook and source ZIP share a verified SHA-256;
`dist/build.json` records the final digest. No Base64 source is embedded.

## Author-source and checkpoint checks

`local_source_smoke.py` was executed against the locally available author
repositories and checkpoints. Its machine-readable output is included as
`validation/local_source_smoke.json`.

- **SepWater:** the hash-verified author control encoder produces a 56x56
  residual. Addition to a 56x56 latent succeeds; a 64x64 latent reproduces the
  reported size mismatch. The author configuration specifies 224px images.
  This test executes the control branch, not the complete pretrained model.
- **EditGuard:** the complete author network was constructed using the author
  options parser. Both `degrade.pth` and `clean.pth` loaded strictly, with all
  1,281 state tensors matching each checkpoint. The checkpoints differ in
  1,092 tensors. Their SHA-256 digests are recorded in the JSON output.
  Image inference was not run because the author's transform requires CUDA.

## Not yet validated locally

The four pretrained GPU pipelines, exact Linux environment restoration,
download availability, end-to-end Colab Drive I/O, runtime and final recovery
rates remain to be measured by this notebook. No successful GPU completion,
paper reproduction or guaranteed overnight duration is claimed.
