# Validation

Recovery validation: 2026-09-26. Local runtime: Python3.12, PyTorch2.5.1+cpu.

- All **54 bundled Python modules** and **four notebook code cells** pass AST
  parsing. The notebook passes nbformat schema validation.
- **43 tests pass**, importing the exact release ZIP distributed with the
  notebook. The full recovery suite took about17 seconds on the local machine.
- CPU integration smoke covers the complete per-case workflow for each of the
  five method interfaces using explicitly synthetic test doubles: historical
  replay, baseline capture, fixed serializers, controls, reports and resume.
- Actual PyTorch tensor tests cover all five decoder normalization/threshold
  branches, encoder hooks, serializer hooks, same-latent observation, restoration
  of patched methods and unchanged RNG/output during observation.
- Other tests cover frozen complete payloads, same historical image/message
  selection, read-only original evidence, corruption refusal, terminal nonzero
  BER, technical-error history, pixel-identical PNG round trips, quantization
  rules, incomplete reports, model training-mode rejection, controller lease
  setup and completed-task reuse without environment preparation.
- Compared bundled definitions for all five methods against the actual
  `native_resolution_canary_report.zip` fetched from Drive on2026-09-22.
  All five registries match the historical `trial.json` and inference identity.
  All five historical requirements-lock hashes match their saved dependency
  identities. No replacement checkpoint/configuration was introduced.

## Recovery-specific checks

- Reproduced the real Python `inspect.getsourcefile()` TypeError with a live
  PyTorch class whose defining module is absent from `sys.modules`. Both the
  HiDDeN and RoSteALS audit branches now complete on these synthetic networks,
  without modifying the import cache. The fallback hashes the actual method
  source file; unavailable source is explicitly labelled, not invented.
- RoSteALS checkpoint tensor comparison still rejects missing and modified
  inference weights. The existing evaluation-mode rejection remains active.
- Recovery refuses unknown predecessor identities, changed trial/model/policy
  definitions, and changes outside the reviewed diagnostic-source patch.
- A controller smoke test runs prepare/work only for the two pending methods,
  preserves the other three methods' files byte-for-byte and archives prior
  error statuses. Per-case resume and artifact-corruption checks are also tested.
- Compared the new ZIP with the original September22 ZIP. Only three existing
  modules changed: `probes.py`, `runner.py`, `worker.py`; `recovery.py` is added.
  The ASTs of capture, embedding observation, float decoding, quantization and
  per-case measurement functions are unchanged. All other source modules match
  byte-for-byte, including production adapters, scoring and storage checks.
- Exercised recovery against the actual `diagnostic_identity.json` and
  `plan.json` from the downloaded September22 report, using a temporary copy.
  Both files remained byte-identical and the preserved plan identity matched.
  This test does not claim to verify PNG/NPY files omitted from the review ZIP;
  the normal artifact checks run when the notebook accesses Drive.

## Real-model evidence versus local validation

The September22 Colab report completed SepWater, SepMark and DistSeal. HiDDeN
and RoSteALS failed in source inspection before their diagnostic measurements.
The recovery does not manufacture new results for those two methods.

**Not executed locally:** pretrained inference on the historical COCO images,
isolated Linux environment installation, Google Drive mounting, or a CUDA/A100
run of this recovery notebook. Synthetic CPU tests validate the machinery, not
the research models' decoding success. A resumed Colab run is still required.
