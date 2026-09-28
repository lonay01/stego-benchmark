# Diagnostic recovery validation — 2026-09-21

## Local checks

- 94 tests passed (production and diagnostic suites), including six recovery
  checks. Run with the project `src` and `diagnostics` on `PYTHONPATH`:
  `python -m pytest tests diagnostics/tests -q`.
- All four notebook code cells parse; notebook schema validated with nbformat.
- Every bundled Python file parses. The notebook verifies its source ZIP SHA256.
- Bundled production `src/wmbench/*.py` files are byte-for-byte equal to the
  original files. Recovery does not modify the production registry or adapter files.
- Recovery runs only EditGuard, SepWater and ChunkySeal, with a separate output
  and scratch directory. It leaves the previous diagnostic directory read-only.
- The BasicSR compatibility alias exports the exact torchvision tensor function,
  not a replacement numerical implementation. Build-time dependency installs
  are constrained by the resolved base lock; final installed versions are recorded.

## Real author-checkpoint smoke check

EditGuard author source revision: `174778fb371640bb31b32fcb9f7bfd440c88766b`.
Checkpoint `degrade.pth` SHA256:
`771d0418aae3926c953782a61c064fa6b49b52dba839ed1b7fd3583ffb467983`.

On Windows, Python3.12.14, Torch2.5.1+cpu, torchvision0.20.1+cpu,
BasicSR1.4.2 and setuptools70.3.0, the original adapter with the diagnostic
import shim initialized successfully and loaded the author weights strictly.
This confirms the missing BasicSR/import issue is resolved in that local setup.

A subsequent real 512px encode attempt stopped in the author's inverse wavelet
transform (`common.py:48`), which allocates a tensor with `.cuda()` unconditionally.
The local Torch build is CPU-only. No device or math replacement was made to
produce a misleading passing result. Therefore no successful EditGuard roundtrip
or BER is claimed from this smoke check. Use the GPU runtime for that stage.

The full Linux/A100 environments and five historical-image roundtrips for these
three models have NOT been executed locally. SepWater's setuptools pin addresses
the observed `pkg_resources` failure but is not proof of a successful forward pass.
ChunkySeal's longer timeout addresses the observed preparation termination, not
network speed or model accuracy. Later failures remain explicit, not skipped.

## What is preserved

Eight completed old method checks and the old 45-case scalar PSNR audit remain
in the previous diagnostic report. They are not relabelled or merged with new
observations. Native/matched image bank files, messages, image IDs and PSNR
rules are not rewritten. The recovery report contains only its selected methods.
