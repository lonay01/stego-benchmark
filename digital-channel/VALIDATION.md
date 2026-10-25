# Validation boundary

The release reuses the pinned production adapters; it does not claim that a new
29-method CUDA digital run has already completed.

Executed locally on 4 October 2026:

- Static parsing of all six notebook code cells and all 80 bundled Python modules;
  notebook schema validation; both source inventories checked by SHA-256.
- 29 unit/integration/smoke tests: exact 79-condition matrix; native geometry;
  deterministic/common randomness; retained crop area; read-only production-bank
  contracts; extending N without changing run identity; saved ZIP regeneration;
  checksum failure; local/Drive routing; isolated decoder command; stage ordering;
  blind full-payload scoring; ECC/technical errors; Clean drift; readback before
  deletion; rejection of foreign cleanup paths; image-balanced and paired statistics.
- Real Meta Revisited HiDDeN checkpoint on CPU: embed one fixture, serialize,
  apply Clean/JPEG70/rotation +5/compound level 1, serialize again, blind-decode
  48 bits. All four fixture observations returned 48 bits with zero errors. These
  are integration observations, not COCO benchmark estimates.
- Real PSNR/SSIM quality-path check. LPIPS plumbing uses a stub in this test:
  pretrained AlexNet LPIPS was **not** executed locally because its weights were
  not cached. Colab uses the actual pinned LPIPS model, never that test stub.

See `dist/validation-results.json` for the executed machine-readable result.
Local tests used Python 3.12, NumPy 1.26.4, Pillow 10.4.0, SciPy 1.17.1 and CPU
PyTorch 2.5.1. Colab pins Python 3.10.12 / SciPy 1.11.4 for the attack/controller
environment; each reviewed decoder has its own original environment. Offline
tests do not claim that the pinned Linux dependency installation was exercised.

The first user run should retain `N_IMAGES=5` and all 29 methods. Check technical
coverage, the full Clean controls, attack outputs, result readback and cleanup
before raising N to 100. A new GPU/runtime or upstream download failure can still
stop an isolated environment. Strict evidence comparisons are intentionally not
disabled to make such failures disappear.
