# Diagnostic release validation — 2026-09-20

## Scope and result

- 88 tests passed: the 67 existing embedding-bank tests and 21 new diagnostic
  tests. The command was `python -m pytest tests diagnostics/tests -q`, with
  `src` and `diagnostics` on `PYTHONPATH`.
- All four notebook code cells pass Python AST parsing; the notebook passes
  nbformat validation. Python files in the source ZIP also pass AST parsing.
- Synthetic end-to-end worker tests check exact historical payload reuse,
  full/native versus common30 metrics, read-only bank behavior and resumption.
- Real Torch CPU tensor fixtures exercise FIN, HiDDeN and DistSeal stage
  plumbing. These fixtures are not pretrained-model accuracy tests.
- Source/registry/locks and the production notebook were not edited. The Git
  working tree contains only the new `embedding-bank/diagnostics/` directory.

## Real FIN checkpoint smoke

This separate test uses FIN's own `test_images/1.png`, not the five historical
COCO images. It reuses the bank's 64-bit payload for COCO ID285 solely as a fixed
message. It cannot establish historical benchmark accuracy or GPU equivalence.

- Source revision: `f77222ffe9089a5a204e0fd37bd1e33a905be1ed`.
- Source image: RGB512, SHA256
  `760563be6c9abc8a3e900b01cfd4d6265f7e70ef5d8b661d4bad3a87d3c9b124`.
- HEAVY FED SHA256:
  `791ab24f051581adbb79c7c3fb0e5e5e9b34327d8fc9ff853547ad3c2db7fb0f`.
- HEAVY INL SHA256:
  `2698ad4c1467d457031e1e8c04d92448c8e98eb69462e4ec08b682e2d18392fa`.
- Python3.12.14, Torch2.5.1+cpu, torchvision0.20.1+cpu, four CPU threads.
- Ten stages executed successfully. Approximately2.5seconds after constructing
  the adapter through the final diagnostic stage, excluding downloads/install.

| Stage | Common30 errors | Full64 errors |
|---|---:|---:|
| Author PIL128, before PNG | 0 | 0 |
| Author PIL128, clipped | 0 | 0 |
| Author PIL128, PNG | 0 | 0 |
| No-INL labelled control | 0 | 0 |
| Current tensor128, before PNG | 0 | 0 |
| Current tensor128, clipped | 0 | 0 |
| Current tensor128, PNG | 0 | 0 |
| Residual transfer to512, float | 2 | 4 |
| Residual transfer to512, clipped | 2 | 4 |
| Residual transfer to512, PNG | 2 | 5 |

This example supports testing the resolution adaptation explicitly; it is not a
general estimate of FIN's BER. Reproduce with `diagnostics/smoke_fin.py` using the
pinned FIN checkout and a new output directory.

## Author assets inspected locally

- EditGuard `degrade.pth`: 37,760,088bytes; safe tensor-only loading succeeded.
  The state has irn/bitencoder/pm/bitdecoder parameter names.
- SepWater `last.ckpt`: 519,137,410bytes; a non-executing pickle-name inspection
  found ae/control/decoder subtrees. Full inference loading remains a Colab test.
  The redundant standalone f4 autoencoder is not downloaded by this diagnostic.
- OmniGuard `checkpoint.zip`: downloaded and inspected. The author hybrid
  checkpoint has a frozen TrustMark-Q component, not a verified trained release
  of the newly published copyright architecture selected in the bank. No silent
  substitution is made.
- The three reviewed download hashes are pinned in
  `bank_diagnostics/assets.py`; these are our observed hashes, not author-signed
  checksums.

## Not yet verified

The complete five-image GPU diagnostic has not run here. Windows CPU smoke does
not validate every isolated Linux/CUDA environment or current download access.
EditGuard and SepWater must pass strict initialization and their round trips in
the notebook. OmniGuard's selected checkpoint, Fang's licensed MATLAB and the
Camera-Shooting Instance integration remain explicit blockers. Production changes
must wait for the saved Colab diagnostic report.
