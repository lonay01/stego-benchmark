# Validation — 13 October 2026

The delivered notebook is a new diagnostic overlay. The original neural source
ZIP was not regenerated: its SHA-256 remains
`5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2`.

## Checks performed

- Parse and compile every notebook code cell and every diagnostic Python source.
- Validate the nested original archive checksum and every diagnostic member hash.
- Check all 256 integer pixel levels through float normalization and rounding.
- RGB PNG lossless round trip; shape/dtype/range rejection; JPEG and resize controls.
- Read-only root separation and path-escape rejection.
- Immutable evidence and image commits; corrupted PNG/float cache rejection.
- Full payload validation, including InvisMark's six reserved UUID positions.
- Same-byte historical replay versus changed-byte regeneration; no false equivalence.
- High BER and blocked generation are not converted into successful decoding.
- Image-level aggregation rather than treating five messages as independent images.
- Mocked upstream decoders check normalization, detector-coordinate removal,
  extractor hooks, eval mode, exception cleanup and direct/file-path agreement.
- Mocked controls → decoding → resume → CSV/Markdown/gallery smoke run.
- Mocked neural generation preserves all-mode batch composition, padding and
  seeds, retains only Native examples, reuses results and rejects identity drift.
- Mocked VAE float/PNG parity and fail-closed disagreement; historical input
  files stay byte-identical throughout the replay.

Run locally with:

```text
python build.py
python -m unittest discover -s tests -v
```

The CPU suite uses NumPy tensor doubles and synthetic model outputs. These are
not experimental observations. It does not validate real CUDA kernels, downloads,
Google Drive behaviour, actual GPU memory requirements or watermark quality.
Those remain the purpose of the delivered Colab run. No paid runtime was started
and no production/previous-neural result was modified during preparation.
