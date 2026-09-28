# Validation record — full-payload canary

Date: 2026-09-21. This records checks executed locally, not a completed Colab
experiment. The old production code, old notebook and saved experimental results
were not modified.

## Offline checks

`python -m pytest embedding-bank/full-payload/tests -q`: **32 passed**.

The tests use the built source ZIP, not a different development import path:

- All 32 capacity contracts; reserved positions and native coding interfaces.
- Stable full-length messages, common prefixes, distinct replicates, no repeated
  30-bit block. Full-information BER, reserved-bit errors and ECC rejection.
- Real PNG round trips using a small pixel-based test adapter; no reference
  payload supplied to its decoder. This adapter is a test fixture, not a method.
- Scalar matching and out-of-range/quantized unattainable targets.
- No re-embedding on resume; no re-decoding of measured, imperfect results.
- Retry only technical failures, retaining prior error records.
- Corrupted artifacts and changed task identities stop reuse without overwrite.
- Read-only source inputs, stable N=5 to N=100 prefix identity, dependency-lock
  restoration, explicit missing-checkpoint blockers and owned-scratch cleanup.
- Summary denominators, per-mode CSV output and retained auxiliary measurements.
- Notebook schema, AST parsing of all four code cells and every bundled Python
  source; source ZIP checksums; unchanged production source bytes in the bundle.
- Fresh-process controller imports do not load Torch, TensorFlow, Diffusers or
  ONNX Runtime; model loading is confined to isolated workers.

## Real CPU smoke checks

Both checks used the **FIN author's `test_images/1.png`**, resized to RGB512 with
Pillow bicubic. This is **not one of the historical COCO canary images**. One
full-length message was tested per method, plus identical-result resume checks.
Windows, Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, four CPU threads.
This is not a validation of every pinned Linux/CUDA environment.

Source image SHA256:
`760563be6c9abc8a3e900b01cfd4d6265f7e70ef5d8b661d4bad3a87d3c9b124`.

| Check | FIN HEAVY, 64 bits | DWT-DCT-SVD, 4096 bits |
|---|---:|---:|
| Native PNG BER | 0.03125 (2/64) | 0.00390625 (16/4096) |
| Matched 39 dB | Unmatched within fixed scalar policy | 0.011474609375 (47/4096) |
| Matched 42 dB | 0 | 0.051025390625 (209/4096) |
| Matched 45 dB | Unmatched within fixed scalar policy | 0.253662109375 (1039/4096) |
| Repeat call | Identical sealed results reused | Identical sealed results reused |

FIN source was checked at commit
`f77222ffe9089a5a204e0fd37bd1e33a905be1ed`, with real `HEAVY/FED.pt` and `INL.pt`
loaded strictly. All ten auxiliary resolution/serialization stages completed.
The author128 float, clipped float and PNG paths each recovered 64/64 bits;
the residual-to-512 float and PNG paths each recovered 62/64 in this example.
The no-INL control is labelled separately and does not replace normal decoding.

DWT used `invisible-watermark==0.2.0`, PyWavelets 1.8.0, one active chroma
plane, block4 and scales `[0,36,0]`. All four saved modes decoded to 4096 bits.
The maximum-length interface worked; exact message recovery did **not** pass.
These observations are not a leaderboard and were not used to tune alpha or
select a favorable message.

Reproduce with the optional `smoke_cpu.py` using an existing pinned FIN checkout:

```text
python smoke_cpu.py --author-repo PATH_TO_FIN --output NEW_FOLDER --method fin
python smoke_cpu.py --author-repo PATH_TO_FIN --output ANOTHER_NEW_FOLDER --method dwt_dct_svd
```

Run the builder first. The second command also needs invisible-watermark and
PyWavelets; `--deps` accepts a separate directory of locally installed packages.
The script never downloads model weights or writes to the old image bank.

## Not executed locally

- The complete five-image, five-message, 32-entry GPU queue.
- Real full-payload inference for the other methods, including SSL at 2048 bits.
- Installation of every model's isolated Linux/CUDA dependency environment.
- CUDA memory/time measurements or a Colab disconnect/reconnect test.
- A 100-image bank or any channel attacks.

Camera-Shooting Instance has no verified adapter; Fang needs licensed MATLAB;
OmniGuard still needs matching trained copyright weights. These entries remain
visible as blockers. Public-asset availability may also fail at run time. A queue
finishing successfully is not proof that all methods encoded or recovered every
message. Inspect Native BER, exact counts, unmatched targets, per-mode status,
auxiliary diagnostics and error logs before planning the production run.
