# Validation

Date: 2026-09-21. Tests load the exact bundled release, not a separate source tree.

- All bundled Python modules and all four notebook code cells parse successfully.
- The generated notebook passes nbformat schema validation.
- 18 offline tests cover the 29/3 catalogue, full-message layouts, reserved bits,
  exact center-crop pixels at five sizes, blind decode metadata, successful and
  nonzero-BER resume, technical-error history, corrupted PNG/source refusal,
  storage separation, partial reports, runtime geometry checks, infinite PSNR,
  observer RNG isolation and notebook schema.
- Real FIN HEAVY FED+INL CPU smoke: 128x128 crop, 64 raw/useful bits, five messages
  on one author-example image. PNG exact **5/5**, BER **0** for all five.
  The pre-PNG boundary and PNG outputs were recorded. Second call reused the
  same sealed measurement without changing the image/message.
- Real invisible-watermark DWT-DCT-SVD CPU smoke: 512x512, selected 32-bit
  configuration, same author example, five messages. PNG exact **5/5**, BER **0**.
  Resume returned the identical recorded measurement.
- FIN source revision: `f77222ffe9089a5a204e0fd37bd1e33a905be1ed`.
  Input author-example SHA256:
  `760563be6c9abc8a3e900b01cfd4d6265f7e70ef5d8b661d4bad3a87d3c9b124`.
  PyTorch `2.5.1+cpu`; four CPU threads. Evidence is retained locally under
  `.validation/fin_native_smoke` and `.validation/dwt_native_smoke`.

These two real-model smoke tests use an author example, **not the five frozen
COCO images**, and are not results for the 29-model canary. The full GPU run
must produce its own report. No successful GPU/all-model outcome is claimed
by the notebook or this build-time validation record.
