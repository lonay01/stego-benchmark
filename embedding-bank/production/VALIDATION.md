# Validation boundary

This release requires an end-to-end five-image Linux/A100 run before increasing
N to 100. The reviewed method diagnostics are prior evidence, not execution of
this newly integrated release.

## Executed locally, 1 October 2026

The validation runner checks 13 tests, including real pretrained Meta Revisited
HiDDeN CPU inference through the new carrier/matching/clean-decoding worker.
The bundled Python modules and all four code cells are parsed, the notebook
passes nbformat validation, and extracted file hashes match the release manifest.
`validation/local_checks.json` records the exit status and exact output for the
final source archive; any failed validation is visible there.

On one official Meta example resized to a synthetic test parent and then cropped
to 256px, Native PSNR was 24.4583 dB with BER 0%; matched39 had PSNR 39.0073 dB
and BER 2.0833%; matched42 had PSNR 41.9639 dB and BER 4.1667%. The 45 dB search
missed the prescribed tolerance and was recorded as unmatched, without exporting
a falsely labelled matched PNG. These numbers are a local integration smoke,
not COCO evaluation results or a paper reproduction.

Other tests cover the 29-entry inventory, full payloads/reserved bits, native
geometry, per-mode records, zero decoder-reference leakage in the fixture,
nonzero-BER reuse, technical decoder retries without re-embedding, corruption
rejection, count-independent identities, incomplete coverage, scheduler stage
order, fixed author locks/import roots, and synthetic author-adapter tensor
wiring. Synthetic neural cores are explicitly not pretrained author tests.

Local validation is performed by `validate_release.py`: syntax checks for all
packaged Python and notebook cells, artifact/checksum consistency, catalogue
contracts, clean full-payload scoring, scalar matching, decoder retry without
re-embedding, prefix extension, and evidence-tampering checks. Synthetic adapters
test orchestration only; they do not stand in for pretrained model accuracy.

The validation JSON records the actual executed local checks. No cloud runtime,
model installation, or paid GPU experiment is automatically started by building
the notebook. The local machine does not establish all-29 CUDA compatibility.
