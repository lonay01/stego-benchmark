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

That controller update passed 56 tests: the original 43 plus 13 regressions for the unsupported Camera
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

## FNNS / DistSeal loader fix (September 20)

The supplied Colab run failed during initialization of these two models:

- FNNS called `requires_grad_(False)` on a TorchScript module. TorchScript rejects
  that module-level operation. The adapter now freezes each parameter separately;
  the input-image/distortion gradients required by FNNS remain enabled.
- DistSeal's checkpoint contains 418 extra `autoencoder.model.*` keys. The pinned
  upstream `setup_model` constructs VideoWam without that training autoencoder.
  The adapter now excludes only that exact subtree before `strict=True` loading.
  Missing inference weights, unknown other keys and tensor-shape mismatches
  still fail. The separate, registered diffusers DC-AE checkpoint is unchanged.

The current suite contains **67 tests**. The eleven added regressions cover actual
TorchScript input gradients with frozen weights, strict DistSeal tensor loading,
the constructor's separate autoencoder path using small substituted networks,
guarded migration/resume, and notebook upload of a new source ZIP when an older
release is already present. Tests using substituted networks do not establish
successful loading of the full pretrained DistSeal pipeline.

The migration accepts only the two known previous source fingerprints, with an
unchanged dataset, registry and policy. Before its first receipt is written, both
affected methods must have no saved model identity, image records or PNGs.
Existing results from other methods are not modified. A receipt bound to this
bank permits later resumes of newly produced outputs. Different scientific
settings, unknown source hashes and mismatched receipts remain errors.

### Real FNNS checkpoint smoke

The official 48-bit decoder was downloaded from
[Meta's checkpoint host](https://dl.fbaipublicfiles.com/ssl_watermarking/decoder_48b.torchscript.pt).
Its SHA256 is
`d6f0c052efb0a26869322ee657d9982bba79b32c2e0305fd4339aff45b58ac49`.
The test used the registered OmniSealBench revision
`bd141394a0dfeac0b4cfedfdb4feaebb2a2c6db6`, Torch 2.5.1+cpu and
Torchvision 0.20.1+cpu. It retained the registered 10 optimization steps,
strength, seed and payload layout. Network access was disabled during inference.

One real RGB512 image passed Native embedding, all three matched targets, PNG
serialization and clean decoding. All four records were sealed. A second call
performed no embedding or decoding and preserved file hashes. This took about
36 seconds on the local CPU. It is not a Colab timing estimate or a robustness result.
Local evidence is `validation/fnns_loader_fix_20260920/report.json`; the input
hash is the same fixture hash documented below.

Reproduce in a CPU environment with the registered FNNS requirements and cached inputs:

```bash
python scripts/smoke_fnns_bank.py \
  --repository /path/to/pinned/omnisealbench \
  --checkpoint /path/to/decoder_48b.torchscript.pt \
  --image /path/to/rgb512.png \
  --output /path/to/new-smoke-directory
```

The full pretrained DistSeal autoencoder/embedder/detector has **not** been run
locally by this validation. Re-run its five-image Colab canary before treating
that method as validated. No existing scientific result is relabelled as passing
because a loader regression test passes.

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
- Full pretrained DistSeal inference or five-image GPU coverage for the fixed loaders.
- Availability of EditGuard, OmniGuard and SepWater manual checkpoints.
- Licensed MATLAB/toolboxes for Fang or a working Camera-Shooting Instance adapter.
- Detection calibration, robustness under attacks, or research conclusions from COCO100.

These limits are why the distributed notebook starts at five images and reports
incomplete methods explicitly. A catalogue count is never used as a success count.
