# Validation record

Date: 2026-09-28. Local Windows, Python 3.12, PyTorch 2.5.1 CPU,
torchvision 0.20.1. No remote Colab/GPU execution is claimed.

## Completed

- 12 CPU tests passed, with both optional fixture sets enabled.
- Every distributed Python module parsed and compiled; all four code cells
  compiled and the notebook passed nbformat validation.
- All previous-release `src/`, `registry/` and `locks/` bytes were preserved.
  Frozen base SHA256:
  `1d5c1a7b641e2b0a01d3841d86cb2050a369c680940013cb6a14aebd37ea7f3a`.
- A synthetic historical bank exercised preflight for all 100 planned cases,
  central-crop geometry, unchanged message streams, locked resume, unchanged input
  bank hashes, partial reports and a fresh-process report command.
- Tests reject wrong output folders/scope, changed signatures, source/checkpoint
  hashes and damaged stored artifacts. An interrupted fake worker is retried;
  its subsequently measured nonzero BER is retained without another measurement.
- RoSteALS namespace repair was exercised in fresh processes, including the real
  upstream `ldm.util.instantiate_from_config` resolving a `RoSteALS.*` target.
  The instantiated test class was synthetic, not the real pretrained model.
- Actual source-defined RoSteALS and SepWater residual export expressions were
  executed on CPU tensors and matched the expected pixel arithmetic exactly.

## Real Meta checkpoint smoke

Official repository revision:
`c91217c06e953a4eaf270ef9de4a3939bcea3366`.
Checkpoint SHA256:
`5a83d4d5c85f6d58abef7d1083d559dae1eb9171bf63ca0e0c13016fd4741443`.
Official sample `hidden/imgs/00.png`, Git blob:
`e4e1e91a476f0dacf73e0c2c95dfe33fa140750d`.

The actual encoder/decoder checkpoint loaded with strict state dictionaries and
safe metadata allowlisting. The source sample was resized to RGB512 as in the
official demo, then centrally cropped for the 256 profile. One fixed 48-bit
message from the demo was used. Both profiles were evaluated; neither was selected
as a winner.

| Profile | Raw errors | Clipped errors | Rounded errors | PNG errors | Clean-control mismatches | PNG PSNR |
|---|---:|---:|---:|---:|---:|---:|
| checkpoint_256 | 0/48 | 0/48 | 0/48 | 0/48 | 21/48 | 24.4242 dB |
| demo_512 | 0/48 | 0/48 | 0/48 | 0/48 | 23/48 | 35.1061 dB |

These are smoke results on one author-supplied sample, **not** the five-image COCO
canary, not confidence intervals, and not evidence of relative model quality.
The profiles differ in size, JND and strength; PSNR differences are not isolated
resolution effects. Tests verify valid inference and serialization, not zero BER.

## Not yet executed

- The complete new Colab run on the user's actual five COCO images.
- CUDA execution and first-time Linux dependency resolution for Meta.
- Full pretrained RoSteALS/SepWater neural-core checks after the import repair.
  Their preceding measurements are not rebranded as results of this new release.

The source ZIP records all hashes in `dist/build.json`; its checksum is embedded
in the notebook. Repeat builds are byte-identical. Old diagnostic files,
production adapters and the user's pre-existing working-tree edits were not changed.
