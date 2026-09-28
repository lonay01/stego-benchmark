# Meta Revisited HiDDeN

From 29 September 2026, the selected existing-image baseline is **Meta Revisited
HiDDeN**, registry ID `meta_hidden_revisited`. It replaces `hidden_benchmark` in
the current 32-entry catalogue; it is not a 33rd method or a relabelling of the
old measurements.

## What to cite

This is Meta's [Revisited HiDDeN module](https://github.com/facebookresearch/stable_signature/tree/c91217c06e953a4eaf270ef9de4a3939bcea3366/hidden),
distributed with [The Stable Signature: Rooting Watermarks in Latent Diffusion
Models (ICCV 2023)](https://arxiv.org/abs/2303.15435). The module's README asks
users to cite that work. Cite the original [HiDDeN paper](https://arxiv.org/abs/1807.09937)
for the architectural lineage, not as the source of these weights.

Meta removes adversarial training and the image loss and uses stop-gradient JPEG
augmentation. We evaluate the resulting **post-hoc image encoder and decoder**.
We do not fine-tune a diffusion decoder or claim to evaluate generation-time
Stable Signature here. That remains a separate catalogue entry.

Suggested methods wording:

> We evaluate Meta's Revisited HiDDeN implementation released with Stable
> Signature, using the published 48-bit checkpoint and its stored inference
> settings. This configuration is distinct from the original HiDDeN implementation.

## Frozen configuration

| Setting | Selected value |
|---|---|
| Source revision | `c91217c06e953a4eaf270ef9de4a3939bcea3366` |
| Checkpoint | `hidden/ckpts/hidden_replicate.pth` |
| Checkpoint SHA256 | `5a83d4d5c85f6d58abef7d1083d559dae1eb9171bf63ca0e0c13016fd4741443` |
| Profile | `checkpoint_256` |
| Carrier | Central 256×256 pixel crop of the frozen RGB512 COCO source; no resize |
| Variable message | All 48 bits; no reserved positions, repetition or external ECC |
| Encoder / decoder | 4 / 8 blocks; 64 channels |
| Image / watermark scaling | 1.0 / 0.3 |
| JND / channel scaling | Disabled / disabled |
| Input / bit mapping | ImageNet RGB normalization; binary message mapped to −1/+1 |
| Bit decision | Decoder logit strictly greater than zero |
| Saved carrier | RGB8 PNG; Meta demo's clip → round → ToPILImage sequence |
| Compatibility environment | Isolated Python 3.10.12, PyTorch 2.5.1, torchvision 0.20.1, NumPy 1.26.4, Pillow 10.4.0 |

This environment is a tested compatibility choice, not the authors' original
training environment. Resolve and preserve its dependency lock on first setup.
Source files and checkpoint are downloaded from the pinned revision and checked
before execution. Only the unchanged inference classes are loaded; no training,
ViT or diffusion pipeline is installed. Upstream source and weights remain under
their [CC BY-NC 4.0 license](https://github.com/facebookresearch/stable_signature/blob/c91217c06e953a4eaf270ef9de4a3939bcea3366/LICENSE),
not this repository's MIT license.

Five full-length messages per image follow the channel protocols. Native uses
the settings above. Matched Quality uses the same scalar-only rule and targets
as the other methods, comparing against the **same 256×256 original crop**.
The decoder is unchanged between modes. A failure to recover all 48 bits is a
measurement, not a reason to select a different message or strength.

Meta's `demo_512` configuration enables JND and uses watermark scaling 1.5. It is
not the chosen default. If reported, it must appear as a separate configuration;
its scores cannot be combined with `checkpoint_256`. The checkpoint profile was
chosen for configuration provenance, not as a best-of-two leaderboard entry.

## Historical results and notebooks

- Keep `hidden_benchmark` results under their original name and configuration.
  They came from a third-party PyTorch implementation, not the original authors'
  Torch7 checkpoint and not Meta's checkpoint.
- Do not rename old result folders, replace saved bits, or copy their recovery
  scores into the Meta row. There is deliberately no compatible-resume migration
  from that method to this one. Use a new versioned result root.
- Already generated diagnostic notebooks/source ZIPs are frozen evidence. They
  are not silently edited by changing the live registry. Their old results remain
  readable under the releases that created them.
- The root embedding-bank notebook is the historical **common30, one-message**
  workflow. Its source controller now supports the declared 256-pixel carrier,
  but that does not turn its old message protocol into the full-payload study.
  Use a regenerated native/full-payload release for the next experiment, with
  `native_size=256` honoured by its carrier builder. Do not start the 100-image run
  from an old notebook on the assumption that the method changed automatically.

The earlier Meta diagnostic measured 25/25 exact PNG messages in `checkpoint_256`
on five COCO images × five messages, at mean PSNR approximately 24.85 dB.
The separate demo profile measured 21/25 exact messages, BER 0.3333%, PSNR 36.41 dB.
These are small clean-channel diagnostics, not a reproduction of the paper's
full evaluation or proof of robustness after channel attacks.

## Local adapter checks

Run from `embedding-bank`, with `src` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -p 'test_meta_hidden.py' -v
```

The real pretrained smoke is enabled with `WMBENCH_META_FIXTURES` pointing to a
folder containing the pinned `models.py`, `attenuations.py`, checkpoint and the
official `hidden/imgs/00.png` sample saved as `demo.png`. It checks PNG pixels,
48-bit blind extraction, the 256-pixel crop, Matched PSNR against that crop and
resume. Quality is recorded, not forced to match a paper number. No remote GPU
or full 100-image execution is implied by these checks.

Validation on 29 September 2026: six Meta-specific tests passed, including real
pretrained CPU inference, plus 68 existing bank/recovery tests. On the official
sample and its fixed 48-bit demo message, the saved PNG had 0/48 bit errors and
PSNR 24.4242 dB. This smoke used local Python 3.12 / torch 2.5.1+cpu, not the
target Colab environment. Modified Python modules compiled successfully. No
production notebook or historical archive was regenerated in this change.
