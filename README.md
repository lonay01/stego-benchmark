# stego-benchmark

## Evaluation baselines

All three channels report [published author references, our clean decoding, and paired degradation after attacks](docs/evaluation-baselines.md). Report absolute recovery as well as additional BER in percentage points. Each embedding mode has its own clean baseline from the exact saved PNG; author results retain their original experimental conditions.

## Full-payload canary (new)

Test the 32 catalogue entries on five historical COCO images and five full-length
messages per image. Decode Native and Matched PSNR39/42/45 PNGs, keep per-task
checkpoints on Drive, and distinguish bit errors from environment/checkpoint
blockers. This is a separate canary, not an automatic migration of the old bank.

- [Notebook](embedding-bank/full-payload/dist/Embedding_Bank_FULL_PAYLOAD_CANARY.ipynb)
- [Matching source ZIP](embedding-bank/full-payload/dist/Embedding_Bank_FULL_PAYLOAD_CANARY_source.zip)
- [Run instructions and payload contracts](embedding-bank/full-payload/README.md)
- [Executed checks and limitations](embedding-bank/full-payload/VALIDATION.md)

Download the notebook and ZIP together; upload the notebook through Colab's
File → Upload notebook menu and supply the ZIP when prompted.
The canary has offline and limited real
CPU smoke validation; the full 32-method GPU run is still to be performed.
Do not start the 100-image bank until this canary has been reviewed.

## Historical embedding bank (common30)

Prepare watermarked images once, then reuse the saved PNGs in channel-specific
experiments. This release saves Native and scalar-only Matched Quality outputs
at 39, 42 and 45 dB, with resumable per-image records.

- [Open the notebook in Colab](https://colab.research.google.com/github/lonay01/stego-benchmark/blob/main/embedding-bank/COCO100_Embedding_Bank_RESUMABLE.ipynb)
- [Notebook](embedding-bank/COCO100_Embedding_Bank_RESUMABLE.ipynb)
- [Setup, output layout and model prerequisites](embedding-bank/README.md)
- [Validation and known limitations](embedding-bank/docs/EMBEDDING_BANK_VALIDATION.md)
- [Downloadable source bundle](embedding-bank/dist/COCO100_Embedding_Bank_source.zip)

Download the notebook **and** source ZIP from the same release. Cell 2 prompts
for the ZIP; no Base64 source is embedded in the notebook. The latest loader fix
supports FNNS TorchScript parameter freezing and strictly loads DistSeal's
inference weights without its unused training-autoencoder subtree. Audited
resume keeps previously saved results; see the validation record for limits.

Start with `N_IMAGES = 5`; increase it to `100` after checking the canary run.
The bank uses **one deterministic 30-bit message per image**, shared across
methods and modes. The revised protocols instead specify **five random messages
at each method's full supported payload length**, with full-message scoring.
**The released notebook has not yet been migrated to this payload policy.**
The two configurations must not be reported as the same experiment. Existing
results remain historical records; changed message assignments require a new
versioned embedding bank rather than overwriting previous artifacts.

The catalogue contains 32 methods, not 32 GPU-validated implementations.
Some methods need manually supplied checkpoints or MATLAB; Camera-Shooting
Instance remains blocked pending integration. The validation record separates
offline tests from actual pretrained-model smoke tests. Model weights, datasets,
generated banks and credentials are not included. Upstream code and weights
remain subject to their own licenses; see [third-party notes](embedding-bank/THIRD_PARTY.md).

## Protocols

- [Digital channel: Native and Matched Quality](docs/digital-channel-protocol.md)
- [Neural channel: Native and Matched Quality](docs/neural-channel-protocol.md)
- [Physical channel: screen–camera, Native and Matched Quality](docs/physical-channel-protocol.md)

## Methods

- [Model catalogue and channel focus](docs/model-catalogue.md)
