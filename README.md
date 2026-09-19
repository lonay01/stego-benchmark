# stego-benchmark

## Embedding bank

Prepare watermarked images once, then reuse the saved PNGs in channel-specific
experiments. This release saves Native and scalar-only Matched Quality outputs
at 39, 42 and 45 dB, with resumable per-image records.

- [Open the notebook in Colab](https://colab.research.google.com/github/lonay01/stego-benchmark/blob/main/embedding-bank/COCO100_Embedding_Bank_RESUMABLE.ipynb)
- [Notebook](embedding-bank/COCO100_Embedding_Bank_RESUMABLE.ipynb)
- [Setup, output layout and model prerequisites](embedding-bank/README.md)
- [Validation and known limitations](embedding-bank/docs/EMBEDDING_BANK_VALIDATION.md)
- [Downloadable source bundle](embedding-bank/dist/COCO100_Embedding_Bank_source.zip)

Download the notebook **and** source ZIP from the same release. Cell 2 prompts
for the ZIP; no Base64 source is embedded in the notebook. The controller fix
for Camera-Shooting Instance records its blocker and continues the queue.

Start with `N_IMAGES = 5`; increase it to `100` after checking the canary run.
The bank uses **one deterministic 30-bit message per image**, shared across
methods and modes. This differs from the five-message design in the protocol
documents; the two configurations must not be reported as the same experiment.

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
