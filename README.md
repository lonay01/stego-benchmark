# stego-benchmark

## Current notebooks — October 2026

These are the current local release artifacts, published without rebuilding their
frozen source archives. The notebooks are experiment runners, not evidence that
every method has completed the full evaluation. Some recovery notebooks retain
`N_IMAGES = 100`; inspect the settings and use 5 for an intentional canary.

| Task | Notebook | Required files and notes |
|---|---|---|
| Create the embedding bank and decode Clean | [Production bank](embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION.ipynb) | [Source ZIP](embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION_source.zip), [setup](embedding-bank/production/README.md) |
| Resume the two affected bank methods | [Meta HiDDeN / FNNS recovery](embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION_HIDDEN_FNNS_FIX.ipynb) | Same production ZIP plus [helper](embedding-bank/production/dist/hotfix_hidden_fnns.py); [scope](embedding-bank/production/HOTFIX_HIDDEN_FNNS.md). Defaults to these two methods, not all 29. |
| Digital attacks and decoding | [Parallel digital runner with lease/I/O recovery](digital-channel/dist/COCO_Digital_Channel_LEASE_IO_FIX.ipynb) | [Source ZIP](digital-channel/dist/COCO_Digital_Channel_LEASE_IO_FIX_source.zip), [execution/resume notes](digital-channel/performance/README.md), [latest repair](digital-channel/performance/LEASE_IO_FIX.md) |
| Neural attacks and decoding | [Neural runner with VideoSeal download recovery](neural-channel/dist/COCO_Neural_Channel_VIDEOSEAL_DOWNLOAD_FIX.ipynb) | [Unchanged base source ZIP](neural-channel/dist/COCO_Neural_Channel_RESUMABLE_source.zip), [setup](neural-channel/README.md), [latest repair](neural-channel/repairs/VIDEOSEAL_DOWNLOAD_RECOVERY.md) |
| Check neural-channel input/output handling | [InvisMark / ChunkySeal diagnostics](neural-channel/two-method-diagnostics/dist/InvisMark_ChunkySeal_NEURAL_DIAGNOSTICS.ipynb) | [Source ZIP](neural-channel/two-method-diagnostics/dist/InvisMark_ChunkySeal_NEURAL_DIAGNOSTICS_source.zip), [diagnostic design](neural-channel/two-method-diagnostics/README.md) |
| Summarize recorded computation | [Resource audit](resource-audit/dist/BENCHMARK_Resource_Audit.ipynb) | Self-contained, CPU-only; [definitions and limits](resource-audit/README.md). Recorded stage time is not a billing statement or training time. |

Keep the matching source ZIP beside each notebook or upload it when prompted.
Existing runs must retain their original source identity, bank and result paths.
Do not start a second writer against an active experiment. Model weights, private
Drive results and executed notebook outputs are not included in this publication.

## Reviewed embedding bank (1 October 2026)

Prepare the 29 selected existing-image configurations with five full messages per
image, native carrier sizes, Native and Matched PSNR39/42/45. Every saved PNG is
decoded without channel attacks. Start at `N_IMAGES = 5`; after review, change it
to `100` and reuse the same persistent bank.

- [Notebook](embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION.ipynb)
- [Matching readable source ZIP](embedding-bank/production/dist/COCO_Embedding_Bank_PRODUCTION_source.zip)
- [Settings, selected profiles, output layout and resume](embedding-bank/production/README.md)
- [Local validation and GPU verification boundary](embedding-bank/production/VALIDATION.md)

This supersedes the historical common30 notebook for new bank creation. Earlier
notebooks and results below retain their original scope and configuration.

## Evaluation baselines

All three channels report [published author references, our clean decoding, and paired degradation after attacks](docs/evaluation-baselines.md). Report absolute recovery as well as additional BER in percentage points. Each embedding mode has its own clean baseline from the exact saved PNG; author results retain their original experimental conditions.

## Current review, 29 September 2026

The active registry contains **29 existing-image entries**, with Meta Revisited HiDDeN in place of the legacy baseline. Three blocked entries have been removed from active selection. Historical notebooks below retain their original inventories; they do not migrate existing banks.

- [Four-method diagnostic and launch instructions](embedding-bank/four-method-review/README.md)
- [Diagnostic notebook](embedding-bank/four-method-review/dist/Four_Method_QUALITY_CHECK.ipynb)
- [Matching readable source ZIP](embedding-bank/four-method-review/dist/Four_Method_QUALITY_CHECK_source.zip)

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

Historical releases used a 32-entry inventory; the current catalogue has 29 existing-image entries. Historical blocker records remain evidence, not active work items. The validation record separates
offline tests from actual pretrained-model smoke tests. Model weights, datasets,
generated banks and credentials are not included. Upstream code and weights
remain subject to their own licenses; see [third-party notes](embedding-bank/THIRD_PARTY.md).

## Protocols

### Digital channel runner (production embedding bank)

- [Resumable notebook](digital-channel/dist/COCO_Digital_Channel_RESUMABLE.ipynb)
- [Matching source ZIP](digital-channel/dist/COCO_Digital_Channel_RESUMABLE_source.zip)
- [Setup, storage lifecycle and exact attack settings](digital-channel/README.md)
- [Validation boundary](digital-channel/VALIDATION.md)

Consumes the completed 29-method production bank without re-embedding. Per
method: save all pending attacks in temporary ZIPs, decode saved PNGs, commit
results to persistent storage, then remove only the disposable attack packages.
Start with five images; increase to 100 after reviewing the canary. Offline/CPU
checks do not replace the first full GPU run.

- [Digital channel: Native and Matched Quality](docs/digital-channel-protocol.md)
- [Neural channel: Native and Matched Quality](docs/neural-channel-protocol.md)
- [Physical channel: screen–camera, Native and Matched Quality](docs/physical-channel-protocol.md)

## Methods

- [Model catalogue and channel focus](docs/model-catalogue.md)
