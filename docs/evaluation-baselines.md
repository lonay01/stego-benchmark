# Author references, clean decoding, and channel degradation

This policy applies to the digital, neural, and physical channel protocols. It separates three questions: what the authors reported, what the selected implementation achieves in our setup without attacks, and how much a channel changes that result. These are complementary measurements, not interchangeable baselines.

## 1. Published reference results

For every selected method, include the authors' published embedding PSNR and clean decoding BER where available. Record the paper version, table or section, link, dataset, sample size, image resolution, payload length, checkpoint variant, embedding strength, and raw-bit or post-ECC scoring interface. Record saving and preprocessing rules where reported; mark unknown details explicitly.

- Distinguish paper results from author-repository examples and third-party reproductions.
- A result measured after JPEG, editing, or photography is not a clean BER reference.
- If only bit accuracy is reported, BER may be derived as 100 minus accuracy in percent, retaining the published precision and marking the conversion. Rounded 100% accuracy is not proof of zero errors in the underlying experiment.
- Message accuracy, detection TPR, and post-ECC success cannot be converted into raw BER.
- Missing values are `not reported` or `not verified`, never zero.

Label each comparison as either **matched experimental conditions**, **partially comparable**, or **context only**, with the relevant differences stated. Matching conditions alone does not establish successful reproduction. Different datasets, payloads, checkpoints, resolutions, quality settings, and metric definitions can explain differences; a gap is not automatically an implementation defect or an architectural weakness. Do not tune on test outcomes just to reproduce a published number.

## 2. Our clean baseline

Decode every saved marked PNG directly, without channel transformations. Use the same decoder, key state, full expected message, and scoring interface that will be used after attacks. Reload the PNG from disk: decoding the encoder's floating-point tensor is a separate diagnostic, not the clean baseline for a PNG-based experiment.

Maintain an independent clean baseline for Native and for each successful Matched Quality target. The latter includes any errors introduced by scalar matching and 8-bit serialization. Do not use Native clean BER as the baseline for a matched-quality attack.

The pairing identity contains the bank version, method/checkpoint and decoder configuration, source/crop identity, image ID, message replicate, payload interface, embedding mode, quality target, and marked-file hash. Reuse a clean result across channels only when these records match. Each attack starts from that exact saved carrier. For physical capture, even the frontal screen photograph is an attacked observation; the clean control remains the digital PNG.

Report clean full-message BER, bit accuracy, exact-message recovery, valid-decoding coverage, and embedding PSNR/SSIM/LPIPS. Use the actual embedding reference crop and dimensions declared for that method. For autoencoder-based methods, source-to-marked distortion remains the embedding quality; reconstruction-only and reconstruction-to-marked measurements may be shown separately but must not replace it.

Keep clean errors visible. Do not discard difficult images, shorten messages, change bit positions, or select checkpoints on the evaluation set to make clean decoding appear perfect.

## 3. Additional degradation after a channel

Report absolute attacked performance **and** its paired change from our clean baseline. Author results remain contextual references and are never the subtraction baseline for channel degradation.

For a case with complete valid decoded messages in both conditions:

    delta_BER_pp = BER_attack_percent - BER_clean_percent

The unit is percentage points. For example, clean BER of 1% and attacked BER of 8% give a degradation of 7 percentage points, not 7% relative degradation. Preserve negative changes: an attack can correct some previously incorrect bits as well as introduce errors. A small delta alone is not good robustness if the clean BER is already high. Do not use BER_attack / BER_clean as the primary measure; it is undefined at zero clean BER.

Also report the loss of exact-message recovery:

    exact_recovery_loss_pp = exact_clean_percent - exact_attack_percent

Use the same eligible cases in both terms. This measure remains payload-length dependent and does not replace BER. As an optional secondary diagnostic, report the fraction of initially exact messages that remain exact after the attack, together with its clean-success denominator. Do not condition the main results on clean success.

An explicit decoder/ECC rejection is unsuccessful exact recovery, but has no BER when it returns no complete message. Physical localisation failures count in end-to-end exact recovery as specified by the physical protocol. Infrastructure failures remain separate. Never invent bit errors for missing strings or treat missing data as successful decoding.

## 4. Pairing and aggregation

For every attack and severity, compute BER differences on the intersection of cases with valid clean and attacked bit strings. Report the paired sample size, the planned eligible size, and missing/rejected/error counts. Show the clean mean on this paired subset alongside the attacked mean and delta; also retain the overall clean summary. A changing subset must not be mistaken for a channel effect.

Average paired differences over attack repeats and messages within each image, then over images, following the channel protocol's weights. Do not subtract averages from different surviving subsets. Compute 95% confidence intervals using 2,000 paired image-level bootstrap resamples with a fixed seed. Carry all associated messages, repeats, modes, and physical-board observations with the sampled image. Repeated messages are not independent images.

Retain each channel's existing family aggregation and coverage rules. If reporting a summary delta, use the same family weights and paired coverage as its attacked score. Native and each matched target remain separate. Do not pool raw and post-ECC interfaces or treat digital, neural, and physical family averages as identical scales.

## 5. Required result tables

Keep two linked tables rather than mixing incompatible conditions in one ranking.

| Reference and clean table | Required contents |
|---|---|
| Configuration | Method, implementation/checkpoint, image dimensions, full payload, raw/ECC interface, mode and target |
| Published reference | Author PSNR and clean BER, exact source, experimental conditions, comparability label |
| Our clean result | Embedding PSNR, SSIM/LPIPS, clean BER, exact recovery, sample size and coverage |
| Interpretation | Known configuration differences and unresolved limitations; no automatic claim of reproduction |

| Channel table | Required contents |
|---|---|
| Condition | Same configuration identity, channel, attack, severity and repeats |
| Absolute recovery | Paired-subset clean BER, attacked BER, clean and attacked exact recovery |
| Degradation | Delta BER in percentage points and exact-recovery loss, with paired confidence intervals |
| Coverage | Planned cases, paired valid cases, rejections, technical failures, matching and capture coverage where applicable |

Save expected and recovered bits, per-case errors and lengths, statuses, pairing identities, and source references so aggregates can be audited. Preserve historical results under their original configurations. Recompute a paired statistic only when the necessary compatible records exist; otherwise mark it unavailable.

This is a reporting and evaluation specification. It does not assert that historical notebooks already calculate these fields or that all author references have been verified. Runtime integration and report validation must be checked before the main channel runs; no new model execution is implied by this documentation update.
