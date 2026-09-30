# Model catalogue and channel focus

This catalogue covers 38 method entries selected for evaluation, including five newly added candidates awaiting integration and smoke tests. It separates **what a method was designed for** from **which channels we test it on**. Inclusion here is not evidence of a completed experiment.

**Scope revision, 29 September 2026:** three blocked entries were removed from active selection. Historical results and frozen releases are retained; this is an integration decision, not a negative measurement of the methods. There are now **29 existing-image entries plus 9 generation-time entries**.

## How to read the channel labels

- **Digital:** watermarking under compression, noise, geometric changes, and conventional image or video processing.
- **Physical:** recovery after screen-camera, print-camera, or related optical transmission.
- **Neural:** watermarking during image generation, or robustness to neural regeneration and editing. These are different tasks and are identified below.
- **Digital / Neural:** a mixed scope, typically source tracing and manipulation or deepfake detection. A fragile localization component is not expected to behave like a robust message decoder.

The labels are our working classification of the linked methods, not mutually exclusive claims by their authors. A neural-network encoder is not automatically a *neural-channel* method. Likewise, marking an AI-generated image after generation does not make a method generation-time watermarking.

The same marked image can be tested through digital, physical, and neural channels where the carrier and decoder allow it. The channel label below does not limit those cross-channel tests or predict their outcome.

## 1. Existing-image track: 22 methods

These selected implementations accept a message and an existing image. They can follow the [digital protocol](digital-channel-protocol.md): five full-payload message replicates per image, Native, and scalar Matched Quality at 39, 42, and 45 dB. Use the full supported variable message length of the selected checkpoint/configuration, without benchmark-added repetition or ECC. Score the entire message and report its length beside BER and image quality. Reserved bits and native ECC are recorded separately; this is not an equal-payload comparison.

This is protocol eligibility, not a claim that all environments have passed the new run. Matching coverage, checkpoint identity, preprocessing, and target-device checks remain required.

| Registry ID | Method / source | Channel focus | Reason and selected scope |
|---|---|---|---|
| `trustmark_q` | [TrustMark-Q](https://github.com/adobe/trustmark) | Digital | General-purpose, resolution-independent image watermarking; Q configuration. |
| `wam` | [WAM](https://github.com/facebookresearch/watermark-anything) | Digital / Neural | Localized messages and partial-image editing, including inpainting; our comparison uses message recovery, not the full localization task. |
| `videoseal` | [VideoSeal](https://github.com/facebookresearch/videoseal) | Digital | Image/video watermarking with robustness to editing and video compression; this benchmark evaluates still images. |
| `pixelseal` | [PixelSeal](https://github.com/facebookresearch/videoseal) | Digital | Robust, imperceptible image/video watermarking; post-hoc embedding. |
| `chunkyseal` | [ChunkySeal](https://github.com/facebookresearch/videoseal) | Digital | High-capacity post-hoc watermarking; the selected payload is 1,024 message bits. The full message is evaluated, not a 30-bit segment. |
| `vine_b` | [VINE-B](https://github.com/Shilin-LU/VINE) | Neural — editing | Uses generative priors to embed a message in an existing image and resist image editing. Not a text-to-image origin method in this benchmark. |
| `vine_r` | [VINE-R](https://github.com/Shilin-LU/VINE) | Neural — editing | A separate VINE checkpoint variant; same post-hoc input/output task. |
| `rosteals` | [RoSteALS](https://github.com/TuBui/RoSteALS) | Digital | Robust message hiding using autoencoder latent space; the selected adapter marks existing images. Using a latent space alone does not imply a neural-editing specialization. |
| `invismark` | [InvisMark](https://github.com/microsoft/InvisMark) | Digital | Post-hoc provenance watermarking for high-resolution images, including AI-generated content; emphasis on robustness to image transformations. |
| `rivagan_benchmark` | [RivaGAN](https://github.com/DAI-Lab/RivaGAN), benchmark variant | Digital | Originally robust video watermarking; our selected 32-bit image implementation is a benchmark variant. |
| `dwt_dct_svd` | [DWT-DCT-SVD implementation](https://github.com/ShieldMnt/invisible-watermark) | Digital | Classical transform-domain baseline. No trained neural checkpoint is required. |
| `mbrs` | [MBRS](https://github.com/jzyustc/MBRS) | Digital | Robustness training with real and simulated JPEG compression. |
| `cin` | [CIN](https://github.com/rmpku/CIN) | Digital | Blind watermarking combining invertible and non-invertible mechanisms, trained with conventional image distortions. |
| `meta_hidden_revisited` | [Meta Revisited HiDDeN](https://github.com/facebookresearch/stable_signature/tree/c91217c06e953a4eaf270ef9de4a3939bcea3366/hidden) | Digital | Official Meta post-hoc encoder/decoder: 48 bits, 256×256, checkpoint scaling 0.3 without JND. Released with Stable Signature (ICCV 2023); neither original 2018 HiDDeN nor generation-time Stable Signature. [Frozen profile and citation](meta-hidden-revisited.md). |
| `arwgan` | [ARWGAN](https://github.com/river-huang/ARWGAN) | Digital | Attention-based robust image watermarking; selected 30-bit encoder/decoder. |
| `ssl_watermarking` | [SSL Watermarking](https://github.com/facebookresearch/ssl_watermarking) | Digital | Image optimization in a self-supervised feature space; this entry uses the multibit interface, not the zero-bit alternative. |
| `maskwm` | [MaskWM](https://github.com/hurunyi/MaskWM) | Digital — local editing | Mask-based watermark localization and extraction; selected global D_32bits checkpoint. Author VAE-finetuned variants are different configurations and are not silently substituted. |
| `robust_wide` | [Robust-Wide](https://github.com/hurunyi/Robust-Wide) | Neural — editing | Designed for robustness against instruction-driven image editing; embeds into an existing image. |
| `sepmark` | [SepMark](https://github.com/sh1newu/SepMark) | Digital / Neural | Source tracing and deepfake detection. The robust tracing decoder is the primary message decoder; the fragile decoder is reported separately. |
| `stegastamp` | [StegaStamp](https://github.com/tancik/StegaStamp), ONNX benchmark variant | Physical | Print-and-photograph recovery. The selected ONNX implementation uses raw 100-bit messages, not the author's text-plus-ECC interface. |
| `pimog` | [PIMoG](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw) | Physical | Screen-camera watermarking with a simulated screen-shooting noise layer. |
| `ssrh` | [SSRH](https://github.com/Dawn-Hsiao/SSRH) | Physical | Screen-shooting-resilient image watermarking. |

The legacy adapters and StegaStamp ONNX use the pinned [WIBE implementation](https://github.com/ispras/wibe/tree/acf94ed81db4243d22fb7a7e075aff1671bc58e0). A link to an author's repository does not establish checkpoint equivalence. Fixed-resolution adapters, raw/ECC choices, and precision changes must be disclosed in each run.

## 2. Generation-time track: 9 methods

All nine have a **Neural — generation** focus. They create marked images through a generator rather than accepting the same COCO carrier used in the existing-image track. Their generated outputs can still undergo the digital, neural, and physical channels.

In this benchmark they use Native mode and a separate generation/control protocol. Residual matching against an unrelated COCO image is not used. Payload descriptions below refer to the selected adapters, not every possible variant of each method.

| Registry ID | Method / source | Channel focus | Selected message or detection interface |
|---|---|---|---|
| `tree_ring` | [Tree-Ring](https://github.com/YuxinWenRick/tree-ring-watermark) | Neural — generation | Keyed latent-noise watermark; detection score, not arbitrary 30-bit recovery. |
| `gaussian_shading` | [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading) | Neural — generation | Selected ChaCha20 variant carries 256 bits; its generated carrier still differs from the common COCO input. |
| `stable_signature` | [Stable Signature](https://github.com/facebookresearch/stable_signature) | Neural — generation | Published SD2 decoder with a fixed 48-bit signature; five arbitrary messages cannot be assigned to this checkpoint. |
| `ringid` | [RingID](https://github.com/showlab/RingID) | Neural — generation | Selected verification interface returns a score. |
| `maxsive` | [MaXsive](https://github.com/Mao718/MaXsive) | Neural — generation | Selected template/correlation interface returns a score. |
| `metr` | [METR](https://github.com/deepvk/metr) | Neural — generation | Selected base configuration uses 10 binary rings / 10 bits. |
| `prc` | [PRC Watermark](https://github.com/XuandongZhao/PRC-Watermark) | Neural — generation | Selected presence-verification interface returns a decision, not an arbitrary 30-bit message. |
| `indexmark` | [IndexMark](https://github.com/maifoundations/IndexMark) | Neural — generation | Autoregressive image generation; selected detector uses a green-token score. |
| `bitmark` | [BitMark](https://github.com/sprintml/BitMark) | Neural — generation | Bitwise autoregressive generation; selected interface returns a detection score, despite “bit” in the method name. |

Generation-time embedding and resilience to neural editing are separate classification axes. For example, VINE and Robust-Wide belong to the common-image track even though their focus is neural editing.

## 3. Candidates with additional requirements: 2 methods

These entries remain in the catalogue but are listed separately from the 22 existing-image methods above. Together with the five new additions below, they bring the planned existing-image protocol set to 29. Restrictions describe the integration record; they are not claims that the methods are impossible to run or that no weights exist publicly.

| Registry ID | Method / source | Channel focus | Requirement or protocol difference |
|---|---|---|---|
| `editguard` | [EditGuard](https://github.com/xuanyuzhang21/EditGuard) | Digital / Neural | Copyright protection and tamper localization, including AI edits. Reuse and verify the previous digital run's degrade.pth checkpoint for the selected 64-bit copyright component. Localization is a separate task. |
| `sepwater` | [SepWater](https://github.com/CVhnu/sepwater) | Physical | Anti-screen-shooting latent-space watermarking. Requires the complete author checkpoint; selected interface uses 100 raw bits. |

## 4. New additions awaiting integration: 5 entries

These entries are selected for the existing-image, full-payload pipeline: an image and message enter the embedder; the resulting image is evaluated in Native or scalar Matched Quality mode before channel attacks and decoding. Source code, published checkpoint references, and message interfaces have been inspected. Integration status below reflects the inventory review dated at the end of this document, not a new execution audit. The IDs are catalogue identifiers, not evidence that an adapter has passed validation.

| Catalogue ID | Method / source | Channel focus | Selected configuration and integration checks |
|---|---|---|---|
| `trustmark_b` | [TrustMark-B](https://github.com/adobe/trustmark) | Digital | Published B encoder/decoder, 100 raw bits. Use the same full-message assignment and raw/ECC convention as TrustMark-Q. The authors describe B as favouring robustness over quality relative to Q. This is another checkpoint variant of TrustMark, not an independent architecture. |
| `trustmark_p` | [TrustMark-P](https://github.com/adobe/trustmark) | Digital | Published P encoder/decoder, 100 raw bits. The authors describe P as prioritising visual quality. Preserve and record its native preprocessing and strength settings rather than assuming they equal Q. Use the same full-message assignment as Q; this is a TrustMark variant. |
| `fin` | [FIN — AAAI 2023](https://github.com/QQiuyp/FIN) | Digital | Flow-based robust watermarking with an invertible noise layer; 64-bit message. [HEAVY checkpoints](https://github.com/QQiuyp/FIN/tree/main/experiments/HEAVY) provide FED and INL. The official demonstration resizes inputs to 128 pixels on the shorter side. Select and pin the checkpoint variant, validate the 512×512 carrier adaptation, and keep any demonstration-time attack outside embedding so that the benchmark alone controls distortions. |
| `fnns_omniseal` | [FNNS, OmniSealBench implementation](https://github.com/facebookresearch/omnisealbench/blob/main/src/omnisealbench/models/fnns.py) | Digital | Optimises an image perturbation against a fixed decoder. The [model card](https://github.com/facebookresearch/omnisealbench/blob/main/src/omnisealbench/cards/fnns.yaml) specifies 48 output bits, 16 leading detection bits, and 10 optimisation steps. The remaining 32 positions form the full variable message: L_i = 32, not 48. Pin the detection prefix, bit order, optimiser settings, and decoder checkpoint. Score all 32 message bits; report detection-prefix recovery separately. Report this benchmark implementation explicitly, not as an unchanged reproduction of the original FNNS setup. |
| `distseal_posthoc_dcae` | [DistSeal, post-hoc DC-AE](https://github.com/facebookresearch/distseal) | Neural — latent-space embedding | Select the existing-image post-hoc mode, not generation-time watermarking. The [published configuration](https://github.com/facebookresearch/distseal/blob/main/configs/training/diffusion/posthoc_latent.yaml) specifies 64 bits and 512×512 images; the [inference notebook](https://github.com/facebookresearch/distseal/blob/main/notebooks/posthoc_dcae.ipynb) embeds a supplied message into autoencoder latents and decodes the marked image. Validate checkpoint dimensions and record the DC-AE and watermarking weights. Latent embedding alone does not establish robustness to neural editing. |

TrustMark's [implementation](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py) lists the variant-specific checkpoint files and their checksums. OmniSealBench supplies a [checkpoint download script](https://github.com/facebookresearch/omnisealbench/blob/main/scripts/download_image_models.sh) for the FNNS decoder; that repository is archived, so the integration must pin a revision. DistSeal publishes its watermarking checkpoint reference in the [DC-AE model card](https://github.com/facebookresearch/distseal/blob/main/cards/detector_dc-ae.yaml).

For DistSeal, both Native quality and Matched Quality are measured against the original COCO image, not its unmarked autoencoder reconstruction. Reconstruction distortion is part of the total image change. For every addition, message recovery after PNG saving, fixed-resolution handling, and achievable PSNR coverage must be verified before a full run. A compatible interface does not guarantee successful decoding or attainment of all PSNR targets.

## Catalogue totals and reporting

| Evaluation group | Entries |
|---|---:|
| Existing-image track | 22 |
| Generation-time track | 9 |
| Additional requirements | 2 |
| New additions awaiting integration | 5 |
| **Total** | **38** |

These counts describe benchmark entries, not 38 independent architectures or 38 completed runs. VINE-B/VINE-R and TrustMark-Q/B/P are intentionally separate entries within their respective families. Channel tags overlap, so they should not be turned into exclusive group counts without an explicit rule.

Source links support the stated research scope. Payload and integration notes describe the selected benchmark configurations, based on the implementation inventory reviewed on 13 September 2026. Execution status and missing-checkpoint gates must be checked against the actual run records before claiming that a method was evaluated. This catalogue does not replace those records or a checkpoint manifest.

The payload policy was revised on 21 September 2026 to match the three channel protocols. This changes the planned message assignment and scoring, not model membership or the recorded execution status. Listed payload lengths are configuration-specific and still require interface validation; they are not claims about the global maximum capacity of an architecture.
