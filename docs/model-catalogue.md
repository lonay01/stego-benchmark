# Model catalogue and channel focus

This catalogue covers the 36 method entries retained for the benchmark. It separates **what a method was designed for** from **which channels we test it on**. Inclusion here is not evidence of a completed experiment.

## How to read the channel labels

- **Digital:** watermarking under compression, noise, geometric changes, and conventional image or video processing.
- **Physical:** recovery after screen-camera, print-camera, or related optical transmission.
- **Neural:** watermarking during image generation, or robustness to neural regeneration and editing. These are different tasks and are identified below.
- **Digital / Neural:** a mixed scope, typically source tracing and manipulation or deepfake detection. A fragile localization component is not expected to behave like a robust message decoder.

The labels are our working classification of the linked methods, not mutually exclusive claims by their authors. A neural-network encoder is not automatically a *neural-channel* method. Likewise, marking an AI-generated image after generation does not make a method generation-time watermarking.

The same marked image can be tested through digital, physical, and neural channels where the carrier and decoder allow it. The channel label below does not limit those cross-channel tests or predict their outcome.

## 1. Common 30-bit message track: 22 methods

These selected implementations accept a message and an existing image. They can follow the [digital protocol](digital-channel-protocol.md): five common 30-bit messages per image, Native, and scalar Matched Quality at 39, 42, and 45 dB. Longer payloads retain their fixed filling rules; only the common segment is aligned.

This is protocol eligibility, not a claim that all environments have passed the new run. Matching coverage, checkpoint identity, preprocessing, and target-device checks remain required.

| Registry ID | Method / source | Channel focus | Reason and selected scope |
|---|---|---|---|
| `trustmark_q` | [TrustMark-Q](https://github.com/adobe/trustmark) | Digital | General-purpose, resolution-independent image watermarking; Q configuration. |
| `wam` | [WAM](https://github.com/facebookresearch/watermark-anything) | Digital / Neural | Localized messages and partial-image editing, including inpainting; our comparison uses message recovery, not the full localization task. |
| `videoseal` | [VideoSeal](https://github.com/facebookresearch/videoseal) | Digital | Image/video watermarking with robustness to editing and video compression; this benchmark evaluates still images. |
| `pixelseal` | [PixelSeal](https://github.com/facebookresearch/videoseal) | Digital | Robust, imperceptible image/video watermarking; post-hoc embedding. |
| `chunkyseal` | [ChunkySeal](https://github.com/facebookresearch/videoseal) | Digital | High-capacity post-hoc watermarking; the selected payload is 1,024 bits, not 30 total bits. |
| `vine_b` | [VINE-B](https://github.com/Shilin-LU/VINE) | Neural — editing | Uses generative priors to embed a message in an existing image and resist image editing. Not a text-to-image origin method in this benchmark. |
| `vine_r` | [VINE-R](https://github.com/Shilin-LU/VINE) | Neural — editing | A separate VINE checkpoint variant; same post-hoc input/output task. |
| `rosteals` | [RoSteALS](https://github.com/TuBui/RoSteALS) | Digital | Robust message hiding using autoencoder latent space; the selected adapter marks existing images. Using a latent space alone does not imply a neural-editing specialization. |
| `invismark` | [InvisMark](https://github.com/microsoft/InvisMark) | Digital | Post-hoc provenance watermarking for high-resolution images, including AI-generated content; emphasis on robustness to image transformations. |
| `rivagan_benchmark` | [RivaGAN](https://github.com/DAI-Lab/RivaGAN), benchmark variant | Digital | Originally robust video watermarking; our selected 32-bit image implementation is a benchmark variant. |
| `dwt_dct_svd` | [DWT-DCT-SVD implementation](https://github.com/ShieldMnt/invisible-watermark) | Digital | Classical transform-domain baseline. No trained neural checkpoint is required. |
| `mbrs` | [MBRS](https://github.com/jzyustc/MBRS) | Digital | Robustness training with real and simulated JPEG compression. |
| `cin` | [CIN](https://github.com/rmpku/CIN) | Digital | Blind watermarking combining invertible and non-invertible mechanisms, trained with conventional image distortions. |
| `hidden_benchmark` | [HiDDeN implementation](https://github.com/ando-khachatryan/HiDDeN), benchmark variant | Digital | Learned message hiding under simulated distortions; selected combined-noise benchmark checkpoint. |
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

All nine have a **Neural — generation** focus. They create marked images through a generator rather than accepting the same COCO carrier used in the common-message track. Their generated outputs can still undergo the digital, neural, and physical channels.

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

## 3. Candidates with additional requirements: 5 methods

These entries remain in the catalogue but are not counted among the 22 common-message methods above. Restrictions describe the current integration record; they are not claims that the methods are impossible to run or that no weights exist publicly.

| Registry ID | Method / source | Channel focus | Requirement or protocol difference |
|---|---|---|---|
| `editguard` | [EditGuard](https://github.com/xuanyuzhang21/EditGuard) | Digital / Neural | Copyright protection and tamper localization, including AI edits. Reuse and verify the previous digital run's degrade.pth checkpoint for the selected 64-bit copyright component. Localization is a separate task. |
| `omniguard` | [OmniGuard copyright component](https://github.com/xuanyuzhang21/OmniGuard) | Digital / Neural | Hybrid manipulation localization and copyright protection. A matching trained 100-bit copyright checkpoint must be verified; localization weights are not interchangeable. |
| `sepwater` | [SepWater](https://github.com/CVhnu/sepwater) | Physical | Anti-screen-shooting latent-space watermarking. Requires the complete author checkpoint; selected interface uses 100 raw bits. |
| `fang_screen` | [Fang Screen-Shooting Resilient Watermarking](https://github.com/FangHanNUS/Screen-Shooting-Resilient-Watermarking) | Physical | Requires licensed MATLAB and the relevant toolboxes. Supports 30 useful bits with CRC/BCH; coded and uncoded evaluations must be distinguished. |
| `camera_shooting_instance` | [Camera-Shooting Resilient Image Instance](https://github.com/gyz-star/Camera-Shooting-Resilient-Watermarking-on-Image-Instance-Level) | Physical | Checkpoint access, instance-level preprocessing, and payload interface still require validation. |

## Catalogue totals and reporting

| Evaluation group | Entries |
|---|---:|
| Common 30-bit image-message track | 22 |
| Generation-time track | 9 |
| Additional requirements | 5 |
| **Total** | **36** |

These counts describe benchmark entries, not 36 independent architectures or 36 completed runs. VINE-B/VINE-R and other named variants are intentionally separate entries. Channel tags overlap, so they should not be turned into exclusive group counts without an explicit rule.

Source links support the stated research scope. Payload and integration notes describe the selected benchmark configurations, based on the implementation inventory reviewed on 13 September 2026. Execution status and missing-checkpoint gates must be checked against the actual run records before claiming that a method was evaluated. This catalogue does not replace those records or a checkpoint manifest.
