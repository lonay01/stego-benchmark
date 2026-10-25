# VideoSeal download recovery

Notebook: `../dist/COCO_Neural_Channel_VIDEOSEAL_DOWNLOAD_FIX.ipynb`.

The 100-image neural run stopped during VideoSeal preparation on 22 October
2026. WIBE's Nextcloud ZIP request timed out. This is a checkpoint transport
failure, not a failed watermark decode or a GPU out-of-memory error.

## Scope

Only VideoSeal's `prepare` stage is intercepted. Before the original stage runs,
the recovery places the checkpoint in the existing WIBE required-file location:
`assets/videoseal/model_files/videoseal/y_256b_img.pth` inside its temporary slot.
The unchanged adapter then loads that file without downloading the Nextcloud ZIP.

The saved bank's `videoseal/_meta/model_identity.json` is required. Its complete
registry entry must match the frozen production catalogue, and its checkpoint
SHA-256 must be:

```text
3d2ff2523d2a89e3532c6dfdcf693098799326e9b3e74c185e815e9baa8340a3
```

Download order:

1. [Meta's official checkpoint](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth),
   as listed in the [pinned VideoSeal model zoo](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/README.md#model-zoo).
2. A [revision-pinned Hugging Face mirror](https://huggingface.co/Svetozar1993/videoseal-1.0/blob/35116c2e949486b08c2632df4ad8f73f9b177b55/y_256b_img.pth).
   This is a third-party mirror, not a different model or a new authority for
   weights. Its downloaded bytes must match the pre-existing bank hash.

There are at most three network attempts: Meta once, then the mirror twice.
TLS verification remains enabled. Transfers are streamed, have socket timeouts
and respect the session deadline. Partial downloads never become final files.
A complete response with a wrong hash stops immediately.

The verified checkpoint is cached durably under:

```text
OUTPUT_ROOT/_repairs/videoseal-download-v1/assets/<sha256>/y_256b_img.pth
```

This requires about 218 MiB on Drive, plus temporary copying space. The cache
survives normal model-slot cleanup. A later runtime verifies the cache and copies
it locally without downloading again. Existing conflicting files are not replaced.
Repair receipts are saved under `_repairs/videoseal-download-v1/receipts/` and
the chosen download URL is printed in the persistent preparation log.

No changes are made to the bank, payloads, decoder code, model card, dependencies,
attack settings, seeds, metrics, results, execution manifests or pending attack
ZIPs. The original strict inference/dependency/geometry/Clean checks still run.
The frozen release already excludes the narrowly identified transport ZIP from
VideoSeal's inference identity once the extracted checkpoint is present; the
repair does not add another exception to that comparison.

The earlier SDXL default-key-order overlay is included byte-for-byte. It remains
active in workers. The original release ZIP is not rebuilt or replaced.

## Resume

1. Upload the new notebook to Colab. Use the same A100-SXM4-80GB profile and a
   runtime separate from any running digital experiment.
2. Keep `N_IMAGES = 100`, `MODELS = "all"`, `GPU_BATCH_SIZE = 4`, and the existing
   bank/output paths. Use 5 only when intentionally selecting the earlier canary.
3. Run all cells in order and authorize Drive mounting. The old source ZIP is
   reused from `OUTPUT_ROOT/_release/`; upload the unchanged
   `COCO_Neural_Channel_RESUMABLE_source.zip` only if requested.
4. Check for `VIDEOSEAL CHECKPOINT VERIFIED` in the preparation log. The original
   controller then reuses committed rows and valid cached attacks and continues
   the remaining work. Environment setup and scientific integrity checks remain.

Do not delete the output folder, change the batch size or run two controllers
against the same results. A real dependency/GPU identity change must still stop.

## Validation

Validation on 23 October 2026 (Europe/Moscow): **60 tests passed**. All eight
notebook code cells passed static parsing and the notebook passed `nbformat`
validation. Every original extracted release file retained its checksum.

The official Meta endpoint responded but stalled in this local network check.
The pinned mirror was downloaded completely: **228,351,471 bytes**, with the exact
SHA-256 recorded above. Downloaded checkpoint contents were not unpickled.

Reproduce the static and offline checks:

```sh
python neural-channel/validate_videoseal_fix.py
# Optionally test the actual checkpoint bytes through durable cache + new slot:
python neural-channel/validate_videoseal_fix.py --checkpoint /path/to/y_256b_img.pth
```

The validator checks every notebook cell, all 92 original Python source files,
the unchanged source ZIP and prior notebook, checksum-guard propagation, the
actual saved registry, network failures, mirror hash rejection, cache reuse,
the frozen WIBE adapter's download-skip branch, and prior package/identity tests.
The JSON report is beside the generated notebook. No GPU acceptance run is
claimed; a Colab resume must still pass the original clean controls.
