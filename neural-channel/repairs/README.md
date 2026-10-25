# Neural channel: scheduler identity recovery

For the later VideoSeal checkpoint-download timeout in the **100-image run**,
use [VideoSeal download recovery](VIDEOSEAL_DOWNLOAD_RECOVERY.md). Its notebook
retains this scheduler repair unchanged and adds only checkpoint transport/cache
handling. The instructions below document the earlier 5-image recovery.

Use `../dist/COCO_Neural_Channel_SDXL_IDENTITY_FIX.ipynb` to resume the existing
neural experiment. This is a narrow compatibility overlay for the original
release, not a new protocol or a replacement set of results.

## What failed

The previous run stopped at the SDXL execution-identity check before generating
the WAM diffusion attacks. The failed worker did not log its newly observed
identity, so the exact historical difference cannot be established from that
traceback alone.

A reproducible cause was found in the pinned Diffusers 0.32.2 library:
`configuration_utils.register_to_config` constructs `_use_default_values` using
`list(set(...))`. Its order depends on Python's hash seed. Two independently
started workers can therefore have identical scheduler parameters but different
JSON lists. The original strict `freeze()` comparison rejects this permutation.

The offline reproduction starts two real Diffusers processes with different
`PYTHONHASHSEED` values. Their lists differ, while their complete canonicalized
configs, 20-step timestep arrays and sigma arrays are identical.

## Exact scope

- Compare only `scheduler_config._use_default_values` as an unordered collection
  of unique strings. Missing members, extra members and malformed lists fail.
- Preserve all other comparisons: weights, libraries, GPU, driver, scheduler
  parameters and every other ordered list must still match.
- Reuse the exact historical execution identity only after this comparison.
  Do not rewrite `_execution/*.json` or invalidate existing package/result hashes.
- Save the actual observed identity and repair provenance under
  `_repairs/scheduler-default-order-v1/`.
- Do not modify attack settings, seeds, batch size, image selection, embedding,
  watermark decoders, metrics, retention policy or the digital-channel run.
- Apply the same narrow guard to scheduler identities for SDXL, instruction
  editing and inpainting. This is the same Diffusers bookkeeping issue, not an
  additional change to any attack. VAE identities without the field are unchanged.
- If any substantive identity difference remains, stop and print its field paths.
  Do not remove execution manifests to force continuation.

The notebook adds one checksum-verified, readable Python module next to the old
sources. The controller forwards that entry point to its isolated workers.
Three existing notebook commands change only their module name; all other
existing code cells remain unchanged. The original `neural-release.json` and
`neural-run.json` identities remain valid.

## Run

1. Upload the new notebook to Colab. Use a separate runtime from any running
   digital experiment. Resume with the same A100-SXM4-80GB execution profile.
   The runtime must still satisfy the original library/driver identity checks.
2. Keep `N_IMAGES = 5`, `MODELS = "all"`, `GPU_BATCH_SIZE = 4`, the original bank
   folder, and `OUTPUT_ROOT = .../watermarking_neural_channel_v1`.
3. Run all cells in order and reconnect Drive. The original source ZIP is normally
   reused from `OUTPUT_ROOT/_release/`. If requested, upload the unchanged
   `COCO_Neural_Channel_RESUMABLE_source.zip`.
4. Completed TrustMark work and committed WAM rows remain reusable. Dependency
   setup and the original input/clean checks can still run; this patch does not
   replace the general resume logic or skip scientific integrity checks.
5. If a permutation is encountered, the log prints `IDENTITY REPAIR ...` and the
   audit-receipt path. A real difference still stops the run with explicit fields.

Do not change the output directory, erase results, or rebuild the original ZIP.
Original source ZIP SHA-256:

```text
5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2
```

## Verification

Validation on 2026-10-11: **41 tests passed** (26 existing tests and 15 repair
regressions). Seven notebook code cells passed AST parsing and the notebook
passed `nbformat` validation. All 92 Python files in the original ZIP parsed.
Every original extracted file retained its checksum after the tests.

Coverage includes actual Diffusers hash-seed reproduction, a small random-weight
CPU VAE, synthetic generation/package reuse, preservation of legacy manifest
bytes and identity digests, strict rejection of real parameter changes, child
entry-point propagation, and idempotent/conflict-safe repair installation.

The tests do **not** constitute an official-weight SDXL/CUDA acceptance run.
No Colab job was started or altered while preparing this notebook.

Reproduce from the pinned CPU diagnostic environment:

```sh
python neural-channel/validate_identity_fix.py
```

This command builds only the recovery notebook, extracts the unchanged base ZIP
into a temporary directory, tests the actual installed overlay, and writes
`../dist/COCO_Neural_Channel_SDXL_IDENTITY_FIX_validation.json`.
Do **not** use `build_notebook.py` or `validate_release.py` for this recovery:
those rebuild the base archive and would give an existing run a different release
identity. The recovery validator requires the original test dependencies,
including `nbformat`, Pillow 10.4.0 and Diffusers 0.32.2; it downloads no weights.
