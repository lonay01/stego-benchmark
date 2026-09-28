# Five-method upstream reference checks

This is a separate diagnostic, not a new production bank. It does not edit the
previous canary, previous diagnostics, model adapters, messages or checkpoints.

## Run in Colab

1. Upload `dist/Embedding_Bank_AUTHOR_REFERENCE.ipynb` to Colab.
2. Select A100 and high system RAM, matching the prior diagnostic. This is an
   execution recommendation, not a claim that this release has passed GPU tests.
3. Run all four code cells. Authorize Drive and upload the matching
   `Embedding_Bank_AUTHOR_REFERENCE_source.zip` when prompted.
4. Leave `N_IMAGES=5` and all five model IDs. Correct only paths if necessary.

Defaults:

| Setting | Path |
|---|---|
| Read-only native bank | `/content/drive/MyDrive/watermarking_native_resolution_canary_v1` |
| New durable output | `/content/drive/MyDrive/watermarking_author_reference_v1` |
| Temporary installations | `/content/watermarking_author_reference_scratch_v1` |
| Report | `<output>/_reports/REPORT.md` |
| Review ZIP | `<output>/author_reference_report.zip` |

The input bank must contain `trial.json`, `_inputs/manifest.json`, the first ten
original RGB512 PNGs, and the five methods' historical native PNG/metadata and
environment locks. A downloaded report ZIP alone is not sufficient. No substitute
COCO images are downloaded. SepWater's verified checkpoint is reused from
`<bank>/_assets` or the earlier recovery asset folders; the previously audited
author download is available if it is missing.

One model environment is installed at a time. Each phase runs in a fresh process;
the reference process never instantiates a benchmark adapter. After a successful
model, its owned scratch installation is removed. Drive results are retained.
Reserve at least 30 GiB free scratch before each method; the controller stops
below 20 GiB and caps a slot at 100 GiB. Failed slots are preserved for diagnosis.
The six-hour session budget is a stop limit, not an ETA or a Colab uptime promise.

On reconnect, run cells 1–3 again with the same settings. Completed measurements,
including nonzero BER, are reused after artifact verification. Technical error
records are archived before retry. Corrupted saved artifacts are not overwritten.
Keep the same GPU and model runtime for each paired comparison; a runtime change
requires review, not removal of a safety check. Stop other runners before starting.

## What is measured

| Methods | Images × full messages | Check |
|---|---:|---|
| HiDDeN, RoSteALS, SepWater | Historical 5 × 5 each | Fresh upstream neural core versus frozen adapter; cross-decoding |
| SepMark, DistSeal post-hoc DC-AE | Historical 5 × 5 each | Reproduce adapter output and run source-defined PNG export |
| SepMark, DistSeal post-hoc DC-AE | Next manifest 5 × 5 each | Same fixed serializer policy on different images |

Total: 175 image/message cases, not 175 independent images. Historical results
are separated from next-five validation. The latter is not a claim that images
were absent from upstream training data. Messages use all selected information
positions: 30, 100, 100, 128 and 64 bits respectively. Native crops remain
128, 256, 224, 256 and 512 pixels. No resampling of the frozen central crops.

For the first three methods:

1. Re-embed and decode with the frozen adapter; retain raw float output and PNG.
2. Reload upstream classes in a fresh process, independently load the same
   checkpoint and compare all inference weight/buffer hashes.
3. Encode the identical crop/message/seed through the upstream neural core.
4. Decode both branches' raw pixels and both PNGs with both decoders. Compare
   complete bit vectors and logits, not just aggregate BER.
5. Record tensor differences with fixed `atol=1e-6`, `rtol=1e-5`; keep exact
   equality and maximum/mean differences as separate fields.

`core_agreement` requires weight equality, float-embedding agreement and all
same-pixel decoder checks. It does **not** require zero BER and does not certify
paper-level performance. A different PNG serializer can coexist with matching
neural cores. `review_difference` is evidence to inspect, not permission to tune
thresholds, shorten payloads or select a best export per message.

## Source-defined exports

| Method | Selected source path | Policy |
|---|---|---|
| HiDDeN | `utils.tensor_to_image` | Clip then uint8 cast; no display montage |
| RoSteALS | `inference.py` residual export | Author float64 source + float32 residual, clip, uint8 cast |
| SepWater | `embed.py` residual export | Same author expression; full raw100 replaces text/BCH |
| SepMark | First `single_image` expression in `test_Dual_Mark.py` | Half-up (`+0.5`), not floor |
| DistSeal | `distseal.utils.display.save_img` | Clip RGB, multiply by 255, uint8 cast |

The selected functions/expressions are extracted from hash-verified upstream
files and executed without unrelated CLI imports. Source revisions and Git blob
hashes are in `source_contracts.json`. Missing or altered code fails closed.
SepMark also exports the adapter's exact RGB float with the same half-up rule to
separate serializer effects from signed/RGB residual arithmetic. DistSeal's
post-hoc notebook displays and evaluates float images; `save_img` is a repository
export utility, **not** evidence that the paper evaluated a PNG round-trip.

## What this reference does not claim

- HiDDeN remains the ando-khachatryan third-party PyTorch combined-noise checkpoint,
  not the original authors' Torch7 implementation. Its clean encoder/decoder
  bypasses the demonstration's noise layers and random crop.
- RoSteALS uses the pinned WIBE fork and distributed inference YAML/checkpoint,
  not a newly selected release. Its demo's raw tensor decoding is not confused
  with PNG decoding. Both boundaries are measured.
- SepWater's unused training loss and noise modules are omitted, as in the old
  adapter. Every `ae.*`, `control.*`, `decoder.*` tensor must still load from the
  checkpoint. The independent test covers inference, not training initialization.
- RoSteALS/SepWater receive the frozen arbitrary 100-bit vectors. Optional text
  and BCH wrappers are deliberately excluded; no text-recovery claim is made.
- SepMark/DistSeal get a serializer check and different-image validation, not a
  separately loaded second encoder. SepMark's selected distributed checkpoint
  still has the provenance limitation already recorded in the old bank.
- No automatic production patch, attack suite, matched-PSNR search, training,
  clean-success filtering, payload shortening or winner selection is performed.

## Review and development

Send `author_reference_report.zip`. It contains summaries, exact decoded vectors,
logits, weight inventories, source audits and logs, but not large PNG/NPY/weight
files. Those remain on Drive. Review disagreements and native error rates before
deciding whether to modify the production adapters or run 100 images.

From this repository directory, build with `python build_notebook.py`.
Tests: `python -m pytest tests -q` from this directory or the extracted source ZIP,
with NumPy, Pillow, PyTorch, torchvision, nbformat and pytest available.
Six optional upstream-source smoke checks use `WMBENCH_UPSTREAM_FIXTURES` as
described in `tests/test_upstream_smoke.py`; they are skipped when not supplied.
The build pins the previous recovery source ZIP and packages readable Python
files. It never rebuilds or changes the old production release.

### Pinned source references

- [HiDDeN implementation](https://github.com/ando-khachatryan/HiDDeN/tree/556f76dd0602e4351ed4c19bd2ee87d50ef3c0de)
- [RoSteALS inference](https://github.com/yakulex/RoSteALS/blob/d193f13ba06e1ab373dad90b68f37c466cb7a1c3/inference.py)
- [SepWater embedding](https://github.com/CVhnu/sepwater/blob/4b99a4d3a99f504a09e5e743cffe745b8856e36a/embed.py)
- [SepMark test/export](https://github.com/sh1newu/SepMark/blob/4bf060da5b31abc2d623b6edf8aafcdd2d2f8db1/test_Dual_Mark.py)
- [DistSeal export utility](https://github.com/facebookresearch/distseal/blob/d0a71cd272bb29e3b49cdc965bbaf4d24f2684ac/distseal/utils/display.py)
