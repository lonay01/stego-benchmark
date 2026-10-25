# Meta HiDDeN and FNNS: production resume repair

This repair applies to `COCO_Embedding_Bank_PRODUCTION_MAGICKWAND_FIX.ipynb`.
It does not change any model architecture, weights, payload, resolution, strength,
PSNR target, matching search, decoder or scoring rule.

## What failed

- **Meta Revisited HiDDeN:** `attenuations.cpython-310.pyc` was included in the
  weight inventory. The generated cache differed between sessions, while all
  three pinned source/checkpoint hashes and both dependency hashes were equal.
  The repair excludes exactly that cache entry when comparing identities, checks
  the three real assets against the pinned hashes, retains the original identity,
  and records both inventories. Author classes are compiled from verified source
  rather than reading the generated cache. Unknown differences still stop the run.
- **FNNS:** `native/000000080666_m05.png` on Drive was zero bytes; its sealed
  record expected SHA256 `3aea2d916b8f44cd97aea35e79aa97f166be08c006987059cf6f2743be67048a`.
  Drive had no prior revision. This is a storage failure, not a decoding result.

## Recovery rules

For FNNS, a valid content-addressed backup restores exactly the recorded bytes.
If there is no intact backup, only the affected image/message pair is retired:
all existing carrier files and dependent encode/decode records are copied to
`fnns_omniseal/_meta/repairs/<pair>/<event>/original/`, verified, then removed
from the active bank. The same source, payload, task seed and model settings are
used to recreate that pair's Native and Matched modes. Its previous matched
results must not be mixed with a regenerated Native. The recovery journal is
resumable after interruption. Original decoder metadata and source crops stay
in place. Changed sources, payload contracts, keys or non-carrier artifacts are
not treated as a repairable PNG failure.

The new result is a **technical retry**, not proof of byte-identical GPU inference.
The journal retains the previous hashes and the new Native artifacts. No BER
or PSNR result selects a retry or a winner. Intact pairs are never re-embedded.
New FNNS PNGs receive verified backup copies. Read-back and hashes cannot
guarantee that Google Drive will never lose data later; retain an SSD backup too.

## Run

1. Stop the previous runner. Use the same Drive account and output root.
2. Open `COCO_Embedding_Bank_PRODUCTION_HIDDEN_FNNS_FIX.ipynb` in Colab.
3. Keep `N_IMAGES=100` for the existing 100-image bank. A100 with high RAM matches
   its previous execution. The selection contains only the two repaired methods.
4. Run the cells in order. Upload the supplied `hotfix_hidden_fnns.py` when
   prompted. The original source ZIP is reused from Drive; if absent, upload the
   unchanged `COCO_Embedding_Bank_PRODUCTION_source.zip` included in the bundle.
5. Inspect the final report and `production_bank_report.zip`. Reporting includes
   all 29 methods; the other 27 are not loaded or run. Normal nonzero BER and
   unattainable matched targets are not treated as implementation failures.

The helper is readable Python, cached on Drive and pinned by SHA256 in the
notebook. It lives outside the frozen scientific release; that release stays at
SHA256 `6d7ddfcd05abdaf46f97ab7b6b174bff098b3043edcf069db64041554263f7b5`.
The same helper is explicitly invoked in both affected isolated worker processes.
Build with `python build_hidden_fnns_hotfix.py`.

Local validation covers notebook syntax/schema, the immutable release, Meta
identity drift, scoped FNNS recovery and interruption/resume. It does not replace
the user's two-method CUDA run. Validation logs are saved under `validation/`.
