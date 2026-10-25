"""Small resume receipts for completed methods; no repeated large-catalogue reads.

The same result-file size/mtime inventory is checked as in fast-resume v1.
This is not a fresh content audit: FULL_AUDIT still rereads the evidence.
External edits that preserve both size and mtime require FULL_AUDIT, as before.
"""

import time
from wmbench.io import read_json, sha256
from wmbench_digital.spec import digest
from .cache import directory_inventory, fingerprint, pack, unpack

CATALOGUES = ("cases.json", "coverage.json", "_fast/catalogue.json.gz")


def selection_key(ctx, model):
    return digest(dict(identity=ctx["identity"], model=model, images=ctx["images"]))


def seal(ctx, model):
    """Call only after the ordinary finished()/mark_finished() gate succeeds."""
    folder = ctx["output"] / model["id"]
    receipt = folder / "_fast/complete.json.gz"
    # Validate the envelope, not just its presence.
    unpack(receipt)
    catalogues = {n: fingerprint(folder / n) for n in CATALOGUES}
    if not all(catalogues.values()):
        return  # An old/incomplete catalogue uses the ordinary path next time.
    value = dict(version=1, selection_key=selection_key(ctx, model),
                 complete_sha256=sha256(receipt), catalogues=catalogues,
                 catalogue_sha256={n: sha256(folder / n) for n in CATALOGUES},
                 counts=read_json(folder / "tables/counts.json"), time=time.time())
    pack(folder / "_fast/completed-selection.json.gz", value)


def check(ctx, model, full_audit=False):
    """Return counts only if selection, catalogues, records and tables agree."""
    if full_audit:
        return None
    folder = ctx["output"] / model["id"]
    path = folder / "_fast/completed-selection.json.gz"
    if not path.exists():
        return None
    value = unpack(path)
    if value.get("version") != 1 or value["selection_key"] != selection_key(ctx, model):
        return None
    for name, stamp in value["catalogues"].items():
        current = fingerprint(folder / name)
        if current is None:
            return None
        if current != stamp:
            expected = value.get("catalogue_sha256", {}).get(name)
            if expected is None or sha256(folder / name) != expected:
                return None
    receipt = folder / "_fast/complete.json.gz"
    if not receipt.exists() or sha256(receipt) != value["complete_sha256"]:
        return None
    snap = unpack(receipt)["snapshot"]
    current = directory_inventory(folder / "records")
    if not all(snap["records"].values()) or any(current.get(n) != s for n, s in snap["records"].items()):
        return None
    from .runner import tables_agree
    if not tables_agree(folder, snap):
        return None
    counts = read_json(folder / "tables/counts.json")
    if counts != value["counts"] or counts["complete_carriers"] != counts["carriers"]:
        return None
    return counts
