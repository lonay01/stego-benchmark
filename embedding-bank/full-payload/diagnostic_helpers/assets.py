"""Acquire published candidates and inventory them without unpickling Python objects."""

from __future__ import annotations

import argparse
import pickletools
import shutil
import zipfile
from pathlib import Path

from wmbench.io import read_json, sha256, write_json
from .core import inside

AUTHORS = {
    "editguard": "https://github.com/xuanyuzhang21/EditGuard",
    "sepwater": "https://github.com/CVhnu/sepwater",
    "omniguard": "https://github.com/xuanyuzhang21/OmniGuard",
    "fang_screen": "https://github.com/FangHanNUS/Screen-Shooting-Resilient-Watermarking",
    "camera_shooting_instance": "https://github.com/gyz-star/Camera-Shooting-Resilient-Watermarking-on-Image-Instance-Level",
}
DOWNLOADS = {
    "editguard/degrade.pth": "1fAC2EIrMfPKuQa_DdYdxmUwLBbbsTJXC",
    "sepwater/last.ckpt": "17RXXzQ-rMGJ7Xd6e-Az7xeylMj2r0iQS",
    "omniguard/checkpoint.zip": "1khdBDUDIRIhPIKlV0ictcbTdWLh-WFY_",
}
# Locally verified downloads on 2026-09-20; not checksums published by the authors.
OBSERVED_SHA256 = {
    "editguard/degrade.pth": "771d0418aae3926c953782a61c064fa6b49b52dba839ed1b7fd3583ffb467983",
    "sepwater/last.ckpt": "39a0ea5bd4baa6a57a8ed2b3b058429df93f87c34205d4eb08680d282a71c3fa",
    "omniguard/checkpoint.zip": "ba6f56e7029a82bd8d1a119f486cd7fc7985bcbebf87f413df676d47a5669dfa",
}


def checkpoint_strings(path: Path) -> dict:
    """Inspect tensor names with pickletools, without executing a checkpoint pickle."""
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.endswith("data.pkl")]
        if len(members) != 1 or members[0].file_size > 64 * 1024**2:
            raise ValueError("Not a single bounded PyTorch state archive")
        strings = [arg for opcode, arg, _ in pickletools.genops(archive.read(members[0]))
                   if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8")]
    prefixes = ("ae.", "control.", "decoder.", "encoder.", "module.encoder.", "module.decoder.",
                "module.bm.encoder.", "module.bm.decoder.")
    return {"prefix_counts": {p: sum(s.startswith(p) for s in strings) for p in prefixes},
            "sample_names": [s for s in strings if ".weight" in s or ".bias" in s][:30],
            "state_dict_marker": "state_dict" in strings,
            "note": "Name inventory only, not architecture/shape validation or a successful model load"}


def fetch(root: Path, relative: str, file_id: str) -> Path:
    import gdown
    path = inside(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".receipt.json")
    if receipt.is_file():
        old = read_json(receipt)
        if old["file_id"] != file_id or not path.is_file() or sha256(path) != old["sha256"]:
            raise ValueError(f"Downloaded asset changed: {path}")
        return path
    if not path.is_file():
        temporary = path.with_name(path.name + ".part")
        print(f"DOWNLOAD author asset {relative}", flush=True)
        value = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
        if not value or not temporary.is_file() or temporary.stat().st_size < 1024:
            raise RuntimeError(f"Author download unavailable: {file_id}")
        if not zipfile.is_zipfile(temporary):
            raise ValueError(f"Downloaded file is not a checkpoint/ZIP (possibly an access page): {file_id}")
        temporary.replace(path)
    checksum = sha256(path)
    if relative in OBSERVED_SHA256 and checksum != OBSERVED_SHA256[relative]:
        raise ValueError(f"Author asset differs from the reviewed download: {path}. Preserve and inspect it.")
    write_json(receipt, {"file_id": file_id, "sha256": checksum, "bytes": path.stat().st_size,
                        "author": AUTHORS[relative.split('/')[0]], "upstream_checksum": "not published"})
    return path


def acquire(root: Path, bank: Path, enabled=True):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative, identifier in DOWNLOADS.items():
        try:
            # Reuse an exact known manual file only in a diagnostic copy.
            old_name = "sepwater/sepwater.ckpt" if relative == "sepwater/last.ckpt" else relative
            old = inside(bank / "_manual_assets", old_name)
            target = inside(root, relative)
            if old.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, target)
                write_json(target.with_name(target.name + ".local-origin.json"),
                           {"source": str(old), "sha256": sha256(old)})
            if not enabled and not target.exists():
                raise FileNotFoundError("Automatic downloads disabled; supply the author file manually")
            # Local manual copies retain their true provenance; do not claim they were downloaded.
            if not target.with_name(target.name + ".local-origin.json").exists():
                target = fetch(root, relative, identifier)
            result = {"asset": relative, "status": "downloaded_not_yet_model_validated",
                      "sha256": sha256(target), "bytes": target.stat().st_size}
            if relative.endswith(".zip"):
                with zipfile.ZipFile(target) as archive:
                    result["archive_members"] = [i.filename for i in archive.infolist()]
                    # Extract only bounded checkpoint candidates, no scripts or archive paths.
                    for index, member in enumerate(archive.infolist()):
                        if Path(member.filename).suffix.lower() not in (".pth", ".pt", ".ckpt"):
                            continue
                        if member.file_size > 4 * 1024**3:
                            continue
                        dest = root / "omniguard/candidates" / f"{index:03d}_{Path(member.filename).name}"
                        if dest.exists() and dest.stat().st_size != member.file_size:
                            raise ValueError(f"Candidate size differs from the archive: {dest}")
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            temporary = dest.with_name(dest.name + ".part")
                            with archive.open(member) as src, temporary.open("wb") as out:
                                shutil.copyfileobj(src, out, 1024**2)
                            temporary.replace(dest)
                        try:
                            info = checkpoint_strings(dest)
                        except Exception as error:
                            info = {"error": str(error)}
                        write_json(dest.with_name(dest.name + ".inventory.json"), info)
            else:
                result["inventory"] = checkpoint_strings(target)
            rows.append(result)
        except Exception as error:
            rows.append({"asset": relative, "status": "blocked", "reason": str(error),
                         "url": f"https://drive.google.com/file/d/{identifier}/view"})
        write_json(root / "asset_report.json", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    acquire(args.root, args.bank, not args.no_download)
