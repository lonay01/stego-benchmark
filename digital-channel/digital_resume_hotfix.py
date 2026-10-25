"""Make temporary ZIP cleanup idempotent after a Colab runtime reset.

The original release and StegaStamp repair remain unchanged. Completed results
are verified first. Owner/checksum checks remain mandatory for every ZIP that
still exists; an absent ZIP requires no deletion and no owner file.
"""

import ast
import hashlib
from pathlib import Path

VERSION = "digital-missing-cache-v1"
STORAGE_SHA = "69f1f1362d337e8b955b8aee2e2da08e3a25d3cc62f6774c3de6c270bd2bf7de"


def install_cleanup_patch():
    """Move exactly one owner check inside the existing-file deletion branch."""
    from wmbench_digital import storage, runner

    source = Path(storage.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != STORAGE_SHA:
        raise RuntimeError("Unexpected cleanup version; refusing to apply resume repair")
    tree = ast.parse(source.decode("utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "remove_packages")
    loops = [n for n in function.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "manifest"]
    if len(loops) != 1:
        raise RuntimeError("Expected exactly one manifest cleanup loop")
    loop = loops[0]
    owner_test = ast.parse('read_json(folder / "owner.json") != {"identity": identity, "method": method}', mode="eval").body
    exists_test = ast.parse("file.exists()", mode="eval").body
    owner = [n for n in loop.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(owner_test)]
    deletion = [n for n in loop.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(exists_test)]
    if len(owner) != 1 or len(deletion) != 1:
        raise RuntimeError("Expected exactly one owner check and one conditional ZIP deletion")
    loop.body.remove(owner[0])
    deletion[0].body.insert(0, owner[0])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(storage.__file__) + " [missing-cache resume fix]", "exec"), storage.__dict__)
    runner.remove_packages = storage.remove_packages


def main():
    # Keep the previously supplied StegaStamp handling byte-for-byte unchanged.
    import stegastamp_clean_hotfix

    install_cleanup_patch()
    stegastamp_clean_hotfix.main()


if __name__ == "__main__":
    main()
