"""Resume release 1.0.0 across permutations of Diffusers bookkeeping keys.

The original release, frozen execution identities, attack code, seeds and result
hashes stay untouched. This entry point delegates every stage to that release.
Only scheduler_config._use_default_values is compared as an unordered set of
unique strings. Diffusers 0.32.2 builds it using list(set(...)); its order does
not select or configure the scheduler's numerical operations.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


VERSION = "scheduler-default-order-v1"
BASE_ZIP_SHA256 = "5b70653cf2fb7b23d6ff1b8585aa1b684808ba311d956cf5fc20da4b6a6311e2"
ENTRYPOINT = "neural_sdxl_identity_fix"
HASH_ENV = "WMBENCH_NEURAL_IDENTITY_FIX_SHA256"
DEFAULT_KEYS = "_use_default_values"


def module_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def canonical_identity(value):
    """Copy metadata; normalize exactly one field, never the live scheduler."""
    result = copy.deepcopy(value)
    config = result.get("scheduler_config")
    if isinstance(config, dict) and DEFAULT_KEYS in config:
        keys = config[DEFAULT_KEYS]
        if (not isinstance(keys, list) or not all(isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys))):
            raise ValueError("scheduler_config._use_default_values must be unique strings")
        config[DEFAULT_KEYS] = sorted(keys)
    return result


def changed_paths(left, right, prefix=""):
    """Report genuine differences without silently ignoring missing fields."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(changed_paths(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def reconcile_identity(results, engine, observed):
    """Reuse a legacy identity only after exact semantic-equivalence checking.

    Existing package/record references contain the digest of the legacy JSON.
    Retaining that JSON rather than rewriting it keeps those references valid.
    The actual newly observed metadata is retained in a separate audit receipt.
    """
    from wmbench.io import read_json, sha256
    from wmbench_full_payload.storage import EvidenceError, freeze
    from wmbench_neural.spec import digest

    root = Path(results)
    path = root / "_execution" / f"{engine}.json"
    if not path.exists():
        # A new engine has no historical references to preserve. Its metadata
        # can be deterministic from the outset; original freeze() still runs.
        return canonical_identity(observed)
    saved = read_json(path)
    normalized_saved = canonical_identity(saved)
    normalized_observed = canonical_identity(observed)
    if normalized_saved != normalized_observed:
        differences = changed_paths(normalized_saved, normalized_observed)
        raise EvidenceError(
            f"Execution identity changed for {engine}: {', '.join(differences[:16])}. "
            "Only a permutation of scheduler_config._use_default_values is allowed. "
            "Saved results and execution metadata were not modified."
        )
    if saved != observed:
        receipt = dict(
            repair=VERSION, repair_sha256=module_sha256(),
            base_zip_sha256=BASE_ZIP_SHA256, engine=engine,
            allowed_difference="scheduler_config._use_default_values order only",
            preserved_execution_file=f"_execution/{engine}.json",
            preserved_file_sha256=sha256(path),
            preserved_identity_sha256=digest(saved),
            observed_identity_sha256=digest(observed), observed_identity=observed,
        )
        target = root / "_repairs" / VERSION / f"{engine}-{digest(observed)}.json"
        freeze(target, receipt)
        print(f"IDENTITY REPAIR {engine}: service-key order only; legacy identity "
              f"and results preserved. Audit: {target}", flush=True)
    return saved


def redirect_child(command, env, expected_sha256):
    """Propagate this entry point through process_guard; leave other jobs alone."""
    forwarded = list(command)
    for index in range(len(forwarded) - 1):
        if forwarded[index] == "-m" and forwarded[index + 1] == "wmbench_neural.runner":
            forwarded[index + 1] = ENTRYPOINT
            child_env = dict(env)
            child_env[HASH_ENV] = expected_sha256
            return forwarded, child_env
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate", "prepare", "quality",
        "geometry", "controls", "generate", "decode", "report"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    actual_hash = module_sha256()
    if os.environ.get(HASH_ENV) != actual_hash:
        raise RuntimeError("Repair checksum is absent or changed. Rerun the notebook repair cell.")
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    from wmbench_full_payload.storage import freeze
    from wmbench_neural import runner

    results = settings["results"] if args.action == "generate" else settings.get("output")
    if args.action in {"run", "validate", "generate", "report"}:
        freeze(Path(results) / "_repairs" / VERSION / "repair.json", dict(
            repair=VERSION, repair_sha256=actual_hash, base_zip_sha256=BASE_ZIP_SHA256,
            scope="scheduler bookkeeping-list order only; original release and results unchanged"))
    if args.action == "run":
        from wmbench import colab_canary
        original_monitor = colab_canary.monitored_process

        def monitor(command, env, *positional, **keywords):
            command, env = redirect_child(command, env, actual_hash)
            return original_monitor(command, env, *positional, **keywords)

        with patch.object(colab_canary, "monitored_process", monitor):
            runner.main()
    elif args.action == "generate":
        from wmbench_neural import generate
        original_channel = generate.NeuralChannel

        def channel(engine, cache):
            instance = original_channel(engine, cache)
            instance.identity = reconcile_identity(settings["results"], engine, instance.identity)
            return instance

        with patch.object(generate, "NeuralChannel", channel):
            runner.main()
    else:
        runner.main()


if __name__ == "__main__":
    main()
