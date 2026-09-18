"""JSON task worker; a single process owns one heavy adapter and one CUDA context."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .io import read_json, sha256, write_json
from .generation_errors import GenerationBlocked
from .matching import load_rgb, match_psnr, psnr, save_rgb
from .store import JobStore


def adapter_instance(config: dict):
    module, name = config["model"]["adapter"].split(":")
    return getattr(importlib.import_module(module), name)(config["options"])


def runtime_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    return info


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def decoded_metrics(result: dict, spec: dict, metadata: dict) -> dict:
    """Extraction is blind; expected bits are consulted only after the adapter returns."""
    data = dict(result)
    expected = metadata.get("expected_bits") or spec.get("payload") or []
    bits = result.get("bits")
    if bits is not None:
        bits = np.asarray(bits).reshape(-1).tolist()
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("Decoder returned non-binary values as bits")
        if len(bits) != len(expected):
            raise ValueError(
                f"Decoder length {len(bits)} != expected native length {len(expected)}"
            )
        indices = spec.get("common_indices", [])
        data.update(
            {
                "bits": bits,
                "expected_bits": expected,
                "native_ber": float(np.mean(np.asarray(bits) != np.asarray(expected))),
                "native_exact": int(bits == expected),
                "evaluated_bits": len(indices),
            }
        )
        if indices:
            recovered = [bits[i] for i in indices]
            target = [expected[i] for i in indices]
            ber = float(np.mean(np.asarray(recovered) != np.asarray(target)))
            data.update(
                {
                    "ber": ber,
                    "exact": int(recovered == target),
                    "score": 1 - ber,
                    "score_name": f"bit_agreement_{len(indices)}",
                }
            )
        elif expected:
            data.update({"score": 1 - data["native_ber"], "score_name": "native_bit_agreement"})
    score = data.get("score")
    if score is not None and not np.isfinite(score):
        raise ValueError("Non-finite detection score")
    if result.get("higher_is_marked") is False:
        data["score"] = -float(score)
        data["score_name"] = "negative_" + data.get("score_name", "adapter_score")
    return data


def execute_task(adapter, spec: dict, store: JobStore) -> tuple[str, dict, list[Path]]:
    stage = spec["stage"]
    folder = store.root / "artifacts" / store.identifier(spec)
    folder.mkdir(parents=True, exist_ok=True)
    output, metadata = folder / "image.png", folder / "metadata.json"
    if stage == "embed":
        adapter.embed(spec["input_path"], str(output), spec["payload"], str(metadata))
        source, result = load_rgb(spec["input_path"]), load_rgb(output)
        if source.shape != result.shape:
            raise ValueError("Adapter changed the common image grid")
        data = {
            "image_path": str(output),
            "metadata_path": str(metadata),
            "psnr": psnr(source, result),
        }
    elif stage == "origin":
        adapter.generate(
            spec["prompt"],
            spec["seed"],
            str(output),
            str(metadata),
            payload=spec["payload"] or None,
        )
        data = {
            "image_path": str(output),
            "metadata_path": str(metadata),
            "clean_path": read_json(metadata).get("clean_path"),
        }
        if load_rgb(output).shape != (512, 512, 3):
            raise ValueError("Origin output is not a 512x512 RGB image")
    elif stage == "origin_clean":
        adapter.generate_clean(spec["prompt"], spec["seed"], str(output))
        if load_rgb(output).shape != (512, 512, 3):
            raise ValueError("Origin clean calibration output is not 512x512")
        data = {
            "image_path": str(output),
            "carrier": "unwatermarked_generated",
            "purpose": "independent_origin_calibration",
        }
    elif stage == "match":
        source, native = load_rgb(spec["source_path"]), load_rgb(spec["input_path"])
        image, data = match_psnr(source, native, **spec["matching"])
        save_rgb(output, image)
        actual = psnr(source, load_rgb(output))
        if (
            data["status"] == "ok"
            and abs(actual - spec["matching"]["target"]) > spec["matching"]["tolerance"]
        ):
            raise ValueError("Saved PNG failed matched-PSNR tolerance")
        data.update(
            {"image_path": str(output), "metadata_path": spec["metadata_path"], "psnr": actual}
        )
        return data["status"], data, [output]
    elif stage == "attack":
        data = adapter.apply(
            spec["input_path"], str(output), spec["attack"], spec["seed"], spec["prompt"]
        )
        data["image_path"] = str(output)
    elif stage == "decode":
        metadata_value = read_json(spec["metadata_path"])
        data = decoded_metrics(
            adapter.decode(spec["input_path"], spec["metadata_path"]), spec, metadata_value
        )
        return "ok", data, []
    elif stage == "quality":
        return (
            "ok",
            adapter.compare(
                spec["first_path"],
                spec["second_path"],
                spec.get("prompt", ""),
                spec.get("mask_box"),
            ),
            [],
        )
    else:
        raise ValueError(f"Unknown task stage: {stage}")
    # Includes cryptographic/carrier sidecars. A PNG alone is not a completed embed.
    paths = [
        path for path in folder.rglob("*") if path.is_file() and not path.name.endswith(".part")
    ]
    if stage in ("embed", "origin") and not metadata.exists():
        raise ValueError("Adapter did not save its decoding metadata/key")
    return "ok", data, paths


def run_tasks(config: dict, task_path: Path):
    store = JobStore(config["run_root"])
    tasks = read_json(task_path)
    adapter = None
    for index, spec in enumerate(tasks, 1):
        if store.valid(spec):
            continue
        started = time.monotonic()
        try:
            with store.claim(spec):
                for key in (
                    "input_path",
                    "source_path",
                    "first_path",
                    "second_path",
                    "metadata_path",
                ):
                    if (
                        key in spec
                        and spec.get(key + "_sha256")
                        and sha256(spec[key]) != spec[key + "_sha256"]
                    ):
                        raise ValueError(f"Input changed since task planning: {key}")
                if adapter is None and spec["stage"] != "match":
                    print(f"WM_DIAG loading adapter {spec.get('model_id', '')}", flush=True)
                    seed_everything(0)
                    adapter = adapter_instance(config)
                    if hasattr(adapter, "initialize"):
                        adapter.initialize()
                seed_everything(spec.get("seed", 0))
                print(f"WM_DIAG start {spec['stage']} {spec.get('model_id', '')}", flush=True)
                status, data, paths = execute_task(adapter, spec, store)
                if config.get("execution_provenance"):
                    data["execution_provenance"] = config["execution_provenance"]
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        data["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                        data["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    pass
                store.publish(spec, status, data, paths, time.monotonic() - started)
                print(
                    f"[{index}/{len(tasks)}] {spec['stage']} {spec.get('model_id', '')} {status}",
                    flush=True,
                )
        except GenerationBlocked as error:
            # No PNG/BER is published. This terminal record is resumable, but
            # include_terminal=False rejects it when planning decode/quality.
            store.publish(
                spec,
                "generation_blocked",
                {
                    "reason": str(error),
                    "error_type": type(error).__name__,
                    "policy": "fixed_seed_no_resampling_safety_checker_enabled",
                    "execution_provenance": config.get("execution_provenance"),
                },
                [],
                time.monotonic() - started,
            )
            print(f"[{index}/{len(tasks)}] BLOCKED {spec.get('model_id', '')}: {error}", flush=True)
        except FileExistsError:
            print(
                f"Claim exists (another worker or stale claim): {store.identifier(spec)}",
                flush=True,
            )
        except Exception as error:
            detail = {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            store.publish(spec, "failed", detail, [], time.monotonic() - started)
            print(f"[{index}/{len(tasks)}] FAILED {spec.get('model_id', '')}: {error}", flush=True)
            # An initialization failure would otherwise reload a broken model N times.
            if adapter is None or config.get("fail_fast", False):
                raise


def main():
    action = sys.argv[1]
    if action == "download-hf":
        from wmbench.downloads import download_snapshot
        import json

        download_snapshot(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            json.loads(sys.argv[5]) if len(sys.argv) > 5 else [],
        )
    elif action == "initialize":
        config = read_json(sys.argv[2])
        print(f"WM_DIAG initialize {config['model']['id']}", flush=True)
        seed_everything(0)
        adapter = adapter_instance(config)
        if hasattr(adapter, "initialize"):
            adapter.initialize()
        write_json(sys.argv[3], runtime_info())
        print("WM_DIAG initialization completed", flush=True)
    elif action == "run":
        run_tasks(read_json(sys.argv[2]), Path(sys.argv[3]))
    elif action == "runtime":
        write_json(sys.argv[2], runtime_info())
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
