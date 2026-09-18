"""Actual CPU embedding-bank smoke using cached weights, with networking disabled.

Run with the already provisioned validation environment. Each method is isolated
in a subprocess. The single cached natural-image fixture is not a COCO100 run.
No dependencies, checkpoints or source repositories are installed/downloaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from PIL import Image

from wmbench.embedding_bank import POLICY, process_image, valid_record
from wmbench.io import atomic_bytes, read_json, sha256, write_json
from wmbench.protocol import load_registry


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("dwt_dct_svd", "pimog", "ssrh", "stegastamp")


def forbid_network():
    def check(event, _args):
        if event == "socket.connect":
            raise RuntimeError("This offline smoke test forbids network connections")

    sys.addaudithook(check)


def assets(model_id: str, cache: Path):
    if model_id == "pimog":
        return (
            cache / "PIMoG",
            cache / "PIMoG/models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth",
        )
    if model_id == "ssrh":
        return cache / "SSRH", cache / "SSRH/models/SSRH.pth"
    if model_id == "stegastamp":
        return cache, cache / "stega_stamp.onnx"
    return cache, None


class CountedAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.embeds = 0
        self.decodes = 0

    def embed(self, *args, **kwargs):
        self.embeds += 1
        return self.adapter.embed(*args, **kwargs)

    def decode(self, *args, **kwargs):
        self.decodes += 1
        return self.adapter.decode(*args, **kwargs)


def run_child(args):
    forbid_network()
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    report_path = args.output / f"{args.model}_smoke.json"
    model = load_registry(PROJECT)[args.model]
    repo, checkpoint = assets(args.model, args.source_cache)
    report = {"model_id": args.model, "purpose": "CPU interface smoke, not benchmark results"}
    try:
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(
                f"Cached checkpoint unavailable; no download attempted: {checkpoint}"
            )
        report["checkpoint_sha256"] = sha256(checkpoint) if checkpoint else None
        if checkpoint:
            expected = [a.get("sha256") for a in model.get("assets", []) if a.get("sha256")]
            if expected and report["checkpoint_sha256"] not in expected:
                raise ValueError("Cached checkpoint differs from the selected registry release")
        options = {"model_id": args.model, "repo_root": str(repo), "device": "cpu"}
        if checkpoint:
            options["weights"] = str(checkpoint)
        if args.model == "dwt_dct_svd":
            from wmbench.adapters.legacy import InvisibleWatermarkAdapter

            instance = InvisibleWatermarkAdapter({**options, "method": "dwtDctSvd"})
        else:
            from wmbench.adapters.physical import PIMoG, SSRH, StegaStampONNX

            instance = {"pimog": PIMoG, "ssrh": SSRH, "stegastamp": StegaStampONNX}[args.model](
                options
            )
            instance.initialize()
        loaded = time.monotonic()
        adapter = CountedAdapter(instance)
        image = read_json(args.output / "fixture.json")["image"]
        bank = args.output / "bank"
        provenance = {
            "id": f"offline_cpu_smoke_{args.model}",
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "target_environment": "Validation environment; not production Linux/CUDA",
        }
        records = process_image(adapter, model, bank, image, POLICY, provenance)
        first_pass = time.monotonic()
        files = {
            path.relative_to(bank).as_posix(): sha256(path)
            for path in (bank / args.model).rglob("*")
            if path.is_file()
        }
        calls_before = (adapter.embeds, adapter.decodes)
        process_image(adapter, model, bank, image, POLICY, provenance)
        calls_after = (adapter.embeds, adapter.decodes)
        if calls_before != calls_after or adapter.embeds != 1:
            raise AssertionError("Resume repeated embedding or decoding")
        for relative, expected_hash in files.items():
            if sha256(bank / relative) != expected_hash:
                raise AssertionError(f"Resume changed a sealed artifact: {relative}")
        for path in (bank / args.model / "_meta/records").glob("*.json"):
            if valid_record(bank, path) is None:
                raise AssertionError(f"Unsealed record after smoke: {path}")
        modes = []
        for record in records:
            check = record.get("identity_check", {})
            if record["status"] == "ok" and check.get("status") != "measured":
                raise AssertionError(f"Decoder interface failed: {check}")
            modes.append(
                {
                    "mode": record["mode"],
                    "status": record["status"],
                    "psnr_db": record.get("psnr"),
                    "alpha": record.get("alpha"),
                    "common30_ber": check.get("common30_ber"),
                    "common30_exact": check.get("common30_exact"),
                    "native_ber": check.get("native_ber"),
                    "output": record.get("output"),
                }
            )
        report.update(
            status="passed_interface",
            native_bits=model["native_bits"],
            load_seconds=loaded - started,
            first_pass_seconds=first_pass - loaded,
            resume_seconds=time.monotonic() - first_pass,
            embed_calls=adapter.embeds,
            decode_calls=adapter.decodes,
            resume_verified=True,
            modes=modes,
            runtime=provenance,
        )
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    report["total_seconds"] = time.monotonic() - started
    write_json(report_path, report)
    return int(report["status"] != "passed_interface")


def prepare_fixture(output: Path, source_cache: Path):
    source = PROJECT / "validation/arwgan_cpu/natural_512.png"
    notes = (
        "Cached natural 512px canary; historical COCO100 manifest is not validated by this test."
    )
    if not source.is_file():
        source = source_cache / "PIMoG/Datasets/images/29.png"
        notes = "Author-provided PIMoG demonstration image, resized; not a COCO benchmark member."
    if not source.is_file():
        raise FileNotFoundError("No cached natural-image fixture; no download attempted")
    target = output / "bank/_inputs/images/000000000139.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if image.size == (512, 512) and image.mode == "RGB":
            atomic_bytes(target, source.read_bytes())
            preprocessing = "None: exact cached PNG bytes"
        else:
            image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC).save(target)
            preprocessing = "RGB conversion and Pillow BICUBIC resize to512x512"
    fixture = {
        "source": str(source),
        "source_sha256": sha256(source),
        "notes": notes,
        "preprocessing": preprocessing,
        "image_id_note": "139 is the fixed test identifier, not a revalidated COCO membership claim.",
        "image": {
            "image_id": "139",
            "path": target.relative_to(output / "bank").as_posix(),
            "sha256": sha256(target),
        },
    }
    write_json(output / "fixture.json", fixture)
    return fixture


def run_parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = prepare_fixture(args.output, args.source_cache)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(PROJECT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    reports = []
    for model_id in METHODS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            model_id,
            "--source-cache",
            str(args.source_cache),
            "--output",
            str(args.output),
        ]
        if args.windows_dll_dir:
            command += ["--windows-dll-dir", str(args.windows_dll_dir)]
        log = args.output / f"{model_id}.log"
        print(f"Offline CPU smoke: {model_id}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            try:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=False,
                )
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        report_path = args.output / f"{model_id}_smoke.json"
        report = (
            read_json(report_path)
            if report_path.is_file()
            else {
                "model_id": model_id,
                "status": "failed",
                "error": f"Child exit{exit_code}; see{log.name}",
            }
        )
        report["exit_code"] = exit_code
        reports.append(report)
        print(f"{model_id}: {report['status']}", flush=True)
    packages = {}
    for name in ("torch", "numpy", "Pillow", "onnxruntime", "invisible-watermark"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    summary = {
        "scope": "Four actual cached CPU implementations, one natural canary, Native +scalar39/42/45.",
        "limitations": "No COCO100 benchmark, no attacks, no GPU or production environment validation.",
        "network": "Network connect forbidden in each child; no installs/downloads.",
        "fixture": fixture,
        "packages": packages,
        "models": reports,
    }
    write_json(args.output / "report.json", summary)
    lines = [
        "# Embedding bank: offline CPU smoke",
        "",
        "Actual cached models on one natural-image canary. Not research results or GPU validation.",
        "",
        "| Method | Mode | PSNR (dB) | Common30 BER | Full-message BER | Match state |",
        "|---|---|---:|---:|---:|---|",
    ]
    for report in reports:
        for mode in report.get("modes", []):

            def formatted(value):
                return f"{value:.6f}" if isinstance(value, (int, float)) else str(value)

            lines.append(
                "| "
                + " | ".join(
                    [
                        report["model_id"],
                        mode["mode"],
                        formatted(mode.get("psnr_db")),
                        formatted(mode.get("common30_ber")),
                        formatted(mode.get("native_ber")),
                        mode["status"],
                    ]
                )
                + " |"
            )
        if report["status"] != "passed_interface":
            lines += ["", f"- {report['model_id']}: FAILED — {report.get('error', '')}"]
    lines += [
        "",
        "Each method ran in an isolated CPU subprocess. Resume must add zero embed/decode calls",
        "and preserve every artifact hash. Bit errors remain visible and do not trigger re-embedding.",
        "",
        "No pretrained weights, dependencies or repositories were downloaded.",
        "",
    ]
    atomic_bytes(args.output / "REPORT.md", "\n".join(lines).encode())
    print(f"Report: {args.output / 'REPORT.md'}", flush=True)
    return int(any(r["status"] != "passed_interface" for r in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=METHODS)
    parser.add_argument("--windows-dll-dir", type=Path)
    args = parser.parse_args()
    args.source_cache = args.source_cache.resolve()
    args.output = (
        args.output
        or PROJECT
        / "validation"
        / ("embedding_bank_cpu_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    _dll_handle = (
        os.add_dll_directory(str(args.windows_dll_dir))
        if args.windows_dll_dir and os.name == "nt"
        else None
    )
    return run_child(args) if args.model else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
