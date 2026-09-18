"""Notebook-friendly staged orchestration. No GPU library is imported in this process."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .channels import channel_models
from .environment import code_fingerprint, prepare_model, validate_prepared, worker_environment
from .io import digest, read_json, sha256, write_csv, write_json
from .protocol import (
    LEGACY_DATASET_HASH,
    attack_manifest,
    full_payload,
    load_registry,
    paired_seed,
    payload_bank,
)
from .quality import quality_model
from .store import JobStore


class Experiment:
    def __init__(self, project: str | Path, configuration: str | Path | dict):
        self.project = Path(project).resolve()
        self.config = (
            read_json(configuration)
            if not isinstance(configuration, dict)
            else copy.deepcopy(configuration)
        )
        self.work = Path(self.config["work_root"]).expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        os.environ["UV_CACHE_DIR"] = str(self.work / "cache/uv")
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(self.work / "cache/python")
        self.registry = load_registry(self.project)
        selected = self.config.get("models", "all")
        self.models = list(self.registry) if selected == "all" else selected
        for model_id in self.models:
            if model_id not in self.registry:
                raise ValueError(f"Unknown model: {model_id}")
            overrides = self.config.get("model_overrides", {}).get(model_id, {})
            for key, value in overrides.items():
                if key == "options":
                    self.registry[model_id].setdefault(key, {}).update(value)
                else:
                    self.registry[model_id][key] = value
        self.attacks = attack_manifest(
            self.config.get("repeats", 1), self.config.get("identity_only", False)
        )
        self.operators = channel_models()
        if self.config.get("quality_metrics", True):
            self.operators["quality_metrics"] = quality_model()
        self.all_models = {**self.registry, **self.operators}
        self.prepared = {}
        self.preparation = []
        self.store = None
        self.planned = []

    def inventory(self) -> list[dict]:
        return [
            {
                "id": m,
                "family": self.registry[m].get("family"),
                "track": self.registry[m]["track"],
                "native_bits": self.registry[m].get("native_bits"),
                "backend": self.registry[m].get("backend", "python"),
                "status": self.registry[m].get("capability", {}).get("status", "ready_unverified"),
            }
            for m in self.models
        ]

    def prepare(self) -> list[dict]:
        operators = {"channel_" + a["engine"] for a in self.attacks if a["engine"] != "identity"}
        if self.config.get("quality_metrics", True):
            operators.add("quality_metrics")
        self.preparation = []
        for model_id in [*self.models, *sorted(operators)]:
            model = self.all_models[model_id]
            try:
                self.prepared[model_id] = prepare_model(
                    self.project, self.work, model, self.config.get("device", "cuda")
                )
                self.preparation.append({"model_id": model_id, "status": "ready", "reason": ""})
                print(f"READY {model_id}", flush=True)
            except Exception as error:
                self.preparation.append(
                    {
                        "model_id": model_id,
                        "status": "not_ready",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"NOT READY {model_id}: {error}", flush=True)
        write_json(self.work / "preparation_status.json", self.preparation)
        write_csv(self.work / "preparation_status.csv", self.preparation)
        return self.preparation

    def _load_prepared(self):
        needed = set(self.models) | {
            "channel_" + a["engine"] for a in self.attacks if a["engine"] != "identity"
        }
        if self.config.get("quality_metrics", True):
            needed.add("quality_metrics")
        for model_id in sorted(needed):
            path = self.work / "locks" / model_id / "prepared.json"
            if path.exists():
                prepared = read_json(path)
                validate_prepared(prepared, self.project, self.work, self.all_models[model_id])
                self.prepared[model_id] = prepared

    def _start(self):
        self._load_prepared()
        dataset = read_json(self.config["dataset_manifest"])
        count = int(self.config.get("image_limit", 5))
        if len(dataset["images"]) < count:
            raise ValueError(f"Need {count} images; dataset has only {len(dataset['images'])}")
        if (
            self.config.get("require_legacy_dataset", False)
            and dataset.get("legacy_semantic_hash") != LEGACY_DATASET_HASH
        ):
            raise ValueError(
                "Core profile requires the exact old COCO100 manifest and verified PNG hashes"
            )
        self.images = dataset["images"][:count]
        if not self.config.get("identity_only", False) and any(
            not r.get("prompt", "").strip() for r in self.images
        ):
            raise ValueError(
                "Import COCO captions first; the AI attack prompts must not be silently empty"
            )
        for image in self.images:
            if sha256(image["path"]) != image["sha256"]:
                raise ValueError(f"Dataset image changed: {image['path']}")
        identity = {
            "configuration": self.config,
            "dataset": dataset["content_hash"],
            "models": {m: self.registry[m] for m in self.models},
            "code": code_fingerprint(self.project),
        }
        # Optional persistent artifact root; environments can remain on ephemeral local SSD.
        run_parent = Path(self.config.get("run_parent", self.work / "runs"))
        self.store = JobStore(run_parent / digest(identity)[:20])
        write_json(self.work / "current_run.json", {"run_root": str(self.store.root)})
        write_json(self.store.root / "protocol.json", identity)
        write_json(self.store.root / "attack_manifest.json", self.attacks)
        write_json(self.store.root / "dataset_manifest.json", dataset)
        write_json(self.store.root / "model_registry.json", [self.registry[m] for m in self.models])
        write_csv(self.store.root / "model_registry.csv", self.inventory())
        write_json(self.store.root / "payload_manifest.json", payload_bank())

    def _hash(self, path):
        return sha256(path)

    def _spec(self, stage: str, model_id: str, **fields) -> dict:
        operator = fields.pop("operator", model_id)
        environment = self.prepared.get(operator, {}).get("fingerprint", "not_prepared")
        spec = {
            "schema": 1,
            "stage": stage,
            "model_id": model_id,
            "operator": operator,
            "environment": environment,
            **fields,
        }
        for key in ("input_path", "source_path", "first_path", "second_path", "metadata_path"):
            if key in spec:
                spec[key + "_sha256"] = self._hash(spec[key])
        self.planned.append(spec)
        return spec

    def _dispatch(self, specs: list[dict]):
        grouped = defaultdict(list)
        for spec in specs:
            if not self.store.valid(spec):
                grouped[spec["operator"]].append(spec)
        for operator, pending in grouped.items():
            group_path = self.store.root / "plans" / f"{pending[0]['stage']}_{operator}.json"
            write_json(group_path, pending)
            if pending[0]["stage"] == "match":
                python = sys.executable
                config = {"run_root": str(self.store.root), "model": {}, "options": {}}
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.project / "src")
            elif operator not in self.prepared:
                for spec in pending:
                    self.store.publish(
                        spec, "not_prepared", {"reason": "See preparation_status.csv"}, []
                    )
                continue
            else:
                prepared = self.prepared[operator]
                config = read_json(prepared["config"])
                config["run_root"] = str(self.store.root)
                python = prepared["python"]
                env = worker_environment(self.project, self.work, self.all_models[operator])
            config_path = group_path.with_suffix(".worker.json")
            write_json(config_path, config)
            log_path = self.store.root / "logs" / f"{pending[0]['stage']}_{operator}.log"
            print(
                f"{pending[0]['stage']}: {operator}, {len(pending)} pending; log {log_path.name}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [str(python), "-m", "wmbench.worker", "run", str(config_path), str(group_path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    # Stream progress to Jupyter/terminal; no unbounded in-memory subprocess capture.
                    print(line.rstrip(), flush=True)
                code = process.wait()
                if code:
                    print(
                        f"Worker {operator} exited {code}; other groups will continue", flush=True
                    )

    def _ok(self, spec):
        return self.store.valid(spec, include_terminal=False)

    def _units(self):
        units, specs = [], []
        bank = payload_bank()
        for model_id in self.models:
            model = self.registry[model_id]
            for index, image in enumerate(self.images):
                payload_id = index % 5
                length = model.get("native_bits") or 0
                namespace = model.get("payload_namespace", model_id)
                payload, indices = full_payload(
                    bank[payload_id],
                    length,
                    image["image_id"],
                    model.get("reserved_bits"),
                    namespace,
                    payload_id,
                )
                if model["track"] == "origin":
                    fixed = model.get("options", {}).get("fixed_payload")
                    if fixed:
                        payload = list(map(int, fixed))
                    indices = list(range(len(payload)))
                group = f"{model['track']}_bits{len(indices)}_" + str(
                    model.get("capability", {}).get("ecc", "raw")
                )
                common = {
                    "image_id": image["image_id"],
                    "payload_id": payload_id,
                    "payload": payload,
                    "common_indices": indices,
                    "track": model["track"],
                    "group": group,
                    "seed": paired_seed(image["image_id"], "embedding", payload_id),
                    "mode": "native",
                    "prompt": image.get("prompt", ""),
                }
                if model["track"] == "origin":
                    spec = self._spec("origin", model_id, **common)
                else:
                    spec = self._spec("embed", model_id, input_path=image["path"], **common)
                specs.append(spec)
                units.append({"model_id": model_id, "image": image, "common": common, "spec": spec})
        self._dispatch(specs)
        return units

    def _marked_inputs(self, units):
        inputs, matches = [], []
        for unit in units:
            record = self._ok(unit["spec"])
            if not record:
                continue
            value = {
                **unit,
                "record": record,
                "path": record["data"]["image_path"],
                "metadata": record["data"]["metadata_path"],
            }
            if "native" in self.config.get("modes", ["native", "matched42"]):
                inputs.append(value)
            if unit["common"]["track"] != "posthoc":
                continue
            for target in self.config.get("matched_targets", [42]):
                mode = f"matched{target:g}"
                if mode not in self.config.get("modes", ["native", "matched42"]):
                    continue
                common = {**unit["common"], "mode": mode}
                spec = self._spec(
                    "match",
                    unit["model_id"],
                    operator="controller",
                    input_path=value["path"],
                    source_path=unit["image"]["path"],
                    metadata_path=value["metadata"],
                    matching={
                        "target": target,
                        "tolerance": 0.05,
                        "max_alpha": 4,
                        "iterations": 64,
                    },
                    parent_job=record["job_id"],
                    **common,
                )
                matches.append((value, spec, common))
        self._dispatch([item[1] for item in matches])
        for value, spec, common in matches:
            record = self._ok(spec)
            if record:
                inputs.append(
                    {
                        **value,
                        "common": common,
                        "record": record,
                        "path": record["data"]["image_path"],
                    }
                )
        return inputs

    def _received(self, inputs):
        received, clean_cache, requests = [], {}, []
        for unit in inputs:
            clean_path = (
                unit["record"]["data"].get("clean_path")
                if unit["common"]["track"] == "origin"
                else unit["image"]["path"]
            )
            for attack in self.attacks:
                seed = paired_seed(unit["image"]["image_id"], attack["id"], attack["repeat"])
                attack_fields = {
                    "attack": attack,
                    "seed": seed,
                    "prompt": unit["image"]["prompt"],
                    "image_id": unit["image"]["image_id"],
                }
                for role, path in (("positive", unit["path"]), ("negative_test", clean_path)):
                    if not path:
                        continue
                    if attack["engine"] == "identity":
                        received.append(
                            {
                                "unit": unit,
                                "role": role,
                                "path": path,
                                "attack": attack,
                                "seed": seed,
                            }
                        )
                        continue
                    cache_key = digest({"path_hash": self._hash(path), **attack_fields})
                    if role == "negative_test" and cache_key in clean_cache:
                        spec = clean_cache[cache_key]
                    else:
                        spec = self._spec(
                            "attack",
                            unit["model_id"] if role == "positive" else "clean_control",
                            operator="channel_" + attack["engine"],
                            input_path=path,
                            role=role,
                            **attack_fields,
                        )
                        requests.append(spec)
                        if role == "negative_test":
                            clean_cache[cache_key] = spec
                    received.append(
                        {
                            "unit": unit,
                            "role": role,
                            "attack": attack,
                            "seed": seed,
                            "attack_spec": spec,
                        }
                    )
        self._dispatch(requests)
        valid = []
        for item in received:
            if "attack_spec" in item:
                record = self._ok(item["attack_spec"])
                if not record:
                    continue
                item["path"] = record["data"]["image_path"]
            valid.append(item)
        return valid

    def _calibration_inputs(self):
        negative_path = self.config.get("negative_calibration_manifest")
        if not negative_path:
            return []
        negatives = read_json(negative_path)["images"]
        if len({r["image_id"] for r in negatives}) != len(negatives):
            raise ValueError("Duplicate calibration IDs")
        from .matching import load_rgb
        import hashlib

        pixel_hashes = set()
        for image in negatives:
            if self._hash(image["path"]) != image["sha256"]:
                raise ValueError(f"Calibration PNG changed: {image['path']}")
            pixel_hash = hashlib.sha256(load_rgb(image["path"]).tobytes()).hexdigest()
            if pixel_hash in pixel_hashes:
                raise ValueError("Duplicate calibration pixels under different IDs/files")
            pixel_hashes.add(pixel_hash)
        if {r["image_id"] for r in negatives} & {r["image_id"] for r in self.images}:
            raise ValueError("Calibration and test image IDs overlap")
        if {r["sha256"] for r in negatives} & {r["sha256"] for r in self.images}:
            raise ValueError("Calibration and test image pixels overlap")
        if pixel_hashes & {
            hashlib.sha256(load_rgb(r["path"]).tobytes()).hexdigest() for r in self.images
        }:
            raise ValueError("Calibration pixels duplicate test images")
        return negatives

    def _decode(self, received, units):
        specs, seen = [], set()
        for item in received:
            unit = item["unit"]
            common = {**unit["common"], "seed": item["seed"]}
            if item["role"] == "negative_test":
                common["mode"] = "control"
            spec = self._spec(
                "decode",
                unit["model_id"],
                input_path=item["path"],
                metadata_path=unit["metadata"],
                role=item["role"],
                attack=item["attack"],
                **common,
            )
            if digest(spec) not in seen:
                specs.append(spec)
                seen.add(digest(spec))
        negatives = self._calibration_inputs()
        if negatives:
            by_model = defaultdict(list)
            for unit in units:
                record = self._ok(unit["spec"])
                if record:
                    by_model[unit["model_id"]].append((unit, record))
            for model_id, references in by_model.items():
                reference_ids = self.config.get("calibration_reference_ids")
                if reference_ids:
                    # Freeze the key/payload reference bank before looking at outcomes.
                    # Expanding N or losing an embed must not rotate calibration keys.
                    lookup = {str(u["image"]["image_id"]): (u, r) for u, r in references}
                    if any(str(i) not in lookup for i in reference_ids):
                        print(
                            f"CALIBRATION INCOMPLETE {model_id}: fixed reference bank unavailable",
                            flush=True,
                        )
                        continue
                    references = [lookup[str(i)] for i in reference_ids]
                origin_clean_specs = []
                if self.registry[model_id]["track"] == "origin":
                    for image in negatives:
                        if not image.get("prompt", "").strip():
                            raise ValueError(
                                "Origin calibration requires fixed held-out COCO captions"
                            )
                        origin_clean_specs.append(
                            self._spec(
                                "origin_clean",
                                model_id,
                                image_id=image["image_id"],
                                track="origin",
                                prompt=image["prompt"],
                                seed=paired_seed(image["image_id"], "origin_clean_calibration", 0),
                                role="calibration",
                                mode="control",
                            )
                        )
                    self._dispatch(origin_clean_specs)
                for index, image in enumerate(negatives):
                    unit, record = references[index % len(references)]
                    common = {
                        **unit["common"],
                        "image_id": image["image_id"],
                        "mode": "calibration",
                    }
                    input_path = image["path"]
                    if origin_clean_specs:
                        clean_record = self._ok(origin_clean_specs[index])
                        if not clean_record:
                            continue
                        input_path = clean_record["data"]["image_path"]
                    specs.append(
                        self._spec(
                            "decode",
                            model_id,
                            input_path=input_path,
                            metadata_path=record["data"]["metadata_path"],
                            role="calibration",
                            attack={"id": "identity", "family": "identity", "repeat": 0},
                            **common,
                        )
                    )
        self._dispatch(specs)

    def _quality(self, received, inputs):
        if not self.config.get("quality_metrics", True):
            return
        specs, clean = [], {}
        for item in received:
            if item["role"] == "negative_test":
                key = (
                    item["unit"]["model_id"],
                    item["unit"]["image"]["image_id"],
                    item["attack"]["id"],
                    item["attack"]["repeat"],
                )
                clean[key] = item["path"]
        for unit in inputs:
            if unit["common"]["track"] == "posthoc":
                specs.append(
                    self._spec(
                        "quality",
                        unit["model_id"],
                        operator="quality_metrics",
                        first_path=unit["image"]["path"],
                        second_path=unit["path"],
                        comparison="embedding",
                        **unit["common"],
                    )
                )
        for item in received:
            unit = item["unit"]
            if item["attack"]["engine"] == "identity":
                continue
            first = (
                unit["path"]
                if item["role"] == "positive"
                else (
                    unit["record"]["data"].get("clean_path")
                    if unit["common"]["track"] == "origin"
                    else unit["image"]["path"]
                )
            )
            common = {
                **unit["common"],
                "prompt": item["attack"].get("prompt", unit["image"]["prompt"]),
            }
            fields = {
                "comparison": "channel_" + item["role"],
                "attack": item["attack"],
                "mask_box": item["attack"].get("mask_box"),
                **common,
            }
            specs.append(
                self._spec(
                    "quality",
                    unit["model_id"],
                    operator="quality_metrics",
                    first_path=first,
                    second_path=item["path"],
                    **fields,
                )
            )
            if item["role"] == "positive":
                key = (
                    unit["model_id"],
                    unit["image"]["image_id"],
                    item["attack"]["id"],
                    item["attack"]["repeat"],
                )
                if key in clean:
                    fields["comparison"] = "paired_marked_vs_clean_output"
                    specs.append(
                        self._spec(
                            "quality",
                            unit["model_id"],
                            operator="quality_metrics",
                            first_path=clean[key],
                            second_path=item["path"],
                            **fields,
                        )
                    )
        self._dispatch(list({digest(spec): spec for spec in specs}.values()))

    def run(self) -> Path:
        self._start()
        self.planned = []
        units = self._units()
        inputs = self._marked_inputs(units)
        received = self._received(inputs)
        self._decode(received, units)
        self._quality(received, inputs)
        # Unique plans define this report; old records from a different env are not merged in.
        self.planned = list({digest(s): s for s in self.planned}.values())
        write_json(self.store.root / "planned_jobs.json", self.planned)
        from .report import build_report

        build_report(
            self.store,
            self.planned,
            self.config,
            self.inventory(),
            len(units),
            len(inputs),
            len(received),
        )
        return self.store.root
