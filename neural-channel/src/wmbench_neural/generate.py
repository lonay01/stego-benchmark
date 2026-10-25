"""GPU batches overlap CPU metrics, PNG encoding and verified Drive writes."""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import time
import numpy as np
from PIL import Image

from wmbench.io import read_json, atomic_bytes, write_json, sha256
from wmbench_full_payload.storage import freeze, inside, EvidenceError
from .channel import NeuralChannel, PerceptualMetric, grid
from .spec import ATTACKS, batches, seed, digest
from .storage import cached, save_package, terminal_index


def pixel_quality(left, right):
    from skimage.metrics import structural_similarity
    delta = left.astype(np.float64)-right.astype(np.float64)
    mse = float(np.mean(delta**2))
    mask = np.zeros(left.shape[:2], dtype=bool)
    mask[128:384, 128:384] = True
    return dict(psnr=float(10*np.log10(255**2/mse)) if mse else "Infinity",
        ssim=float(structural_similarity(left, right, data_range=255, channel_axis=2,
                   gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
        mae_inside_mask=float(np.abs(delta)[mask].mean()), mae_outside_mask=float(np.abs(delta)[~mask].mean()))


def load_images(bank, cases, cache=None):
    images = []
    for case in cases:
        source = inside(Path(bank), case["output"])
        target = inside(Path(cache), case["id"]+".png") if cache else source
        data = target.read_bytes() if target.exists() else source.read_bytes()
        if hashlib.sha256(data).hexdigest() != case["carrier_sha256"]:
            raise EvidenceError("Carrier changed before neural attack")
        if cache and not target.exists():
            atomic_bytes(target, data)
        with Image.open(io.BytesIO(data)) as image:
            images.append(grid(image))
    return images


def persist(job, cases, attack, images, outputs, details, perceptual, identity):
    entries, pngs = {}, {}
    execution_path = "_execution/"+attack["engine"]+".json"
    freeze(Path(job["results"]) / execution_path, identity)
    execution = dict(path=execution_path, identity_sha256=digest(identity))
    for i, case in enumerate(cases):
        common = dict(case_key=case["key"], seed=seed(case, attack), parameters=attack,
                      execution=execution, timing=details[i], input_carrier_sha256=case["carrier_sha256"],
                      channel_input_pixel_sha256=hashlib.sha256(np.asarray(images[i]).tobytes()).hexdigest())
        if outputs[i] is None:
            entries[case["id"]] = {**common, "status": "blocked_generation", "reason": "pipeline safety checker"}
            continue
        array = outputs[i]
        name = case["mode"] + "/" + case["id"] + ".png"
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG", compress_level=3)
        data = stream.getvalue()
        quality = pixel_quality(np.asarray(images[i]), array)
        if attack["engine"] != "inpaint":
            quality.pop("mae_inside_mask")
            quality.pop("mae_outside_mask")
        entries[case["id"]] = {**common, "status": "generated", "member": name,
            "sha256": hashlib.sha256(data).hexdigest(), "attack_quality": {**quality, "lpips": perceptual[i]}}
        pngs[name] = data
        # Native-size conversion happens in ONE common Pillow environment,
        # not in 29 decoder environments with different Pillow versions.
        if case["size"] == [512, 512]:
            decoder_name, decoder_data = name, data
        else:
            decoder_name = "decoder-input/"+name
            native = io.BytesIO()
            grid(Image.fromarray(array), tuple(case["size"])).save(native, format="PNG", compress_level=3)
            decoder_data = native.getvalue()
            pngs[decoder_name] = decoder_data
        entries[case["id"]].update(decoder_member=decoder_name,
            decoder_sha256=hashlib.sha256(decoder_data).hexdigest())
    result = save_package(job, cases, attack, entries, pngs)
    print(f"WM_DIAG NEURAL SAVED {job['model']['id']} {attack['id']} image {cases[0]['image_id']}: {len(cases)} outputs; {result['bytes']/1024**2:.1f} MiB", flush=True)


def prepare_geometry(job):
    """Auxiliary resize-only controls, made in common Pillow 10.4 before decoding."""
    import PIL
    if PIL.__version__ != "10.4.0":
        raise RuntimeError("Common channel geometry requires the pinned Pillow 10.4.0")
    root = Path(job["results"])
    cases = read_json(root / job["model"]["id"] / "cases.json")
    path = root / job["model"]["id"] / "geometry-inputs.json"
    evidence = read_json(path) if path.exists() else {}
    for index, case in enumerate(cases, 1):
        if time.time() >= job["deadline"]:
            raise TimeoutError("Geometry preparation budget reached")
        image = load_images(job["bank"], [case], Path(job["slot"]) / "tmp/neural-inputs")[0]
        target = inside(Path(job["slot"]) / "tmp/neural-resize", case["id"]+".png")
        data = io.BytesIO()
        grid(image, tuple(case["size"])).save(data, format="PNG", compress_level=3)
        atomic_bytes(target, data.getvalue())
        value = dict(case_key=case["key"], sha256=sha256(target), size=case["size"], pillow=PIL.__version__)
        if case["id"] in evidence and evidence[case["id"]] != value:
            raise EvidenceError("Fixed resize-only control changed; previous evidence retained")
        evidence[case["id"]] = value
        if index % 20 == 0:
            print(f"WM_DIAG NEURAL GEOMETRY {index}/{len(cases)}", flush=True)
    write_json(path, evidence)


def run(job):
    root = Path(job["results"])
    # This file intentionally contains NO expected payload, key state or decoder metadata.
    cases = read_json(root / job["model"]["id"] / "attack-cases.json")
    groups = defaultdict(list)
    for case in cases:
        groups[case["image_id"]].append(case)
    attacks = [a for a in ATTACKS[1:] if a["engine"] == job["engine"]]
    done = terminal_index(root, cases)
    tasks = [(a, values) for a in attacks for values in groups.values()
             if not all(a["id"] in done[c["id"]] for c in values) and not cached(job, values, a)]
    if not tasks:
        print("REUSE all committed/cached neural outputs", flush=True)
        return
    channel = NeuralChannel(job["engine"], Path(job["scratch"]) / "hf")
    freeze(root / "_execution" / (job["engine"]+".json"), channel.identity)
    metric = PerceptualMetric()
    freeze(root / "_execution/attack-quality.json", metric.identity)
    captions = read_json(root / "_inputs/captions.json")["captions"]
    width = job["settings"]["gpu_batch_size"]
    queue = deque()
    input_cache = Path(job["slot"]) / "tmp/neural-inputs"
    # Bounded: at most two 20-carrier packages await CPU metrics/Drive; no
    # unbounded host-memory queue while the GPU continues generating images.
    with ThreadPoolExecutor(max_workers=1) as reader, ThreadPoolExecutor(max_workers=2) as writers:
        pending_input = reader.submit(load_images, job["bank"], tasks[0][1], input_cache)
        for index, (attack, group) in enumerate(tasks):
            if time.time() >= job["deadline"]:
                raise TimeoutError("Session budget reached; saved packages and results retained")
            images = pending_input.result()
            if index+1 < len(tasks):
                pending_input = reader.submit(load_images, job["bank"], tasks[index+1][1], input_cache)
            outputs, details, perceptual = [], [], []
            for indexes, count in batches(list(range(len(group))), width):
                if time.time() >= job["deadline"]:
                    raise TimeoutError("Session budget reached during generation")
                values, flags, timing = channel.apply([images[i] for i in indexes], attack,
                    [seed(group[i], attack) for i in indexes], [captions[str(group[i]["image_id"])]["text"] for i in indexes])
                valid = [i for i in range(count) if values[i] is not None]
                scores = metric.compare([np.asarray(images[indexes[i]]) for i in valid], [values[i] for i in valid]) if valid else []
                score_map = dict(zip(valid, scores))
                outputs.extend(values[:count])
                details.extend([{**timing, "batch_real_items": count, "padding_items": width-count} for _ in range(count)])
                perceptual.extend([score_map.get(i) for i in range(count)])
                print(f"WM_DIAG NEURAL GPU {job['engine']} package {index+1}/{len(tasks)}, {len(outputs)}/{len(group)} items: {timing['batch_seconds']:.1f}s/batch", flush=True)
            while len(queue) >= 2:
                queue.popleft().result()
            queue.append(writers.submit(persist, job, group, attack, images, outputs, details, perceptual, channel.identity))
        for future in queue:
            future.result()
