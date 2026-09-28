"""Synthetic contracts/CPU tensor tests; never claimed as real-model results."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from wmbench.io import read_json, sha256, write_json
from wmbench.matching import load_rgb, save_rgb
from wmbench_full_payload.storage import EvidenceError
from wmbench_native_canary.contracts import models_for_trial
from wmbench_native_canary.worker import process_pair
from wmbench_native_diagnostics import core, probes, worker
from wmbench_native_diagnostics.report import report


class PixelBits:
    """Integration fixture; stores bits in pixels, not a watermark research model."""
    def __init__(self):
        self.embeds = self.decodes = 0

    def embed(self, source, output, payload, metadata):
        self.embeds += 1
        pixels = load_rgb(source).copy()
        pixels.reshape(-1)[:len(payload)] = np.array(payload) * 255
        save_rgb(output, pixels)
        write_json(metadata, {"expected_bits": payload, "native_bits": len(payload)})

    def decode(self, source, metadata):
        self.decodes += 1
        data = read_json(metadata)
        assert not any(data['expected_bits']), "Answer was exposed to decoder"
        return {"bits": (load_rgb(source).reshape(-1)[:data['native_bits']] > 127).astype(int).tolist()}


def historical_case(tmp_path, project, name="hidden_benchmark"):
    bank, output = tmp_path / "old", tmp_path / "new"
    image_path = bank / "_inputs/images/000000000001.png"
    image_path.parent.mkdir(parents=True)
    yy, xx = np.indices((512, 512))
    array = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(array).save(image_path)
    image = {"image_id": "1", "path": image_path.relative_to(bank).as_posix(), "sha256": sha256(image_path)}
    model = models_for_trial(project)[name]
    adapter = PixelBits()
    for rep in range(1, 6):
        process_pair(adapter, model, bank, image, rep, {"runtime": {"kind": "fixture"}})
    cases = core.freeze_cases(bank, output, model, [image])
    return bank, output, model, cases


@pytest.mark.parametrize("name", core.METHODS)
def test_freeze_preserves_same_full_payload_and_history(tmp_path, project, name):
    bank, root, model, cases = historical_case(tmp_path, project, name)
    before = {p.relative_to(bank).as_posix(): sha256(p) for p in bank.rglob('*') if p.is_file()}
    assert len(cases) == 5
    assert len(cases[0]['assignment']['payload']) == model['native_bits']
    assert len({tuple(c['assignment']['payload']) for c in cases}) == 5
    core.freeze_cases(bank, root, model, [cases[0]['image']])
    assert before == {p.relative_to(bank).as_posix(): sha256(p) for p in bank.rglob('*') if p.is_file()}
    assert not core.complete(root, cases[0], core.case_key('test', cases[0]))


def test_changed_copied_evidence_is_not_overwritten(tmp_path, project):
    _, root, _, cases = historical_case(tmp_path, project)
    path = core.validate_case(root, cases[0]) / 'historical.png'
    path.write_bytes(b'damaged')
    with pytest.raises(EvidenceError):
        core.validate_case(root, cases[0])


def test_quantization_and_clipping():
    raw = np.array([-1, 0, .5, 1, 2], dtype=np.float32)
    assert probes.quantize(raw, 'round_even').tolist() == [0, 0, 128, 255, 255]
    assert probes.quantize(raw, 'floor').tolist() == [0, 0, 127, 255, 255]
    assert probes.quantize(raw, 'half_up').tolist() == [0, 0, 128, 255, 255]
    half = np.array([2.5, 3.5], dtype=np.float32) / 255
    assert probes.quantize(half, 'round_even').tolist() == [2, 4]
    assert probes.quantize(half, 'half_up').tolist() == [3, 4]


def test_png_lossless_and_conflict_detection(tmp_path):
    pixels = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    path = tmp_path / 'test.png'
    worker.freeze_png(path, pixels)
    assert np.array_equal(load_rgb(path), pixels)
    worker.freeze_png(path, pixels)
    with pytest.raises(EvidenceError):
        worker.freeze_png(path, pixels ^ 1)


def test_measured_bit_errors_are_not_retried(tmp_path):
    assignment = {'payload': [1, 0], 'capacity': {'information_indices': [0, 1],
        'information_bits': 2, 'reserved_bits': {}, 'interface': 'raw_message'}}
    stages = core.Stages(tmp_path, tmp_path / 'case', 'key', assignment, {})
    first = stages.measure('probe', lambda: {'bits': [0, 0]})
    assert first['ber'] == .5
    assert stages.measure('probe', lambda: pytest.fail('Unexpected retry')) == first


def test_technical_error_is_archived(tmp_path):
    assignment = {'payload': [1], 'capacity': {'information_indices': [0],
        'information_bits': 1, 'reserved_bits': {}, 'interface': 'raw_message'}}
    stages = core.Stages(tmp_path, tmp_path / 'case', 'key', assignment, {})
    assert stages.measure('probe', lambda: 1/0)['status'] == 'error'
    assert stages.measure('probe', lambda: {'bits': [1]})['exact']
    assert list(tmp_path.rglob('technical_error_history/*.json'))


def test_hidden_hook_preserves_output_rng_and_restores():
    class Encoder(torch.nn.Module):
        def forward(self, image, message):
            return image + torch.rand_like(image) * .01
    class Wrapper:
        def __init__(self):
            self.encoder_decoder = SimpleNamespace(encoder=Encoder())
        def embed(self, image, message):
            return (self.encoder_decoder.encoder(image, message) + 1).clamp(0, 2) / 2
    adapter = SimpleNamespace(wrapper=Wrapper())
    inputs = torch.zeros(1, 3, 8, 8)
    torch.manual_seed(4)
    baseline = adapter.wrapper.embed(inputs, None)
    rng = torch.get_rng_state()
    torch.manual_seed(4)
    with probes.capture_forward(adapter, 'hidden_benchmark') as captures:
        actual = adapter.wrapper.embed(inputs, None)
    assert torch.equal(baseline, actual) and torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(captures['raw'], probes.pixels(baseline))
    assert 'embed' not in vars(adapter.wrapper)
    assert not adapter.wrapper.encoder_decoder.encoder._forward_hooks


@pytest.mark.parametrize('name', ['rosteals', 'sepwater'])
def test_latent_is_captured_without_resampling(name):
    class Network:
        def encode_first_stage(self, x):
            return x + torch.rand_like(x)
        def decode_first_stage(self, x):
            return x * 2
    network = Network()
    wrapper = SimpleNamespace(model=network, embed=lambda x, m: x)
    adapter = SimpleNamespace(wrapper=wrapper, _load=lambda: network)
    torch.manual_seed(1)
    with probes.capture_forward(adapter, name) as values:
        latent = network.encode_first_stage(torch.zeros(1, 3, 8, 8))
        raw = network.decode_first_stage(latent)
    assert torch.equal(latent, values['latent'])
    control = probes.reconstruct_same_latent(adapter, name, values['latent'])
    assert np.array_equal(control, probes.pixels((raw+1)/2))
    assert 'decode_first_stage' not in vars(network)


@pytest.mark.parametrize('name', ['sepmark', 'distseal_posthoc_dcae'])
def test_save_boundary_does_not_change_original_writer(name):
    class AE:
        def encode(self, tensor):
            return SimpleNamespace(latent=tensor)
    class Adapter:
        autoencoder = AE()
        def _save(self, tensor, path):
            return tensor.clone()
    adapter = Adapter()
    raw = torch.rand(1, 3, 8, 8)
    with probes.capture_forward(adapter, name) as values:
        if name == 'distseal_posthoc_dcae':
            adapter.autoencoder.encode(raw)
        result = adapter._save(raw, 'not-a-file')
    assert torch.equal(raw, result)
    assert np.array_equal(values['raw'], probes.pixels(raw))
    assert '_save' not in vars(adapter)


@pytest.mark.parametrize('name', core.METHODS)
def test_cpu_end_to_end_resume_report(tmp_path, project, monkeypatch, name):
    bank, root, model, cases = historical_case(tmp_path, project, name)
    case = cases[0]
    adapter = PixelBits()
    length = len(case['assignment']['payload'])
    @contextlib.contextmanager
    def fake_capture(adapter, selected):
        observed = {}
        original = adapter.embed
        def embed(source, output, payload, metadata):
            original(source, output, payload, metadata)
            observed['raw'] = load_rgb(output).astype(np.float32) / 255
            if selected in ('hidden_benchmark', 'rosteals'):
                observed['wrapper'] = observed['raw'].copy()
            if selected in ('rosteals', 'sepwater', 'distseal_posthoc_dcae'):
                observed['ae'] = load_rgb(source).astype(np.float32) / 255
            if selected == 'sepwater':
                observed['signed'] = observed['raw'] * 2 - 1
        adapter.embed = embed
        try:
            yield observed
        finally:
            adapter.embed = original
    monkeypatch.setattr(worker, 'capture_forward', fake_capture)
    monkeypatch.setattr(worker, 'decode_float', lambda a, n, image: {
        'bits': (image.reshape(-1)[:length] > .5).astype(int).tolist()})
    worker.process_case(adapter, model, root, case, 'test-run', {'runtime': {'kind': 'CPU-test-double'}})
    assert core.complete(root, case, core.case_key('test-run', case))
    counts = adapter.embeds, adapter.decodes
    worker.process_case(adapter, model, root, case, 'test-run', {})
    assert counts == (adapter.embeds, adapter.decodes)
    folder = root / name / 'cases' / case['stem']
    assert read_json(folder / 'stages/round_even_png.json')['png_pixels_exact']
    assert read_json(folder / 'stages/baseline_png.json')['same_pixels_as_history']
    write_json(root / 'plan.json', {'cases': {name: [case]}})
    table = report(root)
    assert table[0]['status'] == 'diagnostics_complete'
    assert read_json(root / '_reports/summary.json')['all_diagnostics_complete']


def test_incomplete_report_never_passes(tmp_path):
    write_json(tmp_path / 'plan.json', {'cases': {'hidden_benchmark': [{'stem': 'missing'}]}})
    report(tmp_path)
    assert not read_json(tmp_path / '_reports/summary.json')['all_diagnostics_complete']


def test_notebook_schema_and_source_hash():
    import nbformat
    import hashlib
    root = Path(__file__).resolve().parents[1] / 'dist'
    manifest = read_json(root / 'build.json')
    notebook = nbformat.read(root / manifest['notebook'], as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    assert len(code) == 4
    for cell in code:
        ast.parse(cell)
        assert 'PAYLOAD_BASE64' not in cell
    assert hashlib.sha256((root / manifest['archive']).read_bytes()).hexdigest() == manifest['sha256']
    assert manifest['sha256'] in code[1]


def test_model_audit_rejects_training_modules():
    adapter = SimpleNamespace(wrapper=SimpleNamespace(encoder_decoder=torch.nn.Linear(2, 2)))
    with pytest.raises(EvidenceError, match='training'):
        probes.audit_model(adapter, 'hidden_benchmark')


def test_controller_creates_lease_parent_and_never_prepares_completed(tmp_path, monkeypatch):
    from wmbench_native_diagnostics import runner
    bank, out, scratch, project = [tmp_path / n for n in ('old', 'new', 'scratch', 'release')]
    for path in (bank, scratch, project):
        path.mkdir()
    case = {'model': 'hidden_benchmark'}
    monkeypatch.setattr(runner, 'context', lambda p, s: (bank, out, scratch,
        {'hidden_benchmark': {'id': 'hidden_benchmark'}}, {'identity': 'test', 'cases': {'hidden_benchmark': [case]}}))
    monkeypatch.setattr(runner, 'complete', lambda *a: True)
    monkeypatch.setattr(runner, 'report', lambda *a: [])
    runner.run(project, {'bank': str(bank), 'output': str(out), 'scratch': str(scratch)})
    assert not (out / '_reports/active_controller.json').exists()


@pytest.mark.parametrize('name', core.METHODS)
def test_real_tensor_decoder_domains_and_thresholds(name):
    # Actual PyTorch operators, not a pretrained method: exercise decoder domains.
    n = 30 if name == 'hidden_benchmark' else 64
    expected = [0, 1] * (n // 2)
    image = np.zeros((128, 128, 3), dtype=np.float32)
    # Decoder traverses CHW; set its first row/channel intentionally.
    image[0, :n, 0] = expected
    class Decoder(torch.nn.Module):
        def forward(self, x):
            x = x.flatten(1)[:, :n]
            return (x+1)/2 if name == 'hidden_benchmark' else x
    decoder = Decoder().eval()
    network = SimpleNamespace(decoder=decoder)
    adapter = SimpleNamespace(device='cpu', wrapper=SimpleNamespace(encoder_decoder=network, model=network),
        _load=lambda: network, threshold=0.,
        _extract_logits=lambda x: decoder(x*2-1))
    result = probes.decode_float(adapter, name, image)
    assert result['bits'] == expected


def test_sepwater_author_export_uses_signed_tensor():
    source = np.full((8, 8, 3), 127, dtype=np.uint8)
    signed = np.zeros((8, 8, 3), dtype=np.float32)
    expected = (np.clip(np.clip(signed, -1, 1) - (source.astype(np.float32)/127.5-1)
               + source.astype(np.float64)/127.5-1, -1, 1)*127.5+127.5).astype(np.uint8)
    assert np.array_equal(probes.export_sepwater_author(signed, source), expected)


def test_context_freezes_five_images_and_rejects_changed_model(tmp_path, project):
    from wmbench_native_diagnostics.runner import context
    bank, output, model, cases = historical_case(tmp_path, project)
    images = [cases[0]['image']]
    # Five deterministic fixture images; all 25 old message records must exist.
    original = bank / images[0]['path']
    for index in range(2, 6):
        path = bank / f'_inputs/images/{index:012d}.png'
        path.write_bytes(original.read_bytes())
        image = {'image_id': str(index), 'path': path.relative_to(bank).as_posix(), 'sha256': sha256(path)}
        images.append(image)
        for rep in range(1, 6):
            process_pair(PixelBits(), model, bank, image, rep, {'runtime': {'kind': 'fixture'}})
    write_json(bank / '_inputs/manifest.json', {'images': images})
    trial = {'policy': {'schema': 'native-resolution-canary-v1'}, 'models': models_for_trial(project)}
    write_json(bank / 'trial.json', trial)
    meta = bank / model['id'] / '_meta'
    for file in ['model_identity.json', 'dependency_identity.json', 'environment/request.json']:
        write_json(meta / file, {'fixture': True})
    (meta / 'environment/requirements.lock').write_text('numpy==1.26.4\n')
    settings = {'bank': str(bank), 'output': str(output), 'scratch': str(tmp_path/'scratch'), 'models': [model['id']]}
    result = context(project, settings)
    assert len(result[-1]['cases'][model['id']]) == 25
    assert result[-1] == context(project, settings)[-1]
    trial['models'][model['id']]['native_bits'] = 31
    write_json(bank / 'trial.json', trial)
    with pytest.raises(EvidenceError, match='definition changed'):
        context(project, settings)
