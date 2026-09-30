"""CPU tests do not claim successful GPU inference or paper reproduction."""
import ast
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from authornight.storage import EvidenceError, cached, commit, disjoint, extract_safe, image_zip, read, write
from authornight.spec import make_plan, generated_batches, seed, secret_text
from authornight.metrics import bits_score, pixels, png_bytes, quality
from authornight.latent import export_function, state_audit
from authornight.distseal import sample_pixels
from authornight.report import aggregate, report
from authornight.attacks import operators

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def planned(tmp_path):
    bank, output, scratch = [tmp_path / p for p in ("bank", "out", "scratch")]
    write(bank / "_inputs/manifest.json", {"images": [{"image_id": i} for i in range(100)]})
    settings = dict(bank=str(bank), output=str(output), scratch=str(scratch), n_images=5,
                    max_hours=8, generated_per_class=1, generation_batch_size=4)
    return settings, make_plan(settings, "test-source")


def test_plan_frozen_and_budget_mutable(planned):
    settings, plan = planned
    assert make_plan({**settings, "max_hours": 9}, "test-source") == plan
    with pytest.raises(EvidenceError):
        make_plan({**settings, "n_images": 6}, "test-source")


def test_unknown_output_rejected(tmp_path):
    root = tmp_path / "out"
    write(root / "important.json", {})
    settings = dict(bank=str(tmp_path/'b'), output=str(root), scratch=str(tmp_path/'s'),
                    n_images=5, generated_per_class=1, generation_batch_size=4, max_hours=8)
    with pytest.raises(ValueError, match="empty OUTPUT"):
        make_plan(settings, "a")


def test_seed_and_balanced_prefix():
    batches = list(generated_batches(2, 16))
    assert [x for b in batches if b['repetition'] == 0 for x in b['labels']] == list(range(1000))
    assert len({b['seed'] for b in batches}) == len(batches)
    assert seed('a', 1) == seed('a', 1) != seed('a', 2)
    assert len(secret_text(1, 0).encode('ascii')) == 7


def test_storage_resume_and_tampering(tmp_path):
    archive = tmp_path/'a.zip'
    image_zip(archive, {'a.png': png_bytes(np.zeros((16, 16, 3), np.uint8))})
    commit(tmp_path, 'records/r.json', 'identity', {'status': 'measured'}, [archive])
    assert cached(tmp_path, 'records/r.json', 'identity')['status'] == 'measured'
    archive.write_bytes(b'changed')
    with pytest.raises(EvidenceError):
        cached(tmp_path, 'records/r.json', 'identity')
    assert cached(tmp_path, 'records/r.json', 'identity', verify=False)
    record = read(tmp_path/'records/r.json')
    record['status'] = 'other'
    write(tmp_path/'records/r.json', record)
    with pytest.raises(EvidenceError):
        cached(tmp_path, 'records/r.json', 'identity', verify=False)


def test_missing_or_wrong_identity(tmp_path):
    assert cached(tmp_path, 'x.json', 'x') is None
    commit(tmp_path, 'x.json', 'a', {})
    with pytest.raises(EvidenceError):
        cached(tmp_path, 'x.json', 'b')


def test_safe_paths_and_archives(tmp_path):
    with pytest.raises(EvidenceError):
        disjoint(tmp_path, tmp_path/'nested')
    for name in ('../escape', '/absolute'):
        path = tmp_path/'bad.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(name, 'bad')
        with pytest.raises(EvidenceError):
            extract_safe(path, tmp_path/'extract')


def test_full_bits_not_prefix():
    target = [0]*100
    score = bits_score([-1]*99+[1], target)
    assert score['ber'] == .01 and not score['exact']
    assert score['word_accuracy_threshold_0_2']
    assert not bits_score([1]*20+[-1]*80, target)['word_accuracy_threshold_0_2']


@pytest.mark.parametrize('logits', [[1], [float('nan'), 1]])
def test_bad_decode_not_ber(logits):
    with pytest.raises(ValueError):
        bits_score(logits, [0, 0])


def test_png_and_quantization():
    tensor = torch.tensor([-.1, 0, .5, 1., 1.1]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    assert pixels(tensor, 'floor')[0, :, 0].tolist() == [0, 0, 127, 255, 255]
    assert pixels(tensor, 'round_even')[0, 2, 0] == 128
    for rule in ('floor', 'round_even', 'half_up'):
        data = pixels(tensor, rule)
        assert np.array_equal(np.array(Image.open(io.BytesIO(png_bytes(data)))), data)


def test_exact_author_sample_serializer():
    x = torch.linspace(-1.1, 1.1, 300).view(1, 3, 10, 10)
    expected = (127.5*x + 128.).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    assert np.array_equal(sample_pixels(x), expected)


def test_author_residual_export_ast():
    function = export_function(ROOT/'reference/ro/inference.py')
    original = Image.fromarray(np.random.default_rng(1).integers(0, 256, (73, 89, 3), dtype=np.uint8))
    cover = torch.zeros(1, 3, 32, 32)
    stego = torch.ones_like(cover)*.1
    expected = np.clip(.1+np.array(original)/127.5-1, -1, 1)*127.5+127.5
    assert np.max(np.abs(function(stego, cover, original).astype(float)-expected.astype(np.uint8))) <= 1


def test_invalid_author_export_rejected(tmp_path):
    path = tmp_path/'wrong.py'
    path.write_text('res = 1\n')
    with pytest.raises(EvidenceError):
        export_function(path)


def test_quality_no_infinite_json():
    x = np.zeros((16, 16, 3), np.uint8)
    result = quality(x, x)
    assert result['identical_pixels'] and result['psnr_db'] is None
    json.dumps(result, allow_nan=False)


def test_state_audit_checks_all_active_weights():
    model = torch.nn.Module()
    model.ae = torch.nn.Linear(2, 2)
    assert state_audit(model, model.state_dict())['all_equal']
    with pytest.raises(EvidenceError):
        state_audit(model, {})


def test_partial_report_and_resume(planned):
    settings, plan = planned
    root = Path(settings['output'])
    score = bits_score([-1]*100, [0]*100)
    data = dict(method='rosteals', profile='raw100', status='measured', image_id=0,
                measurements={'raw_internal': score, 'author_png': score}, checks={'roundtrip': True})
    commit(root, 'rosteals/coco/records/one.json', plan['identity'], data)
    before = report(root)
    after = report(root)
    assert before == after
    row = after['groups']['rosteals/coco/raw100']
    assert row['cases'] == 1 and row['planned'] == 25 and not row['coverage_complete']
    assert after['groups']['distseal/coco/raw64']['cases'] == 0
    assert after['fid']['status'] == 'unavailable'


def test_aggregate_is_bit_weighted():
    rows = [{'measurements': {'a': bits_score([1]*10, [0]*10)}},
            {'measurements': {'a': bits_score([-1]*100, [0]*100)}}]
    assert aggregate(rows)['measurements']['a']['ber'] == 10/110


def test_released_attack_grid_all_variants_cpu():
    torch.set_num_threads(2)
    grid = operators(ROOT/'reference/omni', ROOT/'reference/dist/scripts/attacks.yaml')
    assert len(grid) == 87
    x = torch.rand(1, 3, 64, 64)
    for name, args, op in grid:
        actual, mask = op(x.clone(), None, **args)
        assert actual.ndim == 4 and actual.shape[1] == 3 and torch.isfinite(actual).all(), (name, args)
        assert mask is None


def test_all_modules_static():
    for file in (ROOT/'authornight').glob('*.py'):
        ast.parse(file.read_text('utf-8'), filename=str(file))


def test_inception_dataset_uint8(tmp_path):
    from authornight.inception import GeneratedPNGs
    x = np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    image_zip(tmp_path/'pair.zip', {'image_wm.png': png_bytes(x)})
    dataset = GeneratedPNGs(tmp_path, [('pair.zip', 'image')], '_wm')
    assert len(dataset) == 1
    assert dataset[0].dtype == torch.uint8
    assert np.array_equal(dataset[0].permute(1, 2, 0).numpy(), x)
