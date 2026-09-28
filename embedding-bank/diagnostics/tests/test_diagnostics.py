from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT.parent / "src"), str(PROJECT)]

from bank_diagnostics.core import bit_metrics, cached, freeze, scalar_audit, separate_roots
from wmbench.io import digest, read_json, sha256, write_json
from wmbench.matching import psnr, save_rgb, load_rgb


def test_full_payload_is_not_common30():
    payload = [0] * 64
    bits = payload.copy()
    bits[0] = 1
    result = bit_metrics(bits, payload, list(range(34, 64)))
    assert result["common30_exact"] and not result["native_exact"]
    assert result["error_positions"] == [0]
    bits[63] = 1
    assert bit_metrics(bits, payload, list(range(34, 64)))["common30_errors"] == 1


@pytest.mark.parametrize("indices", [[0] * 30, list(range(35, 65)), list(range(29))])
def test_bad_common_indices_fail(indices):
    with pytest.raises(ValueError):
        bit_metrics([0] * 64, [0] * 64, indices)


def test_bad_decoder_bits_fail():
    with pytest.raises(ValueError):
        bit_metrics([float('nan')] * 30, [0] * 30, list(range(30)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_scalar_histogram_matches_exact_grid(seed):
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    native = np.clip(source.astype(int) + rng.integers(-12, 13, source.shape), 0, 255).astype(np.uint8)
    before = source.copy(), native.copy()
    for target in (39, 42, 45):
        result = scalar_audit(source, native, target)
        z = np.rint(np.clip(source.astype(float) + result["audit_alpha"] * (native.astype(float) - source), 0, 255)).astype(np.uint8)
        assert psnr(source, z) == pytest.approx(result["audit_psnr"], abs=1e-10)
        assert result["audit_absolute_error"] <= abs(result["baseline"]["psnr"] - target) + 1e-10
        assert 0 <= result["audit_alpha"] <= 4
    assert np.array_equal(source, before[0]) and np.array_equal(native, before[1])


def test_zero_residual_does_not_fake_a_match():
    a = np.zeros((8, 8, 3), np.uint8)
    result = scalar_audit(a, a, 42)
    assert not result["within_policy"]
    assert result["finding"] == "zero_residual"


def test_quantized_gap_is_not_silently_relaxed():
    a = np.full((8, 8, 3), 128, np.uint8)
    result = scalar_audit(a, a + 1, 39)
    assert not result["within_policy"]
    assert result["tolerance"] == .05
    assert not result["changed_pixels_individually"]


def test_output_must_not_overlap_bank(tmp_path):
    with pytest.raises(ValueError):
        separate_roots(tmp_path, tmp_path / 'out', tmp_path / 'tmp')
    separate_roots(tmp_path / 'bank', tmp_path / 'out', tmp_path / 'tmp')


def test_resume_detects_corruption(tmp_path):
    file = tmp_path / 'result.png'
    file.write_bytes(b'png')
    path = tmp_path / 'done.json'
    write_json(path, {'signature': 'abc', 'artifacts': [{'path':file.name, 'sha256':sha256(file)}]})
    assert cached(path, 'abc')
    with pytest.raises(ValueError):
        cached(path, 'different')
    file.write_bytes(b'changed')
    with pytest.raises(ValueError):
        cached(path, 'abc')


def test_freeze_does_not_overwrite(tmp_path):
    path = tmp_path / 'spec.json'
    freeze(path, {'seed':1})
    with pytest.raises(ValueError):
        freeze(path, {'seed':2})
    assert read_json(path) == {'seed':1}


def test_stage_keeps_bad_ber_and_retries_only_technical_errors(tmp_path):
    from bank_diagnostics.worker import Stages
    stages = Stages(tmp_path, 's', [0]*30, list(range(30)))
    calls = []
    def bad_bits():
        calls.append(1)
        return {'bits':[1]*30}
    first = stages.measure('bits', bad_bits)
    second = stages.measure('bits', bad_bits)
    assert first == second and len(calls) == 1 and first['native_ber'] == 1
    stages.measure('error', lambda: 1/0)
    fixed = stages.measure('error', lambda: {'bits':[0]*30})
    assert fixed['status'] == 'measured'
    assert list((tmp_path / 'technical_error_history').iterdir())


def test_png_stage_never_overwrites_completed_image(tmp_path):
    from bank_diagnostics.worker import png_once, decoded_png, Stages
    target = tmp_path / 'png.png'
    original = np.zeros((8, 8, 3), np.uint8)
    png_once(target, lambda p: save_rgb(p, original))
    png_once(target, lambda p: save_rgb(p, original + 10))
    assert np.array_equal(load_rgb(target), original)
    stages = Stages(tmp_path, 's', [0]*30, list(range(30)))
    stages.measure('png', lambda: decoded_png(target, lambda: {'bits':[0]*30}))
    save_rgb(target, original + 20)
    with pytest.raises(ValueError, match='Damaged'):
        stages.measure('png', lambda: {'bits':[0]*30})


def test_report_never_calls_inference(tmp_path):
    from bank_diagnostics.controller import report
    case = tmp_path / 'models/fin/cases/000000000285'
    case.mkdir(parents=True)
    write_json(case / 'repeat_png_decode.json', {'stage':'repeat_png_decode', 'status':'measured',
               'common30_errors':0, 'native_errors':1, 'payload_bits':64})
    report(tmp_path)
    text = (tmp_path / 'REPORT.md').read_text()
    assert '| fin | not_run | 1/1 | 0/1 |' in text


def test_historical_audit_is_read_only_and_resumes(tmp_path):
    from bank_diagnostics.controller import audit_matching
    bank, out = tmp_path/'bank', tmp_path/'out'
    x = np.full((8,8,3),128,np.uint8)
    source, native = bank/'input.png', bank/'fin/native/a.png'
    save_rgb(source,x); save_rgb(native,x+1)
    records = bank/'fin/_meta/records'
    write_json(records/'000000000285_native.json', {'signature':'old','status':'ok',
               'output':'fin/native/a.png','artifacts':[{'path':'fin/native/a.png','sha256':sha256(native)}]})
    write_json(records/'000000000285_matched_39.json', {'signature':'old','status':'match_not_found_within_policy',
               'alpha':2.,'psnr':42.,'artifacts':[]})
    before = {str(p):sha256(p) for p in bank.rglob('*') if p.is_file()}
    ctx = dict(bank=bank,output=out,models={'fin':{}},images=[{'image_id':'285','path':'input.png'}],identity='v1')
    audit_matching(ctx)
    audit_matching(ctx)
    after = {str(p):sha256(p) for p in bank.rglob('*') if p.is_file()}
    assert before == after
    assert read_json(out/'matching/summary.json')['checked']==1


def test_worker_reuses_same_historical_payload_and_never_changes_bank(tmp_path, monkeypatch):
    import shutil
    from bank_diagnostics import worker
    from wmbench.embedding_bank import message_for_model
    bank, output, slot = (tmp_path / name for name in ('bank', 'diagnostics', 'scratch'))
    model = {'id':'trustmark_q', 'native_bits':100}
    payload, indices = message_for_model(model, '285')
    image = bank / '_inputs/285.png'
    save_rgb(image, np.full((8,8,3), 128, np.uint8))
    png = bank / 'trustmark_q/native/285.png'
    save_rgb(png, load_rgb(image) + 1)
    meta = bank / 'trustmark_q/_meta/keys/285/metadata.json'
    write_json(meta, {'expected_bits':payload})
    record = {'signature':'old', 'status':'ok', 'payload':payload, 'common_indices':indices,
              'source':image.relative_to(bank).as_posix(), 'output':png.relative_to(bank).as_posix(),
              'metadata':meta.relative_to(bank).as_posix(),
              'artifacts':[{'path':p.relative_to(bank).as_posix(),'sha256':sha256(p)} for p in (png,meta)]}
    write_json(bank/'trustmark_q/_meta/records/000000000285_native.json',record)
    identity = {'registry':model,'weights':{}}
    write_json(bank/'trustmark_q/_meta/model_identity.json',identity)
    config = slot/'config.json'
    write_json(config, {'model':model,'options':{}})
    write_json(output/'prepared.json',{'config':str(config),'identity':identity})
    calls = []
    class FakeAdapter:
        def embed(self, source, target, bits, metadata):
            calls.append(bits)
            shutil.copy2(source,target)
            write_json(metadata,{'expected_bits':bits})
        def decode(self, source, metadata):
            assert not Path(metadata).is_relative_to(bank)
            return {'bits':read_json(metadata)['expected_bits']}
    monkeypatch.setattr(worker,'adapter_instance',lambda config:FakeAdapter())
    monkeypatch.setattr(worker,'runtime_info',lambda:{'kind':'test fixture'})
    monkeypatch.setattr(worker,'seed_everything',lambda seed:None)
    before = {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob('*') if p.is_file()}
    job = {'bank':str(bank),'output':str(output),'slot':str(slot),'model':model,'identity':'test',
           'images':[{'image_id':'285','path':image.relative_to(bank).as_posix(),'sha256':sha256(image)}]}
    worker.run_model(job)
    worker.run_model(job)
    assert calls == [payload]
    after = {p.relative_to(bank).as_posix():sha256(p) for p in bank.rglob('*') if p.is_file()}
    assert before == after
    assert read_json(output/'cases/000000000285/repeat_png_decode.json')['common30_exact']


def test_notebook_cells_and_bundle():
    import nbformat
    import zipfile
    dist = PROJECT/'dist'
    path = dist/'Embedding_Bank_DIAGNOSTICS_RESUMABLE.ipynb'
    if not path.exists():
        pytest.skip('Build notebook first')
    book = nbformat.read(path,as_version=4)
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
    with zipfile.ZipFile(dist/'Embedding_Bank_DIAGNOSTICS_source.zip') as archive:
        for name in archive.namelist():
            assert not name.startswith('/') and '..' not in Path(name).parts
            if name.endswith('.py'):
                ast.parse(archive.read(name),filename=name)
    assert 'PAYLOAD_BASE64' not in path.read_text()


def test_tensor_diagnostics_execute_with_contract_stubs(tmp_path, monkeypatch):
    """Exercise real Torch/PIL stage plumbing; these stubs are NOT model validation."""
    torch = pytest.importorskip('torch')
    from bank_diagnostics.worker import Stages, fin_stages, hidden_stages, distseal_stages
    from wmbench.adapters.extended import _TensorAdapter
    torch.set_num_threads(2)
    source=tmp_path/'input.png'
    save_rgb(source,np.random.default_rng(0).integers(0,256,(512,512,3),dtype=np.uint8))
    a=object.__new__(_TensorAdapter)
    a.torch=torch; a.device='cpu'; a.resolution=128; a.threshold=0
    a.fed=lambda pair,rev=False: (pair[0],torch.ones((1,64))) if rev else (pair[0]+.01,None)
    a.inl=lambda x,rev=False:x
    a._extract_logits=lambda x:torch.ones((1,64))
    stages=Stages(tmp_path/'fin','fin',[1]*64,list(range(34,64))); stages.root.mkdir()
    fin_stages(a,source,stages)
    assert len(stages.rows)==10 and all(r['status']=='measured' for r in stages.rows)
    assert all(r['native_exact'] for r in stages.rows)

    utils=types.ModuleType('wibench.utils')
    utils.resize_torch_img=lambda x,size:torch.nn.functional.interpolate(x[None],size=size,mode='bilinear',align_corners=True)[0]
    utils.normalize_image=lambda x:(x*2-1)[None]
    utils.denormalize_image=lambda x:((x+1)/2)[0]
    monkeypatch.setitem(sys.modules,'wibench',types.ModuleType('wibench'))
    monkeypatch.setitem(sys.modules,'wibench.utils',utils)
    net=types.SimpleNamespace(encoder=lambda x,m:x+.01,decoder=lambda x:torch.ones((1,30)))
    wrap=types.SimpleNamespace(params=types.SimpleNamespace(H=128,W=128),device='cpu',encoder_decoder=net,
                              embed=lambda x,m:x+.01,extract=lambda x,m:np.ones((1,30),int))
    hidden=types.SimpleNamespace(wrapper=wrap,device='cpu',_image=lambda p:a._image(p)[0],
                                 _payload=lambda b:types.SimpleNamespace(watermark=torch.tensor([b])))
    stages=Stages(tmp_path/'hidden','hidden',[1]*30,list(range(30))); stages.root.mkdir()
    hidden_stages(hidden,source,stages)
    assert len(stages.rows)==9 and all(r['status']=='measured' for r in stages.rows)

    a.autoencoder=types.SimpleNamespace(encode=lambda x:types.SimpleNamespace(latent=x),
                                      decode=lambda x:types.SimpleNamespace(sample=x))
    a.model=types.SimpleNamespace(embedder=lambda z,m:z+.01,blender=lambda z,p:p)
    stages=Stages(tmp_path/'dist','dist',[1]*64,list(range(34,64))); stages.root.mkdir()
    distseal_stages(a,source,stages)
    assert len(stages.rows)==4 and all(r['status']=='measured' for r in stages.rows)
    assert np.array_equal(load_rgb(stages.root/'ae_only.png'),load_rgb(source))
