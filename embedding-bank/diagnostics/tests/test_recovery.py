"""Recovery packaging and dependency fixes, without claiming GPU validation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent / 'src'), str(ROOT)]

from bank_diagnostics.dependency_fixes import repair_model, constrained_builds
from wmbench.embedding_bank import bank_registry


@pytest.mark.parametrize('name', ['editguard', 'sepwater', 'chunkyseal'])
def test_repairs_are_scoped_and_do_not_mutate_registry(name):
    model = bank_registry(ROOT.parent)[name]
    before = copy.deepcopy(model)
    repaired = repair_model(model)
    assert model == before
    assert repaired['source'] == model['source']
    assert repaired.get('options') == model.get('options')
    if name == 'chunkyseal':
        assert repaired == model
    else:
        assert 'setuptools==70.3.0' in repaired['requirements']
        assert 'validated_lock' not in repaired
    if name == 'editguard':
        assert repaired['build_requirements'] == ['basicsr==1.4.2']


def test_build_constraints_restore_command_on_error(tmp_path, monkeypatch):
    from wmbench import environment
    calls = []
    original = lambda args, cwd=None, env=None: calls.append(args)
    monkeypatch.setattr(environment, 'command', original)
    lock = tmp_path / 'requirements.lock'
    lock.write_text('torch==2.5.1\n')
    with pytest.raises(RuntimeError):
        with constrained_builds(lock):
            environment.command(['uv', 'pip', 'sync'])
            environment.command(['uv', 'pip', 'install', '--no-build-isolation', 'basicsr==1.4.2'])
            raise RuntimeError('test cleanup')
    assert environment.command is original
    assert calls[0] == ['uv', 'pip', 'sync']
    assert calls[1][-2:] == ['--constraint', str(lock)]


def test_recovery_notebook_and_bundle():
    dist = ROOT / 'dist'
    path = dist / 'Embedding_Bank_DIAGNOSTICS_RECOVERY.ipynb'
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    cells = [c for c in book.cells if c.cell_type == 'code']
    assert len(cells) == 4
    for cell in cells:
        ast.parse(cell.source)
        assert not cell.outputs and cell.execution_count is None
    settings = cells[0].source
    assert 'models=["editguard", "sepwater", "chunkyseal"]' in settings
    assert 'audit_matching=False' in settings
    assert 'keep_failed_scratch=True' in settings
    assert '"chunkyseal": 10800' in settings
    assert 'watermarking_embedding_diagnostics_recovery_v1' in settings
    assert 'watermarking_embedding_diagnostics_v1' in settings
    archive = dist / 'Embedding_Bank_DIAGNOSTICS_RECOVERY_source.zip'
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert checksum in cells[1].source
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for name in names:
            assert '..' not in Path(name).parts and not name.startswith('/')
            if name.endswith('.py'):
                ast.parse(z.read(name), filename=name)
        shim_paths = [n for n in names if n.endswith('bank_diagnostics/dependency_fixes.py')]
        assert len(shim_paths) == 2
        assert z.read(shim_paths[0]) == z.read(shim_paths[1])
        # Original inference modules must be byte-for-byte unchanged in the bundle.
        for source in (ROOT.parent / 'src/wmbench').rglob('*.py'):
            suffix = source.relative_to(ROOT.parent).as_posix()
            matches = [n for n in names if n.endswith('/' + suffix) or n == suffix]
            assert len(matches) == 1
            assert z.read(matches[0]) == source.read_bytes()
    assert 'PAYLOAD_BASE64' not in path.read_text(encoding='utf-8')


def test_real_grayscale_alias_if_torchvision_available():
    pytest.importorskip('torchvision')
    from bank_diagnostics.dependency_fixes import install_basicsr_torchvision_compat
    import torchvision.transforms._functional_tensor as actual
    previous = sys.modules.get('torchvision.transforms.functional_tensor')
    try:
        sys.modules.pop('torchvision.transforms.functional_tensor', None)
        install_basicsr_torchvision_compat()
        assert sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale is actual.rgb_to_grayscale
    finally:
        sys.modules.pop('torchvision.transforms.functional_tensor', None)
        if previous is not None:
            sys.modules['torchvision.transforms.functional_tensor'] = previous
