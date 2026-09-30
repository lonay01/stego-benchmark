"""Check distributed bytes and every cell, without running a GPU experiment."""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    nb = json.loads((ROOT/'dist/Three_Method_AUTHOR_NIGHT.ipynb').read_text('utf-8'))
    archive_path = ROOT/'dist/Three_Method_AUTHOR_NIGHT_source.zip'
    payload = archive_path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((ROOT/'dist/build.json').read_text())
    assert sha == manifest['sha256']
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 4 and sha in cells[1]
    assert not any('PAYLOAD_BASE64' in c for c in cells)
    for index, cell in enumerate(cells):
        compile(cell, f'notebook-cell-{index}', 'exec')
    modules = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert not any('download-audit' in name or '__pycache__' in name for name in names)
        for name in names:
            if name.endswith('.py'):
                ast.parse(archive.read(name).decode('utf-8-sig'), filename=name)
                modules += 1
        for name in ('rosteals', 'sepwater', 'distseal'):
            assert archive.read('locks/'+name+'.txt') == (ROOT/'locks'/(name+'.txt')).read_bytes()
        assert 'authornight/inception.py' in names
    assert modules == manifest['python_files']
    print(json.dumps({'code_cells_compiled': len(cells), 'python_files_parsed': modules,
                      'zip_sha256': sha, 'gpu_run': False}, indent=2))


if __name__ == '__main__':
    main()
