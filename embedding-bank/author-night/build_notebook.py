"""Build a deterministic, readable source ZIP and four-cell Colab notebook."""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "Three_Method_AUTHOR_NIGHT"


def build():
    contents = {}
    for directory, patterns in [('authornight', ['*.py']), ('locks', ['*.txt', '*.in']),
                                 ('tests', ['*.py']), ('reference', ['*']), ('validation', ['*.json'])]:
        for pattern in patterns:
            paths = (ROOT/directory).rglob(pattern) if directory == 'reference' else (ROOT/directory).glob(pattern)
            for path in paths:
                if path.is_file() and '__pycache__' not in path.parts:
                    contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ('README.md', 'build_notebook.py', 'local_source_smoke.py', 'validate_dependencies.py', 'validate_release.py'):
        contents[name] = (ROOT/name).read_bytes()
    for name, data in contents.items():
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8-sig'), filename=name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    settings = '''from pathlib import Path
import hashlib, io, json, os, shutil, subprocess, sys, zipfile

N_COCO = 100                  # Fixed before first run: 5..100 IDs from the old manifest.
MAX_HOURS = 8                 # Queue budget, not estimated completion time.
GEN_PER_CLASS = 50            # 50 = author-scale target 50,000 pairs; 1 = 1,000-image pilot.
GEN_BATCH_SIZE = 4            # Keep unchanged when resuming. AE decodes one image at a time.

try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if os.name == 'nt':
    raise RuntimeError('Use Colab A100/high RAM or a Linux NVIDIA GPU server.')

DRIVE_ROOT = Path('/content/drive/MyDrive') if IN_COLAB else Path('/workspace/data')
BANK_ROOT = DRIVE_ROOT / 'watermarking_native_resolution_canary_v1'  # READ ONLY
OUTPUT_ROOT = DRIVE_ROOT / 'watermarking_author_reproduction_v1'   # NEW results
SCRATCH_ROOT = Path('/content/authornight_scratch_v1') if IN_COLAB else Path('/workspace/authornight-scratch')
SOURCE_ARCHIVE = Path('Three_Method_AUTHOR_NIGHT_source.zip')
SETTINGS = dict(bank=str(BANK_ROOT), output=str(OUTPUT_ROOT), scratch=str(SCRATCH_ROOT),
                n_images=N_COCO, max_hours=MAX_HOURS, generated_per_class=GEN_PER_CLASS,
                generation_batch_size=GEN_BATCH_SIZE)
print(f'COCO: {N_COCO*25} image/message/profile cases. DistSeal generation target: {GEN_PER_CLASS*1000:,} pairs.')
print('No ImageNet validation: FID is unavailable. No COCO substitution.')
print('Old results stay unchanged. New output:', OUTPUT_ROOT)
'''
    bootstrap = '''if IN_COLAB:
    drive.mount('/content/drive')
for i, left in enumerate((BANK_ROOT, OUTPUT_ROOT, SCRATCH_ROOT)):
    for right in (BANK_ROOT, OUTPUT_ROOT, SCRATCH_ROOT)[i+1:]:
        a, b = left.resolve(), right.resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Input, output and scratch directories must not overlap')
if not (BANK_ROOT / '_inputs/manifest.json').is_file():
    raise FileNotFoundError(f'Point BANK_ROOT at the original canary folder: {BANK_ROOT}')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(SCRATCH_ROOT).free < 60 * 1024**3:
    raise RuntimeError('Need 60 GiB free temporary disk before setup. Do not delete saved Drive results.')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True)
print('GPU:', gpu.strip())
if 'A100' not in gpu:
    print('Warning: this author stack targets A100; other GPUs have not been validated.')
EXPECTED_SHA = '__SHA__'
release_cache = OUTPUT_ROOT / '_release' / SOURCE_ARCHIVE.name
if release_cache.is_file() and hashlib.sha256(release_cache.read_bytes()).hexdigest() == EXPECTED_SHA:
    archive_bytes = release_cache.read_bytes()
elif SOURCE_ARCHIVE.is_file():
    archive_bytes = SOURCE_ARCHIVE.read_bytes()
elif IN_COLAB:
    print('Upload Three_Method_AUTHOR_NIGHT_source.zip, not the notebook or model weights.')
    uploaded = files.upload()
    matches = [data for data in uploaded.values() if hashlib.sha256(data).hexdigest() == EXPECTED_SHA]
    if len(matches) != 1:
        raise ValueError('Upload the source ZIP supplied with THIS notebook')
    archive_bytes = matches[0]
else:
    raise FileNotFoundError(SOURCE_ARCHIVE)
if hashlib.sha256(archive_bytes).hexdigest() != EXPECTED_SHA:
    raise ValueError('Source archive SHA256 mismatch. Use the notebook and ZIP from the same release.')
release_cache.parent.mkdir(parents=True, exist_ok=True)
if not release_cache.exists():
    temporary = release_cache.with_suffix('.upload.partial')
    temporary.write_bytes(archive_bytes)
    temporary.replace(release_cache)
elif hashlib.sha256(release_cache.read_bytes()).hexdigest() != EXPECTED_SHA:
    raise ValueError('A different release is already cached in this OUTPUT_ROOT. Use a new output folder.')
PROJECT = SCRATCH_ROOT / ('release-' + EXPECTED_SHA[:16])
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    for entry in archive.infolist():
        target = (PROJECT / entry.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()) or target == PROJECT.resolve():
            raise ValueError('Unsafe source archive path')
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Source archive symlink rejected')
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.15'], check=True)
# System dependencies for the unmodified Wand-based corruption libraries and BCH build.
if IN_COLAB:
    subprocess.run(['apt-get', '-qq', 'update'], check=True)
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'libmagickwand-dev', 'build-essential', 'libgl1'], check=True)
from authornight.spec import make_plan
PLAN = make_plan(SETTINGS, EXPECTED_SHA)
print('READY. Run the next cell. Plan:', PLAN['identity'])
print('Allow up to ~120 GiB on Drive for the complete 50k-pair target; an 8h run can be partial.')
'''.replace('__SHA__', checksum)
    run = '''from authornight.runner import run
from authornight.report import report_bundle
try:
    SUMMARY = run(SETTINGS, EXPECTED_SHA)
finally:
    if (OUTPUT_ROOT / 'plan.json').is_file():
        REPORT_ZIP = report_bundle(OUTPUT_ROOT)
        print('Saved partial/final report:', REPORT_ZIP)
'''
    reporting = '''from authornight.report import report, report_bundle
from IPython.display import Markdown, display, FileLink
SUMMARY = report(OUTPUT_ROOT)
display(Markdown((OUTPUT_ROOT / '_reports/report.md').read_text()))
REPORT_ZIP = report_bundle(OUTPUT_ROOT)
print('Send this file for analysis:', REPORT_ZIP)
if IN_COLAB:
    files.download(str(REPORT_ZIP))
else:
    display(FileLink(str(REPORT_ZIP)))
'''
    sections = [
        ('markdown', '# Three-method author-path overnight check\n\n'
         '**RoSteALS · SepWater · DistSeal.** 100 COCO images, complete payloads, author configurations and PNG export. '
         'Separate class-conditional DistSeal generation and robustness tests. **A100 + high RAM.**\n\n'
         'A new experiment; historical bank and production adapters remain untouched. No matched PSNR or retraining. '
         'The default generation target is 50,000 pairs, not a promise to finish it in eight hours. '
         'ImageNet validation is absent: no FID claim. The local release has CPU/static checks, not completed GPU validation.'),
        ('markdown', '## 1. Settings\n\nKeep scientific settings unchanged when resuming. MAX_HOURS may be changed.'),
        ('code', settings),
        ('markdown', '## 2. Drive, readable sources and setup\n\nUpload the matching source ZIP when asked. '
         'Authorize Drive before leaving. Old files are read-only; all new evidence has its own output directory.'),
        ('code', bootstrap),
        ('markdown', '## 3. Run / resume\n\nHeartbeat every 30 seconds. Logs and completed cases go directly to Drive. '
         'An interrupted case is retried, completed verified cases are reused. Stages have time limits and can be partial. '
         'If Colab ends, reconnect with the same GPU and run cells 1–3 again; no keep-alive bypass is used.'),
        ('code', run),
        ('markdown', '## 4. Read and download the report\n\nCan be used on partial results. Check coverage and technical failures before reading BER. '
         'Full protocol, deviations and source links are in README.md inside the ZIP.'),
        ('code', reporting),
    ]
    cells = []
    for index, (kind, text) in enumerate(sections):
        cell = dict(cell_type=kind, id=f'author-night-{index:02}', metadata={}, source=text.splitlines(True))
        if kind == 'code':
            compile(text, f'cell_{index}', 'exec')
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
        'language_info': dict(name='python'), 'accelerator': 'GPU'})
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    (output/(NAME+'_source.zip')).write_bytes(payload)
    (output/(NAME+'.ipynb')).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    (output/'build.json').write_text(json.dumps(dict(sha256=checksum, bytes=len(payload),
        code_cells=4, python_files=sum(x.endswith('.py') for x in contents), files=len(contents)), indent=2)+'\n')
    print('Built', NAME, 'source bytes:', len(payload), 'SHA256:', checksum)


if __name__ == '__main__':
    build()
