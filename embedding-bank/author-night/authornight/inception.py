"""Inception Score is available without ImageNet validation; FID is not."""
import io
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import torch
from .storage import cached, commit, digest, read


class GeneratedPNGs(torch.utils.data.Dataset):
    def __init__(self, root, samples, suffix):
        self.root, self.samples, self.suffix = Path(root), samples, suffix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        archive, name = self.samples[index]
        with zipfile.ZipFile(self.root/archive) as container:
            raw = container.read(name + self.suffix + '.png')
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy()
        return torch.from_numpy(pixels).permute(2, 0, 1)


def run(job):
    from torch_fidelity import calculate_metrics
    root = Path(job['output'])
    plan = read(root/'plan.json')
    samples, records = [], []
    for path in sorted((root/'distseal/generated/records').glob('*.json')):
        row = cached(root, path.relative_to(root).as_posix(), plan['identity'])
        archive = next(iter(row['artifacts']))
        samples.extend((archive, x['name']) for x in row['samples'])
        records.append(row['record_sha256'])
    if len(samples) < 1000:
        print('Inception Score pending: fewer than 1,000 generated samples', flush=True)
        return
    snapshot = digest(records)
    for name, suffix in (('unmarked', '_clean'), ('marked', '_wm')):
        relative = f'distseal/inception/{snapshot}_{name}.json'
        if cached(root, relative, plan['identity']):
            continue
        metrics = calculate_metrics(input1=GeneratedPNGs(root, samples, suffix), cuda=True,
            isc=True, fid=False, kid=False, ppl=False, isc_splits=10, batch_size=16,
            rng_seed=20260930, cache=False, verbose=True)
        commit(root, relative, plan['identity'], {
            'status': 'measured', 'variant': name, 'samples': len(samples), 'record_snapshot': snapshot,
            'metrics': {k: float(v) for k, v in metrics.items()}, 'torch_fidelity': '0.3.0',
            'isc_splits': 10, 'partial_sample_count': len(samples) != 50000,
            'note': 'IS depends on sample count and class coverage; partial IS is not the paper 50k result'})
        print('SAVED Inception Score', name, len(samples), metrics, flush=True)
