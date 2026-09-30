"""Audit SepWater's actual released control weights and 224/256 geometry on CPU."""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from authornight.storage import sha256, write
from authornight.latent import export_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HF_HOME'] = str(args.output.parent/'hf')
    torch.set_num_threads(2)
    source = args.source/'control_Latent_Diffusion_Model/autoencoder3.py'
    raw = source.read_bytes().replace(b'\r\n', b'\n')
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob == '797b7b24b4813e16db353b029409978da5f8fd5d'
    sys.path.insert(0, str(args.source))
    from Latent_Diffusion_Model.modules.diffusionmodules.util import conv_nd, zero_module
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('View', 'SecretEncoder')]
    namespace = dict(nn=torch.nn, torch=torch, np=np, conv_nd=conv_nd, zero_module=zero_module)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    params = yaml.safe_load((args.source/'pre_trained_models/inference.yaml').read_text())['model']['params']['control_config']['params']
    control = namespace['SecretEncoder'](**params).eval()
    checkpoint = torch.load(args.weight, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    control_state = {k[len('control.'):]: v for k, v in state.items() if k.startswith('control.')}
    control.load_state_dict(control_state, strict=True)
    with torch.no_grad():
        residual, _ = control(torch.zeros(1, 3, 56, 56), torch.zeros(1, 100))
        assert tuple(residual.shape) == (1, 3, 56, 56)
        try:
            _ = torch.zeros(1, 3, 64, 64)+residual
        except RuntimeError as error:
            mismatch = str(error)
        else:
            raise AssertionError('256 should fail with this unmodified author config')
    from PIL import Image
    original = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8))
    cover, marked = torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224)+.02
    sep = export_function(args.source/'embed.py')(marked, cover, original)
    ro = export_function(Path(__file__).parent/'reference/ro/inference.py')(marked, cover, original)
    assert np.array_equal(sep, ro)
    result = {'scope': 'CPU source + pretrained secret-control weights; NOT full model GPU inference',
              'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'source_blob': blob,
              'weight_sha256': sha256(args.weight), 'strict_control_tensors': len(control_state),
              'control_shape': list(residual.shape), '256_mismatch': mismatch,
              'author_export_same_as_rosteals': True}
    write(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
