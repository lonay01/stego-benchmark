"""Diagnostic-only import/dependency repairs; no change to model inference math."""
import copy
import sys
import types
from contextlib import contextmanager

from wmbench.adapters.legacy import EditGuardAdapter


@contextmanager
def constrained_builds(lock):
    """Preserve the resolved Torch/Numpy/etc. versions when adding BasicSR.

    Only active inside the separate diagnostic preparation subprocess. The
    production installer file and other models' installation commands are untouched.
    """
    from wmbench import environment
    original = environment.command

    def command(args, cwd=None, env=None):
        args = list(args)
        if '--no-build-isolation' in args:
            if not lock.is_file():
                raise FileNotFoundError(f'Cannot constrain build without {lock}')
            args += ['--constraint', str(lock)]
        return original(args, cwd=cwd, env=env)

    environment.command = command
    try:
        yield
    finally:
        environment.command = original


def repair_model(model):
    result = copy.deepcopy(model)
    if model['id'] not in ('editguard', 'sepwater'):
        return result
    result.pop('validated_lock', None)
    requirements = [r for r in result['requirements'] if not r.lower().startswith('setuptools')]
    # Lightning 1.8 imports pkg_resources; setuptools 84 no longer supplies it.
    requirements.extend(['setuptools==70.3.0', 'wheel==0.45.1'])
    if model['id'] == 'editguard':
        result['adapter'] = 'bank_diagnostics.dependency_fixes:EditGuardImportFix'
        result['build_requirements'] = ['basicsr==1.4.2']
    result['requirements'] = requirements
    return result


def install_basicsr_torchvision_compat():
    """BasicSR1.4.2 imports one function from a removed torchvision module.

    Re-export the same torchvision tensor implementation under its legacy name.
    This is not a replacement BasicSR network, grayscale algorithm or checkpoint.
    """
    import torchvision.transforms._functional_tensor as tensor_functional
    name = 'torchvision.transforms.functional_tensor'
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.rgb_to_grayscale = tensor_functional.rgb_to_grayscale
        sys.modules[name] = module


class EditGuardImportFix(EditGuardAdapter):
    def __init__(self, options):
        install_basicsr_torchvision_compat()
        super().__init__(options)
