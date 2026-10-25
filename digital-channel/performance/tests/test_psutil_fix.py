"""Regression checks for absent psutil; no watermark models or GPU required."""

import ast
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


WORKER = Path(__file__).resolve().parents[1] / "src/wmbench_fast/worker.py"
if not WORKER.exists():  # Tests copied into a packaged release.
    WORKER = Path(os.environ["WM_DIGITAL_TEST_RELEASE"]) / "src/wmbench_fast/worker.py"


def load_resources():
    tree = ast.parse(WORKER.read_text("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resources")
    scope = dict(os=os, subprocess=subprocess)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER), "exec"), scope)
    return scope["resources"]


@contextlib.contextmanager
def without_psutil(name="psutil"):
    original = builtins.__import__

    def importing(module, *args, **kwargs):
        if module == "psutil":
            raise ModuleNotFoundError("No module named " + name, name=name)
        return original(module, *args, **kwargs)

    with patch("builtins.__import__", side_effect=importing):
        yield


class PsutilRepair(unittest.TestCase):
    def test_existing_psutil_path_unchanged(self):
        child = types.SimpleNamespace(is_running=lambda: True,
                                      memory_info=lambda: types.SimpleNamespace(rss=2 * 1024**3))
        me = types.SimpleNamespace(is_running=lambda: True,
                                   memory_info=lambda: types.SimpleNamespace(rss=1024**3),
                                   children=lambda recursive: [child])
        fake = types.SimpleNamespace(Process=lambda: me)
        with patch.dict(sys.modules, psutil=fake), patch.object(subprocess, "check_output", return_value="") as call:
            result = load_resources()()
        self.assertEqual(result, {"rss_gib": 3, "gpu_gib": 0})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0][0], "nvidia-smi")

    def test_missing_psutil_uses_controller_and_worker_pid(self):
        with without_psutil(), patch.dict(os.environ, WM_BENCH_TELEMETRY_PYTHON="controller-python"), \
                patch.object(subprocess, "check_output", side_effect=["2.5\n", f"{os.getpid()}, 2048\n"]) as call:
            result = load_resources()()
        self.assertEqual(result, {"rss_gib": 2.5, "gpu_gib": 2})
        command = call.call_args_list[0].args[0]
        self.assertEqual(command[:3], ["controller-python", "-I", "-c"])
        self.assertEqual(command[-1], str(os.getpid()))
        self.assertEqual(call.call_args_list[0].kwargs["timeout"], 15)

    def test_absent_controller_fails_closed(self):
        with without_psutil(), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Missing controller"):
                load_resources()()

    def test_broken_psutil_internal_import_not_hidden(self):
        with without_psutil("unrelated_internal_dependency"):
            with self.assertRaises(ModuleNotFoundError):
                load_resources()()

    def test_bad_telemetry_fails_closed(self):
        for value in ("nan", "inf", "-1", "0", "not a number"):
            with self.subTest(value=value), without_psutil(), \
                    patch.dict(os.environ, WM_BENCH_TELEMETRY_PYTHON="controller-python"), \
                    patch.object(subprocess, "check_output", return_value=value):
                with self.assertRaises((ValueError, RuntimeError)):
                    load_resources()()

    def test_controller_failure_not_ignored(self):
        with without_psutil(), patch.dict(os.environ, WM_BENCH_TELEMETRY_PYTHON="controller-python"), \
                patch.object(subprocess, "check_output", side_effect=subprocess.TimeoutExpired("monitor", 15)):
            with self.assertRaises(subprocess.TimeoutExpired):
                load_resources()()

    def test_actual_subprocess_without_site_packages(self):
        # -S reproduces a worker with no psutil. Its controller retains the
        # installed dependency and samples the worker PID, not its own PID.
        script = '''
import ast, json, os, pathlib, subprocess, sys
try:
    import psutil
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Smoke worker unexpectedly has psutil")
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text("utf-8"))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resources")
exec(compile(ast.Module(body=[node], type_ignores=[]), sys.argv[1], "exec"))
print(json.dumps(resources()))
'''
        env = dict(os.environ, WM_BENCH_TELEMETRY_PYTHON=sys.executable)
        result = subprocess.run([sys.executable, "-S", "-c", script, str(WORKER)], env=env,
                                capture_output=True, text=True, check=True, timeout=45)
        self.assertIn("worker RAM sampled by controller", result.stdout)
        sample = json.loads(result.stdout.splitlines()[-1])
        self.assertGreater(sample["rss_gib"], 0)


if __name__ == "__main__":
    unittest.main()
