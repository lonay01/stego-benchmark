"""Notebook-only regression tests; no apt, network or model downloads."""

import ast
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("production_builder", ROOT / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SystemDependencyHotfixTests(unittest.TestCase):
    def execute(self, models="all", available=(), colab=True, repaired=True,
                load_error=False, missing_symbol=False):
        present = set(available)
        calls = []

        def find(name):
            if name.startswith("MagickWand") and "wand" in present:
                return "libMagickWand-6.Q16.so.6"
            if name == "GL" and "gl" in present:
                return "libGL.so.1"
            return None

        def load(path):
            if load_error:
                raise OSError("synthetic shared-library dependency failure")
            return SimpleNamespace() if missing_symbol else SimpleNamespace(MagickWandGenesis=object())

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if "install" in command and repaired:
                present.update(("wand", "gl"))

        namespace = dict(MODELS=models, IN_COLAB=colab, os=os, shutil=shutil, subprocess=subprocess)
        with patch.object(ctypes.util, "find_library", side_effect=find), \
             patch.object(ctypes, "CDLL", side_effect=load), \
             patch.object(subprocess, "run", side_effect=run), \
             patch.object(shutil, "which", return_value="/usr/bin/apt-get"), \
             patch.object(os, "geteuid", return_value=0, create=True):
            exec(compile(builder.AUTHOR_SYSTEM_DEPENDENCIES, "author_system_dependencies", "exec"), namespace)
        return namespace, calls

    def test_existing_libraries_do_not_install_anything(self):
        result, calls = self.execute(available=("wand", "gl"))
        self.assertEqual(calls, [])
        self.assertEqual(set(result["AUTHOR_SYSTEM_LIBRARIES"]), {"libmagickwand-dev", "libgl1"})

    def test_missing_wand_installs_only_missing_system_package(self):
        for method in ("rosteals", "sepwater"):
            with self.subTest(method=method):
                result, calls = self.execute(models=[method], available=("gl",))
                self.assertEqual(calls[0][0], ["apt-get", "-qq", "update"])
                self.assertEqual(calls[1][0], ["apt-get", "-qq", "install", "-y",
                                             "--no-install-recommends", "libmagickwand-dev"])
                self.assertTrue(all(kwargs["check"] for _, kwargs in calls))
                self.assertIn("libmagickwand-dev", result["AUTHOR_SYSTEM_LIBRARIES"])

    def test_fresh_colab_installs_both_then_checks_again(self):
        result, calls = self.execute()
        self.assertEqual(calls[1][0][-2:], ["libmagickwand-dev", "libgl1"])
        self.assertEqual(len(result["AUTHOR_SYSTEM_LIBRARIES"]), 2)

    def test_other_methods_do_not_probe_or_install(self):
        for models in (["wam", "distseal_posthoc_dcae"], ["meta_hidden_revisited"], []):
            result, calls = self.execute(models=models)
            self.assertEqual(result["AUTHOR_SYSTEM_LIBRARIES"], {})
            self.assertEqual(calls, [])

    def test_server_missing_library_stops_with_install_instructions(self):
        with self.assertRaisesRegex(RuntimeError, "sudo apt-get update"):
            self.execute(colab=False)

    def test_failed_installation_does_not_start_models(self):
        with self.assertRaisesRegex(RuntimeError, "preflight failed after installation"):
            self.execute(repaired=False)

    def test_found_but_unloadable_or_wrong_library_is_not_success(self):
        for kwargs in ({"load_error": True}, {"missing_symbol": True}):
            with self.assertRaisesRegex(RuntimeError, "preflight failed after installation"):
                self.execute(available=("wand", "gl"), **kwargs)

    def test_generated_notebook_keeps_original_source_archive(self):
        data = (ROOT / "dist" / f"{builder.NAME}_source.zip").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), builder.RESUME_SOURCE_SHA)
        original = (ROOT / "dist" / f"{builder.NAME}.ipynb").read_bytes()
        fixed = (ROOT / "dist" / f"{builder.HOTFIX_NAME}.ipynb").read_bytes()
        self.assertEqual(original, fixed)
        notebook = json.loads(fixed)
        setup = "".join(notebook["cells"][4]["source"])
        self.assertIn(builder.AUTHOR_SYSTEM_DEPENDENCIES, setup)
        self.assertLess(setup.index("ensure_author_system_dependencies(MODELS, IN_COLAB)"),
                        setup.index('PROJECT = SCRATCH_ROOT / "release"'))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])

    @unittest.skipUnless(os.environ.get("WMBENCH_BASELINE_NOTEBOOK"), "Executed baseline notebook not supplied")
    def test_against_user_run_only_setup_source_changed(self):
        baseline = json.loads(Path(os.environ["WMBENCH_BASELINE_NOTEBOOK"]).read_text("utf-8"))
        fixed = json.loads((ROOT / "dist" / f"{builder.HOTFIX_NAME}.ipynb").read_text("utf-8"))
        self.assertEqual(len(baseline["cells"]), len(fixed["cells"]))
        changed = [i for i, (left, right) in enumerate(zip(baseline["cells"], fixed["cells"]))
                   if "".join(left["source"]) != "".join(right["source"])]
        self.assertEqual(changed, [3, 4])
        before = "".join(baseline["cells"][4]["source"])
        after = "".join(fixed["cells"][4]["source"])
        self.assertEqual(after.replace(builder.AUTHOR_SYSTEM_DEPENDENCIES + "\n", "", 1), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
