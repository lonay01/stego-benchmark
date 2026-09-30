import importlib
from pathlib import Path
import sys

import pytest

from authornight import environment
from authornight.spec import SOURCES
from authornight.storage import EvidenceError, commit, digest, freeze, read, sha256, write
from rostealsfix import migration
from rostealsfix.checks import targets, validate_module_origin
from rostealsfix.report import bundle, report
from rostealsfix.worker import main


@pytest.fixture
def run_dirs(tmp_path, monkeypatch):
    parent, root, scratch = [tmp_path / x for x in ("parent", "output", "scratch")]
    old = {"code_sha256": migration.PARENT_RELEASE, "sizes": {"rosteals": 256},
           "messages_per_image": 5, "images": [{"image_id": 285}],
           "sources": {k: list(v) for k, v in SOURCES.items()}, "input": "original JPEG",
           "profiles": {"rosteals": ["raw100", "author_bch56"]}}
    old["identity"] = digest(old)
    monkeypatch.setattr(migration, "PARENT_ID", old["identity"])
    write(parent / "plan.json", old)
    settings = {"parent": str(parent), "output": str(root), "scratch": str(scratch), "max_hours": 8}
    return settings, parent, root, scratch


def test_fixed_worker_and_prepare_roots(tmp_path):
    source = tmp_path / "sources/RoSteALS"
    result = environment.prepared_environment("rosteals", tmp_path, tmp_path, tmp_path, source)
    assert "ImageNet-C/imagenet_c" in result["PYTHONPATH"].replace("\\", "/")
    code = Path(environment.__file__).read_text()
    assert 'checkout("imagenetc", root, scratch) / "ImageNet-C" / "imagenet_c"' in code
    assert '("imagenetc", "ImageNet-C/imagenet_c")' in code


def test_original_namespace_bug_and_real_package_fix(tmp_path, monkeypatch):
    repo, revision = SOURCES["imagenetc"]
    outer = tmp_path / "scratch/sources" / (repo.split("/")[-1] + "-" + revision) / "ImageNet-C"
    package = outer / "imagenet_c/imagenet_c"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("# package marker\n")
    root = tmp_path / "output"
    write(root / "_meta/sources/imagenetc.json", {"files": {
        "ImageNet-C/imagenet_c/imagenet_c/__init__.py": sha256(init)}})
    monkeypatch.delitem(sys.modules, "imagenet_c", raising=False)
    monkeypatch.syspath_prepend(str(outer))
    with pytest.raises(EvidenceError, match="Wrong imagenet_c"):
        validate_module_origin(root, tmp_path / "scratch")
    monkeypatch.syspath_prepend(str(outer / "imagenet_c"))
    importlib.invalidate_caches()
    assert validate_module_origin(root, tmp_path / "scratch") == str(init.resolve())
    init.write_text("# modified\n")
    with pytest.raises(EvidenceError, match="changed"):
        validate_module_origin(root, tmp_path / "scratch")


def test_targets_walks_full_config():
    config = {"target": "main.Model", "params": {"nested": [{"target": "loss.Real"}],
                                                  "noise": {"target": "noise.Real"}}}
    assert set(targets(config)) == {"main.Model", "loss.Real", "noise.Real"}


def test_plan_resume_budget_and_identity(run_dirs):
    settings, parent, root, _ = run_dirs
    before = sha256(parent / "plan.json")
    first = migration.make_plan(settings, "new-code")
    second = migration.make_plan({**settings, "max_hours": 3}, "new-code")
    assert first == second
    assert first["profiles"] == {"rosteals": ["raw100", "author_bch56"]}
    assert sha256(parent / "plan.json") == before
    with pytest.raises(EvidenceError):
        migration.make_plan(settings, "different-code")
    assert read(root / "plan.json") == first


def test_parent_tampering_rejected(run_dirs):
    settings, parent, _, _ = run_dirs
    old = read(parent / "plan.json")
    old["images"][0]["image_id"] = 999
    write(parent / "plan.json", old)
    with pytest.raises(EvidenceError, match="checksum"):
        migration.make_plan(settings, "code")


def test_output_must_not_overlap_parent(run_dirs):
    settings, parent, _, _ = run_dirs
    with pytest.raises(EvidenceError, match="overlap"):
        migration.make_plan({**settings, "output": str(parent / "new")}, "code")


def test_copy_inputs_read_only_and_resume(run_dirs):
    settings, parent, root, _ = run_dirs
    image = parent / "_inputs/285.jpg"
    image.parent.mkdir()
    image.write_bytes(b"test original bytes")
    originals = [{"image_id": 285, "path": "_inputs/285.jpg", "sha256": sha256(image)}]
    write(parent / "_inputs/originals.json", originals)
    sep = parent / "sepwater/keep.json"
    write(sep, {"result": "untouched"})
    before = {p.relative_to(parent): sha256(p) for p in parent.rglob("*") if p.is_file()}
    migration.import_parent(settings, "code")
    migration.import_parent(settings, "code")
    assert not (root / "sepwater").exists()
    assert sha256(root / "_inputs/285.jpg") == sha256(image)
    assert before == {p.relative_to(parent): sha256(p) for p in parent.rglob("*") if p.is_file()}
    (root / "_inputs/285.jpg").write_bytes(b"damaged copy")
    with pytest.raises(EvidenceError, match="Changed recovery copy"):
        migration.import_parent(settings, "code")


@pytest.mark.parametrize("method", ["sepwater", "distseal", "all"])
def test_reject_other_methods(method):
    with pytest.raises(ValueError, match="RoSteALS only"):
        main({"method": method, "phase": "prepare"})


def test_empty_report_is_not_success(run_dirs):
    settings, _, root, _ = run_dirs
    migration.make_plan(settings, "code")
    result = report(root)
    assert not result["complete"] and not result["checks_passed"]
    assert result["profiles"]["raw100"]["cases"] == 0
    assert result["attacks"]["planned_tests"] == 350
    assert bundle(root).is_file()


def add_cases(root, plan):
    for rep in range(5):
        for profile in ("raw100", "author_bch56"):
            commit(root, f"rosteals/coco/records/000000000285_m{rep}_{profile}.json", plan["identity"],
                {"image_id": 285, "repetition": rep, "profile": profile,
                 "checks": {"png_pixels_equal": True}, "measurements": {"author_png": {
                    "nbits": 100, "errors": 1, "exact": False, "word_accuracy_threshold_0_2": True}}})


def add_grids(root, plan, count=70):
    grid = [[f"attack{i}", level] for i in range(14) for level in range(1, 6)]
    freeze(root / "_meta/recovery/rosteals_attack_grid.json", grid)
    for rep in range(5):
        source = f"rosteals/coco/records/000000000285_m{rep}_raw100.json"
        commit(root, f"rosteals/corruptions/000000000285_m{rep}_raw100.json", plan["identity"], {
            "source_record": source, "source_record_sha256": read(root / source)["record_sha256"],
            "scores": [{"attack": name, "severity": level, "status": "measured", "errors": 1, "nbits": 100}
                       for name, level in grid[:count]]})


def test_nonzero_ber_is_not_technical_failure(run_dirs):
    settings, _, root, _ = run_dirs
    plan = migration.make_plan(settings, "code")
    add_cases(root, plan)
    add_grids(root, plan)
    write(root / "_meta/fix/import_check.json", {"passed": True})
    write(root / "_meta/recovery/rosteals_preflight.json", {"passed": True})
    assert report(root)["complete"]


def test_partial_grid_and_stale_source_not_complete(run_dirs):
    settings, _, root, _ = run_dirs
    plan = migration.make_plan(settings, "code")
    add_cases(root, plan)
    add_grids(root, plan, count=69)
    result = report(root)
    assert result["native_complete"] and result["attacks"]["complete_grids"] == 0
    assert result["attacks"]["measured_tests"] == 345 and not result["complete"]
    source = "rosteals/coco/records/000000000285_m0_raw100.json"
    commit(root, source, plan["identity"], {**read(root / source), "new_evidence": True})
    assert report(root)["integrity_errors"]


def test_duplicate_or_unexpected_cases_not_counted(run_dirs):
    settings, _, root, _ = run_dirs
    plan = migration.make_plan(settings, "code")
    add_cases(root, plan)
    row = read(root / "rosteals/coco/records/000000000285_m0_raw100.json")
    commit(root, "rosteals/coco/records/duplicate.json", plan["identity"], row)
    result = report(root)
    assert result["integrity_errors"] and not result["checks_passed"]


@pytest.mark.parametrize("failure", ["imports", "preflight", "coco", None])
def test_queue_gates_and_never_runs_other_models(run_dirs, monkeypatch, failure):
    from contextlib import nullcontext
    import os
    from types import SimpleNamespace
    from rostealsfix import runner
    settings, _, root, _ = run_dirs
    phases = []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", environ=os.environ))
    monkeypatch.setattr(runner, "local_lock", lambda root: nullcontext())
    monkeypatch.setattr(runner, "prepared_environment", lambda *args: {})
    monkeypatch.setattr(runner, "report", lambda root: {"native_complete": True, "native_checks_passed": True})

    def supervise(python, job, env, deadline):
        assert job["method"] == "rosteals"
        assert job["worker_module"] == "rostealsfix.worker"
        assert deadline > 0
        phases.append(job["phase"])
        if job["phase"] == "prepare":
            write(root / "_meta/rosteals/prepared.json", {"python": "isolated-python", "source": "source"})
        return {"status": "failed" if job["phase"] == failure else "returned"}

    monkeypatch.setattr(runner, "supervise", supervise)
    runner.run(settings, "code")
    all_phases = ["import", "prepare", "imports", "preflight", "coco", "corruptions"]
    expected = all_phases if failure is None else all_phases[:all_phases.index(failure) + 1]
    assert phases == expected
