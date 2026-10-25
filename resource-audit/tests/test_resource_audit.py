import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "resource_audit.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
T = 1790800000  # 2026-09-30; historical, not a moving test clock.


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def timing(root, stage="decode", start=T, duration=100, **extra):
    return put(root / "method" / "timings" / f"{int(start*1e9)}_{stage}.json",
               dict(attempt=int(start*1e9), wall_seconds=duration, exit_code=0, reason="", **extra))


def sample(start=T, stamp=T, utilization=50):
    return dict(attempt=int(start*1e9), unix=stamp, process_tree_rss_gib=3,
                free_disk_gib=120, memory={"host_total_gib": 167, "host_available_gib": 150},
                gpu={"devices": [{"name": "A100", "used_gib": 12, "utilization_percent": utilization}]})


def audit(root, output, **kwargs):
    return a.run_audit([root], output, progress=lambda *_: None, **kwargs)


def test_overlap_not_gpu_hours(tmp_path):
    root = tmp_path / "run"
    timing(root, "prepare", T, 100)
    timing(root, "quality", T + 50, 100)
    s = audit(root, tmp_path / "out")
    assert s["recorded_stage_hours"] == pytest.approx(200/3600)
    assert s["observed_calendar_hours"] == pytest.approx(150/3600)
    assert s["billable_gpu_hours"] is None and s["money_spent"] is None


def test_start_not_session_filename_or_mtime(tmp_path):
    root = tmp_path / "run"
    p = timing(root, start=T)
    p.rename(p.with_name("1500000000000000000_decode.json"))
    s = audit(root, tmp_path / "out")
    assert s["first_evidence"] == a.iso(T)


def test_telemetry_overlap_and_partial_tail(tmp_path):
    root = tmp_path / "run"
    timing(root)
    path = root / "method/logs" / f"{int(T*1e9)}_decode_telemetry.jsonl"
    put(path, "\n".join(json.dumps(sample(stamp=T+i)) for i in (0, 10, 20)) + '\n{"broken":')
    s = audit(root, tmp_path / "out")
    assert s["attempts"] == 1
    assert s["recorded_stage_hours"] == pytest.approx(100/3600)
    assert s["observed_calendar_hours"] == pytest.approx(100/3600)
    assert s["telemetry_attempts"] == 1 and s["issues"]
    assert s["per_method"][0]["device_vram_peak_gib"] == 12


def test_missing_final_record_no_extrapolation(tmp_path):
    root = tmp_path / "run"
    p = root / "method/logs" / f"{int(T*1e9)}_decode_telemetry.jsonl"
    put(p, "\n".join(json.dumps(sample(stamp=T+i)) for i in (0, 10, 500, 510)) + "\n")
    s = audit(root, tmp_path / "out")
    assert s["recorded_stage_hours"] == 0
    assert s["observed_calendar_hours"] == pytest.approx(20/3600)
    assert s["outcomes"] == {"no_terminal_record": 1}


def test_unknown_time_not_invented_from_mtime(tmp_path):
    root = tmp_path / "run"
    put(root / "method/timings/old_prepare.json", {"wall_seconds": 400, "exit_code": 0})
    s = audit(root, tmp_path / "out")
    assert s["undated_attempts"] == 1
    assert s["recorded_stage_hours"] == 0
    assert s["first_evidence"] is None


def test_historical_author_outcome(tmp_path):
    root = tmp_path / "old"
    put(root / "_logs" / "author.outcome.json", dict(stage=f"rosteals_coco_{int(T*1e9)}",
        method="rosteals", phase="coco", seconds=180, returncode=0))
    s = audit(root, tmp_path / "out")
    assert s["recorded_stage_hours"] == pytest.approx(.05)
    assert s["per_method"][0]["method"] == "rosteals"


def test_source_unchanged_and_only_metadata_read(tmp_path):
    root = tmp_path / "run"
    timing(root)
    put(root / "method/records/secret.json", {"wall_seconds": 99999})
    put(root / "method/native/ignored_telemetry.jsonl", "not telemetry")
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    s = audit(root, tmp_path / "out")
    assert s["files_read"] == 1
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("offset,expected", [("nested", True), ("same", True), ("parent", True)])
def test_source_output_separation(tmp_path, offset, expected):
    root = tmp_path / "run"
    timing(root)
    output = {"nested": root / "audit", "same": root, "parent": tmp_path}[offset]
    with pytest.raises(ValueError, match="disjoint"):
        audit(root, output)


def test_duplicates_not_counted_twice_and_conflicts_excluded(tmp_path):
    one, two, output = tmp_path / "one", tmp_path / "two", tmp_path / "out"
    timing(one)
    timing(two)
    s = a.run_audit([one, two], output, progress=lambda *_: None)
    assert s["attempts"] == 1 and s["duplicate_timing_records"] == 1
    timing(two, duration=500)
    s = a.run_audit([one, two], output, progress=lambda *_: None)
    assert s["attempts"] == 0 and s["conflict_attempts"] == 1


def test_cache_and_invalidation(tmp_path):
    root, output = tmp_path / "run", tmp_path / "out"
    timing(root)
    audit(root, output)
    assert audit(root, output)["metadata_cache_hits"] == 1
    timing(root, duration=25000)
    assert audit(root, output)["metadata_cache_hits"] == 0
    assert audit(root, output, use_metadata_cache=False)["metadata_cache_hits"] == 0


def test_resource_limit_not_success(tmp_path):
    root = tmp_path / "run"
    p = timing(root)
    d = json.loads(p.read_text())
    d.update(exit_code=-15, reason="timeout: stage time budget exhausted")
    put(p, d)
    s = audit(root, tmp_path / "out")
    assert s["outcomes"] == {"time_limit": 1}


def test_date_clip_and_moscow_midnight(tmp_path):
    root = tmp_path / "run"
    midnight = a.datetime(2026, 9, 30, tzinfo=a.MSK).timestamp()
    timing(root, start=midnight-30, duration=60)
    s = audit(root, tmp_path / "out", start_date="2026-09-30", end_date="2026-09-30")
    assert s["recorded_stage_hours"] == pytest.approx(30/3600)
    assert s["daily"] == [{"date_msk": "2026-09-30", "observed_calendar_hours": 30/3600}]
    s = audit(root, tmp_path / "out")
    assert len(s["daily"]) == 2


def test_budget_is_explicit(tmp_path):
    root = tmp_path / "run"
    timing(root)
    timing(root, start=T+1000)
    s = audit(root, tmp_path / "out", max_files=1)
    assert s["scan_limited"] and s["attempts"] == 1


def test_html_csv_escape(tmp_path):
    root = tmp_path / "run"
    timing(root)
    s = audit(root, tmp_path / "out")
    s["per_method"][0]["method"] = "<script>alert(1)</script>"
    page = a.report_html(s)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert a.safe_cell("=1+1") == "'=1+1"
    assert "Base64" not in page


def test_empty_and_bad_files_are_not_successful_work(tmp_path):
    root = tmp_path / "run"
    put(root / "method/timings/bad.json", "bad")
    put(root / "method/timings/unknown.json", {"unrelated": 500})
    s = audit(root, tmp_path / "out")
    assert s["attempts"] == 0 and len(s["issues"]) == 2


def test_build_static_validation_and_execute_embedded_code(tmp_path):
    import nbformat
    build_spec = importlib.util.spec_from_file_location("builder", Path(__file__).resolve().parents[1] / "build_notebook.py")
    b = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(b)
    notebook = nbformat.read(b.build(), as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{i}", "exec")
    ns = {"__name__": "audit_notebook_smoke"}
    # The embedded functions, not an imported local module, perform this smoke run.
    inside = False
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## 3."):
            inside = True
        if cell.cell_type == "markdown" and cell.source.startswith("## 4."):
            inside = False
        if inside and cell.cell_type == "code":
            exec(compile(cell.source, "embedded-cell", "exec"), ns)
    root = tmp_path / "run"
    timing(root)
    result = ns["run_audit"]([root], tmp_path / "out", progress=lambda *_: None)
    assert result["attempts"] == 1
    assert (Path(result["output"]) / "REPORT.html").exists()


def test_missing_gpu_telemetry_is_not_zero_utilization(tmp_path):
    root = tmp_path / "run"
    timing(root)
    s = audit(root, tmp_path / "out")
    assert s["gpu_names"] == []
    assert s["per_method"][0]["device_vram_peak_gib"] is None
    assert s["telemetry_attempts"] == 0


def test_long_gap_and_duplicate_samples():
    stats = a.new_samples()
    for stamp in (T, T+10, T+10, T+2000, T+2010):
        a.add_sample(stats, sample(stamp=stamp))
    assert a.seconds(stats["intervals"]) == 20
    assert stats["max_gap_seconds"] == 1990


def test_nonfinite_duration_rejected(tmp_path):
    root = tmp_path / "run"
    timing(root, duration=float("nan"))
    s = audit(root, tmp_path / "out")
    assert s["attempts"] == 0 and s["issues"]


def test_oversize_file_does_not_read(tmp_path, monkeypatch):
    root = tmp_path / "run"
    p = put(root / "method/timings/huge.json", "x" * (2 * 1024**2 + 1))
    called = []
    original = Path.open
    def check(self, *args, **kwargs):
        if self == p and args and args[0] == "rb":
            called.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", check)
    s = audit(root, tmp_path / "out")
    assert not called and s["issues"]


def test_changing_file_snapshot_is_not_cached(tmp_path, monkeypatch):
    root, output = tmp_path / "run", tmp_path / "out"
    timing(root)
    original = a.read_evidence
    def changing(*args):
        evidence = original(*args)
        evidence["changed_during_read"] = True
        return evidence
    monkeypatch.setattr(a, "read_evidence", changing)
    s = audit(root, output)
    assert s["issues"] and s["files_read"] == 1
    assert not list((output / "_parse_cache_v1").glob("*.json"))


def test_embedding_bank_meta_layout(tmp_path):
    root = tmp_path / "bank"
    put(root / "rosteals/_meta/timings/legacy_embed.json",
        dict(attempt=int(T*1e9), wall_seconds=600, exit_code=0))
    s = audit(root, tmp_path / "out")
    assert s["attempts"] == 1 and s["per_method"][0]["method"] == "rosteals"


def test_entire_notebook_local_execution(tmp_path):
    source = Path(__file__).resolve().parents[1] / "dist/BENCHMARK_Resource_Audit.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    base = tmp_path / "saved"
    timing(base / a.KNOWN_ROOTS[0])
    ns = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = cell["source"]
        if code.startswith("BASE = "):
            code = code.replace('Path("/content/drive/MyDrive")', repr(base))
            # repr(Path) needs the class name only on Windows.
            ns[base.__class__.__name__] = base.__class__
        exec(compile(code, "notebook-smoke", "exec"), ns)
    assert ns["SUMMARY"]["attempts"] == 1
    assert ns["ARCHIVE"].is_file()
