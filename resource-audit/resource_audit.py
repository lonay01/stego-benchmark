"""Read-only accounting of recorded benchmark work. Python standard library only.

Stage wall time is not GPU-hours, billing, or proof of a successful experiment.
This module does not import model code or inspect image/weight directories.
"""

# %% Imports and source locations
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import os
import re
import time
import uuid

VERSION = "1.0.0"
MSK = timezone(timedelta(hours=3), "MSK")
KNOWN_ROOTS = (
    "watermarking_production_bank_v1",
    "watermarking_digital_channel_v1",
    "watermarking_neural_channel_v1",
    "watermarking_embedding_bank_v1",
    "watermarking_embedding_diagnostics_v1",
    "watermarking_embedding_diagnostics_recovery_v1",
    "watermarking_full_payload_canary_v1",
    "watermarking_native_resolution_canary_v1",
    "watermarking_native_diagnostics_v1",
    "watermarking_author_reference_v1",
    "watermarking_meta_sepwater_rosteals_v1",
    "watermarking_four_method_quality_v1",
    "watermarking_four_method_night_v2",
    "watermarking_author_reproduction_v1",
    "watermarking_rosteals_sepwater_recovery_v1",
    "watermarking_rosteals_author_fix_v1",
    "watermarking_neural_two_method_check_v1",
    "watermarking_stegastamp_recovery_20261016",
    "watermarking_sepmark_recovery_20261016",
    "wmbench_neural_full_v1",
)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def maximum(*values):
    valid = [v for x in values if (v := number(x)) is not None]
    return max(valid) if valid else None


def iso(value):
    return datetime.fromtimestamp(value, MSK).isoformat() if value is not None else None


def attempt_start(attempt):
    # ResourceRecorder stores time.time_ns(), NOT the model-session filename.
    value = number(attempt)
    if value is not None and 1.5e18 <= value <= 4.1e18:
        return int(attempt) / 1e9
    return None


def period(start_date=None, end_date=None, now=None):
    """Local calendar dates; END_DATE is inclusive. Never extrapolate into the future."""
    now = time.time() if now is None else now
    start = datetime.fromisoformat(start_date).replace(tzinfo=MSK).timestamp() if start_date else 0
    end = (datetime.fromisoformat(end_date).replace(tzinfo=MSK) + timedelta(days=1)).timestamp() if end_date else now
    end = min(end, now)
    if start >= end:
        raise ValueError("START_DATE must be before the end of the reporting window")
    return start, end


def merged(intervals):
    result = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def seconds(intervals):
    return sum(b - a for a, b in merged(intervals))


def clipped(intervals, bounds):
    lo, hi = bounds
    return [[max(a, lo), min(b, hi)] for a, b in intervals if min(b, hi) > max(a, lo)]


def category(stage):
    s = stage.lower()
    if any(x in s for x in ("prepare", "install", "import", "preflight")):
        return "Подготовка окружений и проверки запуска"
    if "generate" in s or "attack" in s or "corruption" in s:
        return "Создание атак / искажений"
    if "decode" in s or "control" in s or "clean" in s:
        return "Декодирование / контроль без помех"
    if "embed" in s or "encode" in s:
        return "Встраивание"
    if any(x in s for x in ("quality", "geometry", "verify", "report")):
        return "Проверки и оценка качества"
    return "Другие / смешанные этапы (см. исходное имя)"


def choose_roots(base, extra_roots=()):
    """Only explicit experiment folders, never a recursive scan of all MyDrive."""
    base = Path(base)
    candidates = [base / name for name in KNOWN_ROOTS] + [Path(p) for p in extra_roots]
    return list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir() and not p.is_symlink()))


# %% Bounded discovery and parsing
def discover(root, issues):
    """Known metadata layouts only. No images, records, packages or model assets."""
    directories = [(root / name, "_run") for name in ("timings", "logs", "_logs")]
    directories += [(root / "_meta" / name, "_run") for name in ("timings", "logs")]
    with os.scandir(root) as entries:
        for item in entries:
            if not item.is_dir(follow_symlinks=False) or item.name.startswith(("_", ".")):
                continue
            # Inspect only these fixed children, never descend through image modes.
            method = Path(item.path)
            for prefix in (method, method / "_meta"):
                directories.extend((prefix / n, item.name) for n in ("timings", "logs", "_logs"))
    seen = set()
    for folder, method in directories:
        if folder.is_symlink() or not folder.is_dir():
            continue
        if not folder.resolve().is_relative_to(root.resolve()):
            issues.append({"source": str(folder), "issue": "Metadata path escapes source root; skipped"})
            continue
        try:
            with os.scandir(folder) as entries:
                for item in entries:
                    if not item.is_file(follow_symlinks=False):
                        continue
                    p = Path(item.path)
                    kind = ("telemetry" if p.name.endswith("_telemetry.jsonl") else
                            "outcome" if p.name.endswith(".outcome.json") else
                            "timing" if folder.name == "timings" and p.suffix == ".json" else None)
                    if kind and p not in seen:
                        seen.add(p)
                        yield p, kind, method
        except OSError as error:
            issues.append({"source": str(folder), "issue": type(error).__name__})


def new_samples():
    return dict(first=None, last=None, samples=0, intervals=[], max_gap_seconds=0,
                process_rss_peak_gib=None, device_vram_peak_gib=None,
                host_ram_total_gib=None, host_ram_available_min_gib=None,
                free_disk_min_gib=None, gpu_names=[], gpu_util_hist={}, gpu_samples=0)


def add_sample(s, row):
    stamp = number(row.get("unix"))
    if stamp is None or not 1.5e9 <= stamp <= 4.1e9:
        raise ValueError("invalid telemetry timestamp")
    previous = s["last"]
    if previous is not None:
        gap = stamp - previous
        if gap < 0:
            raise ValueError("non-monotonic telemetry timestamps")
        s["max_gap_seconds"] = max(s["max_gap_seconds"], gap)
        # Ten-second sampler: do not bridge disconnects or assume it kept running.
        if 0 < gap <= 30:
            if s["intervals"] and s["intervals"][-1][1] == previous:
                s["intervals"][-1][1] = stamp
            else:
                s["intervals"].append([previous, stamp])
    s["first"] = stamp if s["first"] is None else s["first"]
    s["last"] = stamp
    s["samples"] += 1
    s["process_rss_peak_gib"] = maximum(s["process_rss_peak_gib"], row.get("process_tree_rss_gib"))
    memory = row.get("memory") or {}
    s["host_ram_total_gib"] = maximum(s["host_ram_total_gib"], memory.get("host_total_gib"))
    for key, value in (("host_ram_available_min_gib", memory.get("host_available_gib")),
                       ("free_disk_min_gib", row.get("free_disk_gib"))):
        value = number(value)
        if value is not None:
            s[key] = value if s[key] is None else min(value, s[key])
    for device in (row.get("gpu") or {}).get("devices", []):
        name = device.get("name")
        if name and name not in s["gpu_names"]:
            s["gpu_names"].append(name)
        s["device_vram_peak_gib"] = maximum(s["device_vram_peak_gib"], device.get("used_gib"))
        utilization = number(device.get("utilization_percent"))
        if utilization is not None and 0 <= utilization <= 100:
            bucket = str(round(utilization))
            s["gpu_util_hist"][bucket] = s["gpu_util_hist"].get(bucket, 0) + 1
            s["gpu_samples"] += 1


def read_evidence(path, kind, max_bytes, deadline):
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError(f"file exceeds limit ({before.st_size} bytes)")
    digest = hashlib.sha256()
    parsed, invalid, consumed = {}, 0, 0
    with path.open("rb") as stream:
        if kind != "telemetry":
            raw = stream.read(before.st_size)
            digest.update(raw)
            consumed = len(raw)
            parsed = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(parsed, dict):
                raise ValueError("expected JSON object")
        else:
            while consumed < before.st_size:
                if time.monotonic() > deadline:
                    raise TimeoutError("audit time budget")
                raw = stream.readline(min(1024 * 1024 + 1, before.st_size - consumed))
                if not raw:
                    break
                consumed += len(raw)
                digest.update(raw)
                if len(raw) > 1024 * 1024:
                    raise ValueError("oversized JSONL record")
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict) or attempt_start(row.get("attempt")) is None:
                        raise ValueError("unrecognized telemetry schema")
                    key = str(row["attempt"])
                    add_sample(parsed.setdefault(key, new_samples()), row)
                except (ValueError, TypeError, AttributeError, KeyError):
                    invalid += 1
    after = path.stat()
    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    return dict(parsed=parsed, sha256=digest.hexdigest(), bytes=consumed,
                size=before.st_size, mtime_ns=before.st_mtime_ns,
                changed_during_read=changed, invalid_records=invalid)


def stage_name(path):
    name = path.name.replace("_telemetry.jsonl", "").replace(".outcome.json", "").removesuffix(".json")
    return re.sub(r"^\d{16,20}_", "", name)


def parse_timing(data, path, root, method, kind):
    if kind == "outcome":
        # Author-reproduction runners: stage = method_phase_<time.time_ns()>.
        duration = number(data.get("seconds"))
        match = re.search(r"_(\d{16,20})$", str(data.get("stage", "")))
        attempt = match.group(1) if match else None
        method, stage = data.get("method", method), data.get("phase", stage_name(path))
        code = data.get("returncode")
    else:
        duration = number(data.get("wall_seconds"))
        attempt, stage, code = data.get("attempt"), stage_name(path), data.get("exit_code")
    if duration is None or duration < 0:
        raise ValueError("missing or invalid recorded duration")
    start = attempt_start(attempt)
    reason = str(data.get("reason") or "")[:500]
    outcome = ("time_limit" if "timeout" in reason or "time budget" in reason else
               "resource_limit" if any(x in reason for x in ("ram_limit", "disk_limit", "host_oom")) else
               "failed" if reason or (code is not None and code != 0) else
               "returned_zero" if code == 0 else "unknown_exit")
    return dict(root=root.name, method=str(method), stage=str(stage), attempt=str(attempt) if attempt is not None else None,
                start=start, end=start + duration if start is not None else None,
                recorded_seconds=duration, exit_code=code, outcome=outcome, reason=reason,
                timing_source=f"{root.name}/{path.relative_to(root).as_posix()}",
                process_rss_peak_gib=number(data.get("peak_process_tree_rss_gib")),
                device_vram_peak_gib=number(data.get("sampled_device_vram_peak_gib")),
                temporary_slot_peak_gib=number(data.get("peak_slot_gib")), telemetry=None)


# %% Accounting: keep stage time and calendar coverage separate
def summarize(rows, bounds):
    eligible, undated, intervals, observed, daily = [], [], [], [], defaultdict(list)
    groups = defaultdict(lambda: {"attempts": 0, "recorded_stage_hours": 0.0,
                                 "intervals": [], "process_rss_peak_gib": None,
                                 "device_vram_peak_gib": None, "outcomes": Counter()})
    for row in rows:
        t = row.get("telemetry") or {}
        scope = clipped([[row["start"], row["end"]]], bounds) if row["start"] is not None and row["end"] is not None else []
        sampled = clipped(t.get("intervals", []), bounds)
        point_in_range = t.get("last") is not None and bounds[0] <= t["last"] <= bounds[1]
        if row["start"] is None:
            undated.append(row)
        if not scope and not sampled and not point_in_range:
            continue
        row["window_stage_seconds"] = seconds(scope) if row["recorded_seconds"] is not None else None
        row["observed_sample_seconds"] = seconds(sampled)
        row["category"] = category(row["stage"])
        eligible.append(row)
        intervals.extend(scope)
        observed.extend(sampled)
        group = groups[(row["root"], row["method"])]
        group["attempts"] += 1
        group["recorded_stage_hours"] += (row["window_stage_seconds"] or 0) / 3600
        group["intervals"].extend(scope)
        group["outcomes"][row["outcome"]] += 1
        for field in ("process_rss_peak_gib", "device_vram_peak_gib"):
            # Peaks cover the whole overlapping attempt, not only the date-clipped portion.
            group[field] = maximum(group[field], row.get(field), t.get(field))
    calendar = merged(intervals + observed)
    for a, b in calendar:
        while a < b:
            day = datetime.fromtimestamp(a, MSK).replace(hour=0, minute=0, second=0, microsecond=0)
            stop = min(b, (day + timedelta(days=1)).timestamp())
            daily[day.date().isoformat()].append([a, stop])
            a = stop
    per_method = []
    for (root, method), g in sorted(groups.items()):
        per_method.append(dict(root=root, method=method, **{k: v for k, v in g.items() if k != "intervals"},
                               stage_calendar_hours=seconds(g["intervals"]) / 3600))
    names = sorted({n for r in eligible for n in (r.get("telemetry") or {}).get("gpu_names", [])})
    return dict(
        version=VERSION, created_at=iso(time.time()), window_start=iso(bounds[0]) if bounds[0] else None,
        window_end=iso(bounds[1]), attempts=len(eligible), unique_methods=len({r["method"] for r in eligible if r["method"] != "_run"}),
        outcomes=dict(Counter(r["outcome"] for r in eligible)),
        recorded_stage_hours=sum(r["window_stage_seconds"] or 0 for r in eligible) / 3600,
        stage_calendar_hours=seconds(intervals) / 3600,
        observed_calendar_hours=seconds(calendar) / 3600,
        first_evidence=iso(calendar[0][0]) if calendar else None,
        last_evidence=iso(calendar[-1][1]) if calendar else None,
        calendar_span_days=(calendar[-1][1] - calendar[0][0]) / 86400 if calendar else None,
        active_calendar_days=len(daily), undated_attempts=len(undated),
        telemetry_attempts=sum(bool((r.get("telemetry") or {}).get("samples")) for r in eligible),
        gpu_names=names, billable_gpu_hours=None, money_spent=None,
        per_method=per_method,
        daily=[{"date_msk": d, "observed_calendar_hours": seconds(v) / 3600} for d, v in sorted(daily.items())],
        stages=eligible, undated=undated,
    )


def run_audit(roots, output_base, start_date=None, end_date=None, *,
              max_files=20000, max_total_mib=2048, max_minutes=30,
              use_metadata_cache=True, progress=print):
    """Read sources; write ONLY into a separate audit folder. Safe during a live run."""
    roots = list(dict.fromkeys(Path(r).resolve() for r in roots))
    if not roots:
        raise ValueError("No experiment folders found. Check Drive mount and EXTRA_ROOTS.")
    if len({r.name for r in roots}) != len(roots):
        raise ValueError("Source root names must be distinct for unambiguous provenance")
    output_base = Path(output_base).resolve()
    for root in roots:
        if root == output_base or root in output_base.parents or output_base in root.parents:
            raise ValueError("Audit output and source folders must be disjoint")
    if min(max_files, max_total_mib, max_minutes) <= 0:
        raise ValueError("Audit limits must be positive")
    bounds = period(start_date, end_date)
    output_base.mkdir(parents=True, exist_ok=True)
    output = output_base / (datetime.now(MSK).strftime("audit_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    output.mkdir()
    cache_dir = output_base / "_parse_cache_v1"
    cache_dir.mkdir(exist_ok=True)
    deadline = time.monotonic() + max_minutes * 60
    issues, manifest, timings, telemetry = [], [], [], []
    total_bytes, count, cache_hits, budget_hit = 0, 0, 0, False
    roots_seen = []
    last_progress = 0
    for root in roots:
        found = 0
        progress(f"Читаю только метаданные: {root.name}")
        try:
            for path, kind, method in discover(root, issues):
                if count >= max_files or time.monotonic() > deadline:
                    budget_hit = True
                    break
                count += 1
                found += 1
                source = f"{root.name}/{path.relative_to(root).as_posix()}"
                try:
                    stat = path.stat()
                    token = hashlib.sha256(str(path).encode()).hexdigest()
                    cache_file = cache_dir / f"{token}.json"
                    evidence, cached = None, False
                    if use_metadata_cache and cache_file.exists():
                        try:
                            old = json.loads(cache_file.read_text(encoding="utf-8"))
                            if (old.get("version") == VERSION and old.get("size") == stat.st_size
                                    and old.get("mtime_ns") == stat.st_mtime_ns
                                    and not old.get("changed_during_read")):
                                evidence, cached = old, True
                        except (ValueError, OSError):
                            pass
                    if evidence is None:
                        if total_bytes + stat.st_size > max_total_mib * 1024**2:
                            budget_hit = True
                            break
                        evidence = read_evidence(path, kind, (64 if kind == "telemetry" else 2) * 1024**2, deadline)
                        total_bytes += evidence["bytes"]
                        evidence["version"] = VERSION
                        if not evidence["changed_during_read"]:
                            temporary = cache_file.with_suffix("." + uuid.uuid4().hex + ".tmp")
                            temporary.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
                            temporary.replace(cache_file)
                    else:
                        cache_hits += 1
                    manifest.append(dict(source=source, kind=kind, metadata_cache=cached,
                                         **{k: v for k, v in evidence.items() if k != "parsed"}))
                    if evidence["invalid_records"]:
                        issues.append(dict(source=source, issue=f"Invalid/truncated JSONL records: {evidence['invalid_records']}"))
                    if evidence["changed_during_read"]:
                        issues.append(dict(source=source, issue="Source changed during read; prefix snapshot only"))
                    if kind == "telemetry":
                        for attempt, stats in evidence["parsed"].items():
                            telemetry.append(dict(root=root.name, method=method, stage=stage_name(path),
                                                  attempt=attempt, stats=stats, source=source))
                    else:
                        timings.append(parse_timing(evidence["parsed"], path, root, method, kind))
                except (OSError, ValueError, TypeError, TimeoutError) as error:
                    issues.append(dict(source=source, issue=str(error)[:250]))
                if time.monotonic() - last_progress > 10:
                    progress(f"Файлов: {count}; прочитано {total_bytes / 1024**2:.1f} МиБ; кеш: {cache_hits}")
                    last_progress = time.monotonic()
        except OSError as error:
            issues.append(dict(source=str(root), issue=type(error).__name__))
        roots_seen.append(dict(root=str(root), evidence_files=found))
        if budget_hit:
            break
    # Deduplicate a copied attempt even when it appears under another run root.
    unique, conflicts, duplicates = {}, set(), 0
    for row in timings:
        key = row["attempt"] or row["timing_source"]
        if key in unique:
            old = unique[key]
            if any(old[k] != row[k] for k in ("method", "stage", "recorded_seconds", "exit_code", "reason")):
                conflicts.add(key)
                issues.append(dict(source=row["timing_source"], issue="Conflicting attempt ID; excluded from totals"))
            else:
                duplicates += 1
        else:
            unique[key] = row
    for key in conflicts:
        unique.pop(key, None)
    for item in telemetry:
        key, stats = item["attempt"], item["stats"]
        if key in conflicts or not stats["samples"]:
            continue
        if key not in unique:
            unique[key] = dict(root=item["root"], method=item["method"], stage=item["stage"], attempt=key,
                               start=attempt_start(key), end=None, recorded_seconds=None, exit_code=None,
                               outcome="no_terminal_record", reason="", timing_source=None, telemetry=None,
                               process_rss_peak_gib=None, device_vram_peak_gib=None, temporary_slot_peak_gib=None)
        row = unique[key]
        if row["method"] != item["method"] or row["stage"] != item["stage"]:
            issues.append(dict(source=item["source"], issue="Telemetry identity mismatch; not attached"))
            continue
        old = row.get("telemetry")
        # Copies/prefix snapshots: select the more complete one, do not sum duplicates.
        if old is None or stats["samples"] > old["samples"]:
            row["telemetry"] = dict(stats, source=item["source"])
    summary = summarize(list(unique.values()), bounds)
    summary.update(roots_requested=[str(r) for r in roots], roots_scanned=roots_seen,
                   issues=issues, files_read=len(manifest), bytes_read=total_bytes, metadata_cache_hits=cache_hits,
                   duplicate_timing_records=duplicates, conflict_attempts=len(conflicts),
                   scan_limited=budget_hit, scan_has_warnings=bool(issues),
                   source_layout_scope="fixed metadata directories only; no image, result-record or archive scan",
                   output=str(output))
    write_outputs(output, summary, manifest, list(unique.values()))
    progress(f"Готово: {output / 'REPORT.html'}")
    return summary


# %% Human-readable report and evidence tables
def safe_cell(value):
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: safe_cell(row.get(c)) for c in columns})


def table(headers, rows):
    esc = lambda v: html.escape("—" if v is None else str(v))
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def fmt(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def report_html(s):
    esc = html.escape
    methods = table(["Эксперимент / метод", "Попыток этапов", "Сумма, ч", "Без перекрытия, ч", "Пик RAM*, ГиБ", "Пик VRAM*, ГиБ"],
                    [[r["root"] + " / " + r["method"], r["attempts"], fmt(r["recorded_stage_hours"]),
                      fmt(r["stage_calendar_hours"]), fmt(r["process_rss_peak_gib"]), fmt(r["device_vram_peak_gib"])] for r in s["per_method"]])
    categories = defaultdict(float)
    for row in s["stages"]:
        categories[row["category"]] += (row["window_stage_seconds"] or 0) / 3600
    stage_table = table(["Вид этапа (по имени в журнале)", "Суммарная длительность, ч"],
                        [[k, fmt(v)] for k, v in sorted(categories.items(), key=lambda p: -p[1])])
    stops = sorted((r for r in s["stages"] if r["outcome"] != "returned_zero"),
                   key=lambda r: r["start"] or 0, reverse=True)
    stop_table = table(["Начало (MSK)", "Метод / этап", "Исход", "Записанная причина"],
                       [[iso(r["start"]), r["method"] + " / " + r["stage"], r["outcome"],
                         r["reason"] or "Нет итогового сообщения"] for r in stops[:50]])
    daily = s["daily"]
    height = 34 + 25 * len(daily)
    bars = [f"<svg viewBox='0 0 850 {max(height, 60)}' role='img' aria-label='Наблюдавшаяся работа по дням'>"]
    for i, row in enumerate(daily):
        y, hours = 24 + i * 25, row["observed_calendar_hours"]
        bars.append(f"<text x='0' y='{y}'>{esc(row['date_msk'])}</text><rect x='120' y='{y-15}' width='{hours/24*600:.2f}' height='17' fill='#2463a8'/><text x='{130+hours/24*600:.2f}' y='{y}'>{hours:.2f} ч</text>")
    bars.append("</svg>")
    gaps = table(["Источник", "Ограничение / ошибка чтения"], [[x["source"], x["issue"]] for x in s["issues"][:100]])
    scope = table(["Папка", "Файлов метаданных обнаружено"], [[r["root"], r["evidence_files"]] for r in s["roots_scanned"]])
    warning = "Сканирование ограничено лимитом; отчёт неполный." if s["scan_limited"] else "Обход выбранных расположений метаданных завершён. Это не доказательство полноты всей истории."
    gpu = ", ".join(s["gpu_names"]) or "Нет данных о модели GPU в прочитанной телеметрии"
    return f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Вычислительный журнал исследования</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#203147;max-width:1120px;margin:36px auto;padding:0 22px}}h1{{font-size:30px}}h2{{margin-top:32px;font-size:22px}}.muted{{color:#586b80}}.note{{padding:14px 18px;background:#edf3fa;border-left:4px solid #2463a8}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#f1f5f9;padding:16px;flex:1;min-width:180px}}.card strong{{font-size:28px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;border-bottom:1px solid #d7e0e9;padding:9px;vertical-align:top;overflow-wrap:anywhere}}th{{background:#eff3f8}}.scroll{{overflow:auto}}svg{{width:100%;font:13px system-ui}}@media print{{body{{margin:0;max-width:none}}.scroll{{overflow:visible}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style>
<h1>Вычислительный журнал исследования водяных знаков</h1>
<p class="muted">Сформирован {esc(s['created_at'])}. Временная зона: MSK (UTC+3). Источник: сохранённые журналы запусков.</p>
<div class="note">Испытания готовых моделей: подготовка окружений, встраивание, атаки и декодирование. Эти данные не означают непрерывное обучение нейросетей.</div>
<h2>Краткая сводка</h2><p>Наблюдаемая история: {esc(s['first_evidence'] or 'не определена')} — {esc(s['last_evidence'] or 'не определена')}; размах {fmt(s['calendar_span_days'])} суток, дней с зарегистрированной работой: {s['active_calendar_days']}.</p>
<div class="cards"><div class="card"><strong>{s['attempts']}</strong>попыток отдельных этапов</div><div class="card"><strong>{s['recorded_stage_hours']:.2f} ч</strong>сумма длительностей этапов</div><div class="card"><strong>{s['observed_calendar_hours']:.2f} ч</strong>покрытие календарного времени без перекрытий</div></div>
<p>В журналах выбранного периода встречается {s['unique_methods']} названий методов (включая исторические версии). Это не число методов, успешно прошедших полный бенчмарк.</p>
{table(['Исход попытки этапа', 'Количество'], sorted(s['outcomes'].items()))}
<p><code>returned_zero</code> — процесс вернул код 0, а не доказанная полнота результатов. <code>no_terminal_record</code> — есть телеметрия, но нет итогового тайминга; текущую работу по старому журналу не утверждаем.</p>
<h2>Что заняло время</h2>{stage_table}
<h2>Остановки и попытки без итоговой записи</h2>{stop_table}
<p class="muted">Последние 50 записей, полный список — attempts.csv. Отсутствие итоговой записи не доказывает ошибку: процесс мог ещё работать при сборе снимка. Причины взяты из журналов, а не предполагаются по длительности.</p>
<h2>Календарь зарегистрированной работы</h2>{''.join(bars)}
<p class="muted">Перекрывающиеся интервалы считаются один раз. Это календарное покрытие, не сумма часов нескольких машин. Для попыток без итогового тайминга учитываются только соседние отсчёты телеметрии с разрывом ≤30 с; пробелы не достраиваются.</p>
<h2>По экспериментам и методам</h2>{methods}
<p class="muted">* RAM — пик памяти дерева процессов, VRAM — выборочный пик всего GPU. Это пики за целую пересекающую период попытку, даже если задан фильтр дат. Параллельные процессы могут наблюдать один GPU; его память не суммируется по методам. Прочерк означает отсутствие данных.</p>
<h2>Ресурсы и стоимость</h2><p>GPU в телеметрии: {esc(gpu)}. Попыток с телеметрией: {s['telemetry_attempts']} / {s['attempts']}.</p>
<p>Фактическое списание Colab compute units и деньги <strong>не определены</strong>: нужны данные биллинга. Сумма длительностей включает установки, ожидания и I/O. Она не равна занятости GPU, оплачиваемым GPU-часам, FLOPs или времени обучения. Расход RAM/VRAM и выборочная загрузка по попыткам приведены в attempts.csv.</p>
<h2>Проверяемость и границы отчёта</h2><p class="note">{esc(warning)}</p>
<p>Прочитано {s['files_read']} файлов; новых данных {s['bytes_read']/1024**2:.1f} МиБ; повторно использовано {s['metadata_cache_hits']} разборов. Удалено из подсчёта дублирующихся таймингов: {s['duplicate_timing_records']}; конфликтующих ID: {s['conflict_attempts']}. Попыток без даты (не включены в период): {s['undated_attempts']}.</p>
{scope}<p>Не обходились изображения, архивы, веса и миллионы записей измерений. Исчезнувшие, перезаписанные, несохранённые логи, ручная разработка, недокументированные проверки и интервалы вне мониторинга не восстановлены. Файлы ошибок и незнакомые форматы перечислены в issues.json. Текущий прогресс и научные метрики этим аудитом не проверяются.</p>
{gaps}<p>evidence_manifest.json содержит SHA-256 прочитанных снимков. Для кеша повторно проверяются только размер и mtime: прежний хеш не означает новый полный контроль содержимого. Для независимой перепроверки выключите USE_METADATA_CACHE. Исходные результаты не изменялись.</p>
</html>"""


def write_outputs(output, summary, manifest, all_rows):
    for name, data in (("summary.json", {k: v for k, v in summary.items() if k not in {"stages", "undated"}}),
                       ("evidence_manifest.json", manifest), ("issues.json", summary["issues"]),
                       ("attempts.json", all_rows)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    flat = []
    for row in all_rows:
        t = row.get("telemetry") or {}
        count = t.get("gpu_samples", 0)
        hist = t.get("gpu_util_hist", {})
        flat.append(dict(row, start_msk=iso(row["start"]), end_msk=iso(row["end"]),
                         gpu_names=t.get("gpu_names"), telemetry_samples=t.get("samples"),
                         sampled_gpu_util_mean_percent=sum(int(k)*v for k, v in hist.items())/count if count else None,
                         sampled_last_msk=iso(t.get("last")), sampled_max_gap_seconds=t.get("max_gap_seconds"),
                         host_ram_total_gib=t.get("host_ram_total_gib"), free_disk_min_gib=t.get("free_disk_min_gib"),
                         process_rss_peak_gib=maximum(row.get("process_rss_peak_gib"), t.get("process_rss_peak_gib")),
                         device_vram_peak_gib=maximum(row.get("device_vram_peak_gib"), t.get("device_vram_peak_gib"))))
    columns = ["root", "method", "stage", "attempt", "start_msk", "end_msk", "recorded_seconds",
               "window_stage_seconds", "observed_sample_seconds", "outcome", "exit_code", "reason", "gpu_names",
               "process_rss_peak_gib", "device_vram_peak_gib", "temporary_slot_peak_gib", "host_ram_total_gib",
               "free_disk_min_gib", "telemetry_samples", "sampled_gpu_util_mean_percent", "sampled_last_msk",
               "sampled_max_gap_seconds", "timing_source"]
    write_csv(output / "attempts.csv", flat, columns)
    write_csv(output / "daily.csv", summary["daily"], ["date_msk", "observed_calendar_hours"])
    write_csv(output / "methods.csv", summary["per_method"], ["root", "method", "attempts", "recorded_stage_hours",
                                                            "stage_calendar_hours", "process_rss_peak_gib", "device_vram_peak_gib", "outcomes"])
    (output / "REPORT.html").write_text(report_html(summary), encoding="utf-8")
