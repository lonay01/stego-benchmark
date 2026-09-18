"""Small resource report for the targeted two-model canary (no GPU dependency)."""

from pathlib import Path
import zipfile

from .io import atomic_bytes, read_json


def write_diagnostic_report(root: Path) -> Path:
    report = read_json(root / "canary_report.json")
    if {r["model_id"] for r in report["rows"]} - {"zodiac", "chunkyseal"}:
        raise ValueError("This diagnostic report is restricted to ZoDiac and ChunkySeal")
    lines = [
        "# Диагностика ZoDiac / ChunkySeal",
        "",
        "Техническая проверка на одном COCO PNG: Native, Matched42 и чистый контроль.",
        "AI-атак здесь нет. Passed не означает устойчивость; FPR на одном контроле не оценивается.",
        "",
    ]
    for row in report["rows"]:
        model_id = row["model_id"]
        out = root / "models" / model_id
        lines += [f"## {model_id}: {row['status']}", "", str(row.get("reason", "")), ""]
        for action in ("prepare", "run"):
            path = out / f"{action}_resources.json"
            if not path.is_file():
                continue
            stats = read_json(path)
            lines.append(
                f"- {action}: {stats['wall_seconds'] / 60:.1f} мин; "
                f"пиковая сумма RSS процессов ≈ {stats['peak_process_tree_rss_gib']:.1f} GiB; "
                f"локальные файлы ≈ {stats['peak_slot_gib']:.1f} GiB."
            )
            if "sampled_device_vram_peak_gib" in stats:
                lines.append(
                    f"- VRAM всей GPU, максимум выборки: {stats['sampled_device_vram_peak_gib']:.1f} GiB; "
                    f"рост cgroup oom_kill: {stats['cgroup_oom_kill_increased']}."
                )
        lines.append("")
    lines += [
        "## Как читать результат",
        "",
        "- `passed` / `completed_with_warnings`: код завершён. BER и matched-quality предупреждения смотрите в CANARY_REPORT.md и roundtrip_result.json.",
        "- `blocked`: ресурсы не прошли порог допуска, модель не скачивалась/не запускалась.",
        "- `timeout`: исчерпан заданный бюджет. Это не доказательство зависания или ошибки модели.",
        "- `SIGKILL`: внешний сигнал; без роста oom_kill нехватка RAM лишь гипотеза.",
        "- Рост oom_kill — событие OOM в отслеживаемой cgroup, не обязательно только этого процесса.",
        "- RSS может дважды учитывать общие страницы; VRAM опрашивается раз в ~10 с, короткие пики могут быть пропущены.",
        "- Подготовка включает загрузку/распаковку/инициализацию. Run включает повторные загрузки адаптера по стадиям.",
        "- ETA ZoDiac относится только к оставшейся оптимизации. Холодное время одной картинки нельзя напрямую умножать на весь датасет.",
        "- Для продолжения выполните этот же notebook заново. Завершённые артефакты проверяются SHA256.",
        "- Смена GPU/Python/весов создаёт новые execution fingerprints; частичный результат автоматически не смешивается.",
        "- Веса после неудачи сохраняются только на локальном SSD. После удаления среды они скачиваются повторно.",
        "- Лимит времени не отключает оплату Colab. Отключите среду после работы.",
        "",
    ]
    atomic_bytes(root / "DIAGNOSTIC_REPORT.md", "\n".join(lines).encode("utf-8"))
    archive_path = root / "diagnostic_report_share.zip"
    # Explicit allowlist: do not accidentally publish extraction keys or checkpoint tensors.
    files = [
        root / n
        for n in (
            "DIAGNOSTIC_REPORT.md",
            "CANARY_REPORT.md",
            "canary_report.json",
            "canary_report.csv",
            "settings.json",
            "catalog.json",
            "dataset_manifest.json",
        )
    ]
    for row in report["rows"]:
        out = root / "models" / row["model_id"]
        files += [
            out / n
            for n in (
                "status.json",
                "resource_preflight.json",
                "prepare_resources.json",
                "run_resources.json",
                "prepare_error.json",
                "run_error.json",
                "roundtrip_result.json",
            )
        ]
        files += list((out / "logs").glob("*.log")) + list((out / "logs").glob("*_telemetry.jsonl"))
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return archive_path
