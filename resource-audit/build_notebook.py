"""Build a self-contained, readable CPU-only notebook (no encoded payload)."""
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def build():
    cells = []

    def cell(kind, text):
        item = dict(cell_type=kind, id=f"audit-{len(cells):02d}", metadata={}, source=text.strip() + "\n")
        if kind == "code":
            ast.parse(item["source"])
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """# Журнал вычислительных экспериментов

Отдельный отчёт для научного руководителя: что запускалось, сколько времени заняло,
какая память использовалась и где были остановки. Только по сохранённым свидетельствам.

**Запуск:** загрузите этот файл в отдельный Colab, выберите **CPU**, выполните все ячейки
и разрешите чтение своего Google Drive. Текущий GPU-бенчмарк не останавливайте и его
тип среды не меняйте. Этот ноутбук не устанавливает библиотеки, не запускает модели,
не удаляет и не исправляет исходные данные.

Здесь учитываются испытания готовых моделей, а не автоматически «обучение в течение месяца».
История может быть неполной. Календарный период, сумма длительностей этапов и оплаченные
GPU-часы — разные величины. Списания Colab по журналам экспериментов не восстанавливаются.
""")
    cell("markdown", "## 1. Подключение Drive")
    cell("code", """from pathlib import Path
try:
    from google.colab import drive
except ImportError:
    print("Локальный запуск: укажите BASE в следующей ячейке.")
else:
    drive.mount("/content/drive")
""")
    cell("markdown", """## 2. Настройки

По умолчанию читается вся доступная история из известных папок наших прогонов.
Для одного месяца можно задать, например, `START_DATE = "2026-09-25"`,
`END_DATE = "2026-10-25"`. Последний день включается, время — московское.
Для старых/перенесённых прогонов добавьте точные папки в `EXTRA_ROOTS`.
Не выбирайте весь MyDrive как папку эксперимента.

Кеш ускоряет повторный разбор неизменившихся файлов по размеру и mtime.
Это не повторная криптографическая проверка: для неё поставьте `USE_METADATA_CACHE = False`.
После срабатывания лимита отчёт будет явно неполным; можно увеличить лимит и перезапустить.
""")
    cell("code", """BASE = Path("/content/drive/MyDrive")  # Локально: папка с копиями результатов.
START_DATE = None  # "2026-09-25" или None: без нижней границы.
END_DATE = None    # "2026-10-25" или None: до момента запуска аудита.
EXTRA_ROOTS = []   # Например: [BASE / "another_saved_experiment"]
OUTPUT_BASE = BASE / "watermarking_resource_audit"
USE_METADATA_CACHE = True
MAX_FILES = 20_000
MAX_READ_MIB = 2_048
MAX_AUDIT_MINUTES = 30
""")
    module = (ROOT / "resource_audit.py").read_text(encoding="utf-8")
    cell("markdown", """## 3. Код аудита

Обычный Python: чтение структурированных JSON/JSONL, подсчёт интервалов, HTML и таблицы.
Нет скрытых архивов, Base64, сетевых загрузок весов или обращения к GPU.
""")
    # Keep the module docstring/future import in the first code cell.
    for part in module.split("# %% "):
        if not part.strip():
            continue
        if part.startswith('"""'):
            cell("code", part)
        else:
            title, source = part.split("\n", 1)
            cell("markdown", "### " + title)
            cell("code", source)
    cell("markdown", """## 4. Построить отчёт

Читаются только известные каталоги `timings`, `logs`, `_logs` и их варианты в `_meta`.
Каталоги изображений, весов, атак и отдельных результатов не обходятся.
Следите за списком найденных папок: отсутствующая история не считается нулевой работой.
""")
    cell("code", """ROOTS = choose_roots(BASE, EXTRA_ROOTS)
print("Найдено папок экспериментов:", len(ROOTS))
for root in ROOTS:
    print(" •", root)
if not ROOTS:
    raise FileNotFoundError("Папки прогонов не найдены. Проверьте BASE / EXTRA_ROOTS и подключение Drive.")

SUMMARY = run_audit(
    ROOTS, OUTPUT_BASE, START_DATE, END_DATE,
    max_files=MAX_FILES, max_total_mib=MAX_READ_MIB,
    max_minutes=MAX_AUDIT_MINUTES, use_metadata_cache=USE_METADATA_CACHE,
)
REPORT_DIR = Path(SUMMARY["output"])
""")
    cell("markdown", """## 5. Результат

Покажите руководителю `REPORT.html`: его можно открыть в браузере и напечатать в PDF.
Таблицы `methods.csv`, `attempts.csv`, `daily.csv` содержат подробности.
В `evidence_manifest.json` — источники и хеши прочитанных снимков,
в `issues.json` — пропуски и ошибки чтения. Все файлы сохраняются на Drive в отдельную папку.

`returned_zero` означает завершение процесса с кодом 0, **не** победу метода и не полноту
100-изображенного бенчмарка. Сводка включает исторические диагностические версии методов.
Ошибки запуска и остановки по бюджету — тоже часть работы, а не «потерянное» время.
""")
    cell("code", """try:
    from IPython.display import HTML, display
except ImportError:
    print("HTML-просмотр доступен в Jupyter/Colab; локально откройте REPORT.html в браузере.")
else:
    display(HTML((REPORT_DIR / "REPORT.html").read_text(encoding="utf-8")))
print("Папка отчёта:", REPORT_DIR)
print("Сумма длительностей этапов, ч:", round(SUMMARY["recorded_stage_hours"], 2))
print("Календарное покрытие без перекрытий, ч:", round(SUMMARY["observed_calendar_hours"], 2))
print("Расход денег / compute units: неизвестен без биллинга.")
""")
    cell("markdown", """## 6. Упаковать отчёт для отправки

Архив содержит только новый отчёт и его таблицы, не сами изображения/веса/биты сообщений.
Отправлять его автоматически ноутбук не будет. Скачайте ZIP через боковую панель «Файлы»
или из Google Drive. Он может содержать имена папок и методов; просмотрите перед отправкой.
""")
    cell("code", """from zipfile import ZipFile, ZIP_DEFLATED
ARCHIVE = REPORT_DIR.with_suffix(".zip")
with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(REPORT_DIR.iterdir()):
        if path.is_file():
            archive.write(path, arcname=path.name)
print("Отчёт для научника:", ARCHIVE)
""")
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.10"),
        colab=dict(name="BENCHMARK_Resource_Audit.ipynb"),
        resource_audit=dict(version="1.0.0", source_sha256=hashlib.sha256(module.encode()).hexdigest())),
        nbformat=4, nbformat_minor=5)
    destination = ROOT / "dist" / "BENCHMARK_Resource_Audit.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    with ZipFile(destination.with_name("BENCHMARK_Resource_Audit_source.zip"), "w", compression=ZIP_DEFLATED) as archive:
        for name in ("resource_audit.py", "build_notebook.py", "README.md", "VALIDATION.md", "tests/test_resource_audit.py"):
            archive.write(ROOT / name, arcname=name)
        archive.write(destination, arcname=destination.name)
    print(destination)
    return destination


if __name__ == "__main__":
    build()
