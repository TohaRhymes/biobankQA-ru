"""Build the MCQ Direct-vs-RAG comparison xlsx for the medical team.

Independent of the OPEN iteration cycle: the MCQ prompts (`MCQ_SYSTEM`,
`RAG_MCQ_SYSTEM`) did not change across iter1/iter2/iter3, so the MCQ
results are stable — one snapshot covers the whole v0.1 cycle.

For each of the 47 MCQ items, show the gold answer + for each of the 12
models, the letters picked in Direct mode and in RAG mode. Lets the
medical team see at a glance which models RAG helps / hurts on closed
questions.

Output: `data/processed/expert_review/biobankqa_mcq_comparison.xlsx`
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
SCORED = REPO_ROOT / "benchmark" / "outputs" / "scored" / "scored.jsonl"
OUT_PATH = REPO_ROOT / "data" / "processed" / "expert_review" / "biobankqa_mcq_comparison.xlsx"

# Same six models the open xlsx uses, in same canonical order — keeps the
# two deliverables visually aligned for the reviewer.
TARGET_MODELS: list[tuple[str, str]] = [
    ("yandex", "gpt-oss-120b/latest"),
    ("gigachat", "GigaChat-2"),
    ("yandex", "yandexgpt/latest"),
    ("openrouter", "deepseek/deepseek-v4-pro"),
    ("gigachat", "GigaChat-2-Max"),
    ("openrouter", "qwen/qwen3.7-max"),
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt_letters(letters: list[str] | None) -> str:
    if not letters:
        return ""
    return ",".join(letters)


def _fmt_gold(item: dict) -> str:
    gold = item.get("gold_letters_canonical") or item.get("gold_letters") or []
    if not gold and item.get("gold_answer_canonical"):
        gold = [item["gold_answer_canonical"]]
    if not gold and item.get("gold_answer"):
        gold = [item["gold_answer"]]
    return ",".join(gold) if gold else "?"


def main() -> int:
    items_by_id = {it["id"]: it for it in _load_jsonl(DATASET) if it["type"] == "mcq"}
    print(f"MCQ items in dataset: {len(items_by_id)}", file=sys.stderr)

    scored = _load_jsonl(SCORED)
    by_key: dict[tuple[str, str, str, str], dict] = {}
    for rec in scored:
        if rec.get("item_type") != "mcq":
            continue
        key = (rec["provider"], rec["model"], rec["mode"], rec["item_id"])
        by_key[key] = rec

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MCQ Direct vs RAG"

    # Header: №, Вопрос, Правильный ответ, [model_direct], [model_rag] × 6
    header = ["№", "Вопрос", "Правильный\nответ"]
    model_headers = []
    for prov, model in TARGET_MODELS:
        # Friendly short label: drop "latest" + provider prefix segment.
        parts = [p for p in model.split("/") if p and p != "latest"]
        short_model = parts[-1] if parts else model
        model_headers.extend([f"{short_model}\nDirect", f"{short_model}\nRAG"])
    header.extend(model_headers)
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="center")

    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 12
    for i in range(len(model_headers)):
        ws.column_dimensions[chr(ord("D") + i)].width = 11

    # Sort items by item_id (mcq_001, mcq_002, ...)
    item_ids_sorted = sorted(items_by_id.keys())

    correct_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
    wrong_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    partial_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    empty_fill = PatternFill(start_color="E2E3E5", end_color="E2E3E5", fill_type="solid")

    for idx, item_id in enumerate(item_ids_sorted, start=1):
        item = items_by_id[item_id]
        gold_str = _fmt_gold(item)
        gold_set = set(gold_str.split(",")) if gold_str != "?" else set()

        row_cells: list = [idx, item["question"], gold_str]

        for prov, model in TARGET_MODELS:
            for mode in ("direct", "rag"):
                rec = by_key.get((prov, model, mode, item_id))
                if rec is None or rec.get("error") is not None:
                    row_cells.append("—")
                    continue
                letters = rec.get("mcq_letters_canonical") or []
                row_cells.append(_fmt_letters(letters))

        ws.append(row_cells)
        cur_row = ws.max_row
        # Color the per-model cells
        col = 4
        for prov, model in TARGET_MODELS:
            for mode in ("direct", "rag"):
                rec = by_key.get((prov, model, mode, item_id))
                cell = ws.cell(row=cur_row, column=col)
                cell.alignment = Alignment(horizontal="center", vertical="top")
                if rec is None or rec.get("error") is not None:
                    cell.fill = empty_fill
                else:
                    letters = set(rec.get("mcq_letters_canonical") or [])
                    if not letters:
                        cell.fill = empty_fill
                    elif letters == gold_set:
                        cell.fill = correct_fill
                    elif letters & gold_set:
                        cell.fill = partial_fill
                    else:
                        cell.fill = wrong_fill
                col += 1
        ws.cell(row=cur_row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=cur_row, column=3).alignment = Alignment(horizontal="center", vertical="top")

    ws.freeze_panes = "D2"

    # --- 2nd sheet: legend ---
    leg = wb.create_sheet("Легенда")
    leg["A1"] = "Условные обозначения цветов в основной таблице:"
    leg["A1"].font = Font(bold=True)
    rows = [
        ("", "зелёный", "ответ полностью совпадает с правильным"),
        ("", "жёлтый", "частичное пересечение (часть букв правильна)"),
        ("", "красный", "ответ полностью неправильный"),
        ("", "серый", "модель не ответила (ошибка API / пустой ответ / quota)"),
        ("", "", ""),
        ("Прочее", "", ""),
        ("", "MCQ-промпты", "(MCQ_SYSTEM, RAG_MCQ_SYSTEM) НЕ менялись между iter1/iter2/iter3, поэтому MCQ-таблица одна на все итерации."),
        ("", "Шаффлирование", "буквы вариантов перетасованы детерминированным seed=42 для каждого вопроса, затем расшифрованы обратно в исходные A-E для сравнения с gold."),
        ("", "Несколько правильных", "если правильных ответов несколько (например A,C), модель должна вернуть все. Частичное совпадение = жёлтый."),
    ]
    for i, (col1, col2, col3) in enumerate(rows, start=3):
        leg.cell(i, 1, col1).font = Font(bold=True)
        leg.cell(i, 2, col2)
        c = leg.cell(i, 3, col3)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for col_letter, w in [("A", 10), ("B", 14), ("C", 90)]:
        leg.column_dimensions[col_letter].width = w
    # Apply fill samples to col2
    leg["B3"].fill = correct_fill
    leg["B4"].fill = partial_fill
    leg["B5"].fill = wrong_fill
    leg["B6"].fill = empty_fill

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_PATH)
    print(f"WROTE {OUT_PATH}", file=sys.stderr)
    print(f"  {len(item_ids_sorted)} MCQ items × {len(TARGET_MODELS)} models × 2 modes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
