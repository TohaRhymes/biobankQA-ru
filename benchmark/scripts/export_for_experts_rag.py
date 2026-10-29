"""Build the v0.1 experts xlsx with Direct + RAG answers side-by-side.

Extends the v0 file (`biobankqa_open_for_experts_v0.xlsx`) so the medical
team can blind-rate both modes:

- 14 open questions × 6 candidate models × 2 modes (direct, rag) = 168 rows
- Models randomised onto anonymous letters A-L (seed=42); reviewer cannot
  tell from the letter alone which model or which mode they are scoring
- RAG answers have `[src:N]` citation tags stripped to the answer text
  only (per owner decision: keep the eval focused on substance)
- Maps sheet stays hidden behind a "do not open until done" header

Input:
  - data/processed/expert_review/biobankqa_open_for_experts_v0.xlsx
    (instructions sheet copied verbatim)
  - data/processed/biobankqa_v0_executable.jsonl
  - benchmark/outputs/raw/raw_outputs.jsonl  (direct mode)
  - benchmark/outputs/raw/rag_outputs.jsonl  (RAG mode)

Output:
  - data/processed/expert_review/biobankqa_open_for_experts_v0.1_rag.xlsx
"""

from __future__ import annotations

import json
import random
import re
import sys
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
RAW_DIRECT = REPO_ROOT / "benchmark" / "outputs" / "raw" / "raw_outputs.jsonl"
RAW_RAG = REPO_ROOT / "benchmark" / "outputs" / "raw" / "rag_outputs.jsonl"

# Output goes into a per-iteration directory under iterations/. Bump this
# constant when starting a new iteration (or override with --iter on the
# CLI). The medical team picks up the highest-numbered iter as "latest".
ITER_DEFAULT = "iter3"
ITER_DIR_ROOT = REPO_ROOT / "data" / "processed" / "expert_review" / "iterations"

# Инструкция sheet is copied verbatim from the iter1 xlsx (rubric is
# stable across iterations). The legacy v0 baseline xlsx was removed
# because its content is fully subsumed by iter1/Direct.
SRC_XLSX = ITER_DIR_ROOT / "iter1" / "biobankqa_open_for_experts.xlsx"

# The six models the medical team agreed to review in v0 — kept identical
# in v0.1 so the team isn't surprised by different model choices.
TARGET_MODELS: list[tuple[str, str]] = [
    ("yandex", "gpt-oss-120b/latest"),
    ("gigachat", "GigaChat-2"),
    ("yandex", "yandexgpt/latest"),
    ("openrouter", "deepseek/deepseek-v4-pro"),
    ("gigachat", "GigaChat-2-Max"),
    ("openrouter", "qwen/qwen3.7-max"),
]

LETTERS = list("ABCDEFGHIJKL")  # 12 anonymous labels for 12 (model, mode)
ANON_SEED = 42

# Strip `[src:N]` and bare `src:N` citations from answer text.
_CITE_PATTERN = re.compile(r"\s*\[?src:(?:\d+|нет|none)\]?\.?", re.IGNORECASE)

# Trailing-JSON tail (DA #54 SS#3): some models put `[src:нет]\n\n{"answer":...}`
# at the end of an otherwise-prose answer. Match only when the JSON object is
# preceded by NON-JSON content (require at least one non-`{` char of prose
# before — otherwise we'd eat the whole structured response too).
_TRAILING_JSON_PATTERN = re.compile(
    r"(?<=[^\s{])\s*\{\s*\"(?:answer|citations|letters)\"\s*:.*?\}\s*$",
    re.DOTALL,
)

# Refusal-phrase rewrite (DA #54 SS#2): RAG models often open with
# "В предоставленных источниках …" — 0 direct cells use this, so it
# de-anonymises the mode. Replace with a neutral disclaimer.
_REFUSAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^В\s+предоставленных\s+источниках\s+прямого\s+ответа\s+нет[.,!?]?\s*",
                re.IGNORECASE),
     "Данных для прямого ответа недостаточно. "),
    (re.compile(r"^В\s+предоставленных\s+источниках\s+нет\s+(?:прямого\s+)?ответа\s+на\s+этот\s+вопрос[.,!?]?\s*",
                re.IGNORECASE),
     "Данных для прямого ответа недостаточно. "),
    (re.compile(r"^В\s+предоставленных\s+(?:фрагментах|материалах|источниках)\s+(?:прямого\s+)?ответа\s+нет[.,!?]?\s*",
                re.IGNORECASE),
     "Данных для прямого ответа недостаточно. "),
    (re.compile(r"^В\s+предоставленных\s+источниках[^.!?]{0,80}[.!?]\s*",
                re.IGNORECASE),
     "Данных для прямого ответа недостаточно. "),
]


@dataclass(frozen=True)
class Variant:
    """One (model, mode) combination — gets one anonymous letter."""
    provider: str
    model: str
    mode: str

    def display(self) -> str:
        m = "Direct (без контекста)" if self.mode == "direct" else "RAG (с контекстом из материалов)"
        return f"{self.model} — {m}"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_citations(text: str | None) -> str:
    if not text:
        return ""
    s = text.strip()
    # Unwrap markdown code fence first (Yandex returns ```json …```).
    m = _FENCE_PATTERN.match(s)
    if m:
        s = m.group(1).strip()
    # Malformed-JSON pre-strip (DA #56 SS#1): models sometimes emit broken
    # answer-key wrappers like:
    #   " answer: 1 - ...
    #   "ответ": 1. ...
    #   "ответ": {"answer": "1. ..."}        (double-wrapped)
    # The format never closes cleanly so json.loads is no help. Iteratively
    # strip leading wrapper keys (English OR Russian) until none match.
    _key_re = re.compile(
        r'^["\{]?\s*["\']?\s*(?:answer|ответ|response|результат)\s*["\']?\s*:\s*["\']?\s*',
        re.IGNORECASE,
    )
    for _ in range(3):
        m_q = _key_re.match(s)
        if not m_q:
            break
        s = s[m_q.end():]
    # Trim trailing citation-block tails the model tacked on.
    for marker in ('", "citations"', '","citations"', '", citations', '", "ссылк', '","ссылк'):
        idx = s.find(marker)
        if idx > 0:
            s = s[:idx]
            break
    s = s.rstrip('"\}\\').strip()
    # If the payload is a structured JSON with an `answer` field, take it.
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                ans = obj.get("answer") or obj.get("text") or ""
                if ans:
                    s = ans
        except json.JSONDecodeError:
            pass
    # Strip a trailing JSON object literal — some models tack the structured
    # answer on AFTER the prose (DA #54 SS#3). Repeat until stable in case
    # there are multiple trailing braces.
    while True:
        m2 = _TRAILING_JSON_PATTERN.search(s)
        if not m2:
            break
        s = s[:m2.start()].rstrip()
    # Second JSON-unwrap pass: trailing-JSON strip may have revealed a now-
    # valid leading JSON object literal that the first pass couldn't parse
    # (e.g. `{"answer":"…"} {"citations":[…]}` → after strip: `{"answer":"…"}`).
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                ans = obj.get("answer") or obj.get("text") or ""
                if ans:
                    s = ans
        except json.JSONDecodeError:
            pass
    # Strip inline [src:N] / src:N tokens.
    cleaned = _CITE_PATTERN.sub("", s)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _neutralize_refusal(text: str) -> str:
    """Rewrite mode-revealing 'В предоставленных источниках …' lead-ins to
    a neutral disclaimer that direct mode could also plausibly emit
    (DA #54 SS#2)."""
    if not text:
        return text
    for pat, repl in _REFUSAL_PATTERNS:
        new = pat.sub(repl, text, count=1)
        if new != text:
            return new
    return text


def _build_lookup(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Index by (provider, model, item_id) — both modes coexist."""
    out: dict[tuple[str, str, str], dict] = {}
    for r in records:
        key = (r["provider"], r["model"], r["item_id"])
        # If multiple records (e.g. retry kept old + new), prefer non-error
        existing = out.get(key)
        if existing is None or (existing.get("error") is not None and r.get("error") is None):
            out[key] = r
    return out


def _copy_sheet(src_ws, dst_wb, title: str):
    """Deep-copy a sheet (cells, basic styles, merges) — openpyxl can't
    do a true cross-workbook clone, so we do it field-by-field."""
    dst = dst_wb.create_sheet(title)
    for row in src_ws.iter_rows():
        for cell in row:
            new_cell = dst.cell(row=cell.row, column=cell.column, value=cell.value)
            if cell.has_style:
                new_cell.font = copy(cell.font)
                new_cell.fill = copy(cell.fill)
                new_cell.alignment = copy(cell.alignment)
                new_cell.border = copy(cell.border)
                new_cell.number_format = cell.number_format
    for mc in src_ws.merged_cells.ranges:
        dst.merge_cells(str(mc))
    for col_letter, dim in src_ws.column_dimensions.items():
        if dim.width:
            dst.column_dimensions[col_letter].width = dim.width
    return dst


def main() -> int:
    if not SRC_XLSX.exists():
        print(f"FATAL: source xlsx not found at {SRC_XLSX}", file=sys.stderr)
        return 2
    if not RAW_RAG.exists():
        print(f"FATAL: RAG outputs not found at {RAW_RAG}", file=sys.stderr)
        return 2
    if not RAW_DIRECT.exists():
        print(f"FATAL: direct outputs not found at {RAW_DIRECT}", file=sys.stderr)
        return 2

    # ---- Load dataset (open items, in id order) ----
    items = [it for it in _load_jsonl(DATASET) if it["type"] == "open"]
    items.sort(key=lambda it: it["id"])
    print(f"Open items: {len(items)}", file=sys.stderr)
    if len(items) != 14:
        print(f"WARN: expected 14 open items, got {len(items)}", file=sys.stderr)

    # ---- Load + index raw outputs ----
    direct_lookup = _build_lookup(_load_jsonl(RAW_DIRECT))
    rag_lookup = _build_lookup(_load_jsonl(RAW_RAG))

    # ---- Build all variants ----
    variants: list[Variant] = []
    for prov, model in TARGET_MODELS:
        for mode in ("direct", "rag"):
            variants.append(Variant(prov, model, mode))
    if len(variants) > len(LETTERS):
        print(f"FATAL: {len(variants)} variants > {len(LETTERS)} letter slots", file=sys.stderr)
        return 2

    # Deterministic anon assignment with constraint: direct and rag of the
    # SAME model must not land on adjacent letters (DA #54 R4). Try seeds
    # until one satisfies the constraint; bail with the unconstrained
    # shuffle if no seed works in a reasonable budget.
    def _gap_ok(mapping: list[Variant]) -> bool:
        for prov, model in TARGET_MODELS:
            pos_direct = next(i for i, v in enumerate(mapping)
                              if v.provider == prov and v.model == model and v.mode == "direct")
            pos_rag = next(i for i, v in enumerate(mapping)
                           if v.provider == prov and v.model == model and v.mode == "rag")
            if abs(pos_direct - pos_rag) < 2:
                return False
        return True

    chosen_seed: int | None = None
    shuffled: list[Variant] = []
    for cand_seed in range(ANON_SEED, ANON_SEED + 1000):
        rng = random.Random(cand_seed)
        cand = list(variants)
        rng.shuffle(cand)
        if _gap_ok(cand):
            shuffled = cand
            chosen_seed = cand_seed
            break
    if not shuffled:
        rng = random.Random(ANON_SEED)
        shuffled = list(variants)
        rng.shuffle(shuffled)
        chosen_seed = ANON_SEED
    print(f"Anon seed: {chosen_seed} (adjacency constraint satisfied: "
          f"{_gap_ok(shuffled)})", file=sys.stderr)
    letter_to_variant: dict[str, Variant] = dict(zip(LETTERS[:len(shuffled)], shuffled))
    variant_to_letter: dict[Variant, str] = {v: L for L, v in letter_to_variant.items()}

    # ---- Open source workbook for instructions + style template ----
    src_wb = openpyxl.load_workbook(SRC_XLSX)

    # ---- Build destination workbook ----
    dst_wb = openpyxl.Workbook()
    dst_wb.remove(dst_wb.active)

    # 1. Instruction sheet — copy verbatim from source, then append a
    #    DA #55 advisory about mode-leakage signals.
    if "Инструкция" in src_wb.sheetnames:
        ins_ws = _copy_sheet(src_wb["Инструкция"], dst_wb, "Инструкция")
        # Append: blank row, header, three caveat bullets.
        start = ins_ws.max_row + 2
        ins_ws.cell(row=start, column=1, value="⚠️ ВАЖНО — об оценке вслепую").font = Font(bold=True, color="CC0000")
        ins_ws.cell(
            row=start + 1, column=1,
            value=(
                "В этой версии (v0.1) для каждого вопроса показаны ответы 6 моделей "
                "в двух режимах (12 анонимных букв A–L). Эти режимы — Direct (модель "
                "отвечает по своим знаниям) и RAG (модель отвечает с привлечением "
                "фрагментов из материалов биобанка) — производят разные по длине, "
                "форматированию и стилю ответы. Возможные «подсказки», по которым "
                "можно догадаться о режиме или модели:"
            ),
        ).alignment = Alignment(wrap_text=True, vertical="top")
        bullets = [
            "Длина ответа. Direct-ответы в среднем длиннее. НЕ оценивайте по длине — "
            "длина не значит качество.",
            "Markdown-форматирование (**жирный**, ###заголовки, маркированные списки). "
            "Некоторые модели любят его, некоторые нет — это не показатель полезности.",
            "Стартовая фраза «По общим принципам биобанкирования…» появляется только в "
            "RAG-режиме, когда модель решает что контекст не отвечает на вопрос. Это "
            "не оценка качества — это честная попытка модели предупредить о "
            "неприменимости контекста.",
            "Пожалуйста, оценивайте СОДЕРЖАНИЕ каждого ответа независимо от подсказок "
            "о режиме/модели — иначе оценка перестаёт отражать качество и начинает "
            "отражать ваше предположение о источнике.",
        ]
        for i, b in enumerate(bullets, start=2):
            c = ins_ws.cell(row=start + i, column=1, value=f"• {b}")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        # Widen column so the wrapped text is readable.
        ins_ws.column_dimensions["A"].width = max(
            ins_ws.column_dimensions["A"].width or 0, 100
        )

    # 2. Open questions sheet — 168 rows.
    ws = dst_wb.create_sheet("Открытые вопросы")
    headers = [
        "№", "Вопрос", "Модель", "Ответ",
        "Полнота (0-5)", "Безопасность (0-5)", "SOP-применимость (0-5)",
        "Галлюцинации (0/1/2)", "Неопределённость (0-5)",
        "Использовали бы в работе? (Y/N)", "Комментарий",
    ]
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 8
    ws.column_dimensions["D"].width = 90
    for col_letter in "EFGHIJ":
        ws.column_dimensions[col_letter].width = 14
    ws.column_dimensions["K"].width = 40

    n_missing = 0
    for q_idx, item in enumerate(items, start=1):
        question_text = item["question"]
        # 12 letters per question, in alphabetical order so each question
        # block looks consistent.
        for letter in LETTERS[:len(variants)]:
            var = letter_to_variant[letter]
            lookup = rag_lookup if var.mode == "rag" else direct_lookup
            rec = lookup.get((var.provider, var.model, item["id"]))
            if rec is None or rec.get("error") is not None:
                answer_text = ""
                n_missing += 1
            else:
                answer_text = _strip_citations(rec.get("response"))
            row = [q_idx, question_text, letter, answer_text, None, None, None, None, None, None, None]
            ws.append(row)
            # Wrap long cells.
            for cell in ws[ws.max_row]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    print(f"Written {(ws.max_row - 1)} answer rows; {n_missing} missing/error cells.", file=sys.stderr)

    # Freeze header.
    ws.freeze_panes = "A2"

    # 3. Mapping sheet — hidden behind a warning header (same convention as v0).
    map_ws = dst_wb.create_sheet("Маппинг моделей")
    warning_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")
    map_ws["A1"] = "⚠️ Не открывать, пока не закончили оценку"
    map_ws["A1"].font = Font(bold=True, color="CC0000")
    map_ws["A1"].fill = warning_fill
    map_ws.merge_cells("A1:D1")
    map_ws.append([None] * 4)
    map_ws.append(["Анонимная буква", "Реальная модель", "Провайдер", "Режим"])
    for c in map_ws[3]:
        c.font = Font(bold=True)
    for letter in LETTERS[:len(variants)]:
        v = letter_to_variant[letter]
        mode_label = "Direct (без контекста)" if v.mode == "direct" else "RAG (с контекстом материалов)"
        map_ws.append([letter, v.model, v.provider, mode_label])
    map_ws.column_dimensions["A"].width = 16
    map_ws.column_dimensions["B"].width = 36
    map_ws.column_dimensions["C"].width = 14
    map_ws.column_dimensions["D"].width = 36

    DST_XLSX.parent.mkdir(parents=True, exist_ok=True)
    dst_wb.save(DST_XLSX)
    print(f"WROTE {DST_XLSX}", file=sys.stderr)
    print(file=sys.stderr)
    print("Anonymous mapping (don't share with reviewers):", file=sys.stderr)
    for letter in LETTERS[:len(variants)]:
        v = letter_to_variant[letter]
        print(f"  {letter} -> {v.provider:11s}  {v.model:32s}  [{v.mode}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
