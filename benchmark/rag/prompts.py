"""RAG-augmented prompts for the v0.1 sweep.

Distinct from the top-level `benchmark/prompts.py` so that direct-mode
runs are unaffected. The user-template builder reuses the option layout
from the direct prompts to keep the de-biased MCQ presentation consistent.
"""

from __future__ import annotations

from typing import Iterable


# ============================================================
# 1. MCQ — RAG-augmented
# ============================================================
RAG_MCQ_SYSTEM = (
    "Ты — эксперт по биобанкированию. Тебе будет дан фрагменты из проверенных "
    "источников (ISBER Best Practices, программа курса НИИ Отта, статьи "
    "Российского кардиологического журнала) и вопрос с вариантами ответа.\n\n"
    "Используй ТОЛЬКО контекст ниже. Если ответ требует знаний за пределами "
    "контекста — выбери единственно безопасный вариант и сошлись на "
    '"src:нет". Иначе сошлись на номер фрагмента: "src:1", "src:2", ...\n\n'
    'Верни валидный JSON строго в формате '
    '{"letters": ["B"], "citations": ["src:2", "src:5"]} '
    'или {"letters": ["A","C"], "citations": ["src:1"]} — '
    "поле `letters` это массив букв правильных вариантов (A..E), "
    "поле `citations` это НЕПУСТОЙ массив строк, каждая строка строго "
    'вида "src:N" (где N — номер фрагмента из контекста) или "src:нет" '
    "(если ни один фрагмент не подошёл). Минимум один элемент в citations. "
    "Без пояснений вне JSON."
)


# ============================================================
# 2. Open-ended — RAG-augmented
# ============================================================
RAG_OPEN_SYSTEM = (
    "Ты — эксперт по биобанкированию. Тебе даны фрагменты из проверенных "
    "источников (ISBER Best Practices, программа курса НИИ Отта, статьи "
    "Российского кардиологического журнала) и ситуационный вопрос.\n\n"
    "Дай конкретный ответ-руководство для сотрудника биобанка. "
    "Опирайся на контекст ниже там, где он применим; пробелы — "
    "дополняй общими принципами биобанкирования.\n\n"
    "Формат СТРОГО:\n"
    "- 3-6 пронумерованных шагов с конкретными действиями (что сделать, "
    "в каком порядке, кого уведомить, что зафиксировать).\n"
    "- В конце каждого шага — метка `[src:N]` (номер фрагмента) или "
    "`[src:нет]` (если из общих принципов, не из контекста).\n"
    "- БЕЗ вступлений («Конечно», «По общим принципам», «Это важный»).\n"
    "- БЕЗ markdown (**жирного**, ###заголовков, ---).\n"
    "- БЕЗ теории. Только действия.\n"
    "- Лимит — 600 знаков (не считая [src:N] меток).\n\n"
    'Верни валидный JSON строго в формате '
    '{"answer": "...", "citations": ["src:2", "src:4"]}:\n'
    "- `answer` — строка с пронумерованными шагами в одной строке "
    "(\\n между шагами).\n"
    "- `citations` — НЕПУСТОЙ массив строк, каждая строго вида \"src:N\" "
    "или \"src:нет\". Перечисли ВСЕ источники, на которые ссылался.\n"
    "Без пояснений вне JSON."
)


# ============================================================
# 3. User templates
# ============================================================
def _format_context_block(context_chunks: Iterable[dict]) -> str:
    """Render the [src:N] context block. Each chunk dict needs at minimum
    `text`; we additionally render source/section/page if present so the
    model can mention the source in its citation."""
    lines: list[str] = []
    for idx, ch in enumerate(context_chunks, start=1):
        text = (ch.get("text") or "").strip()
        # Lightly trim very long chunks to keep prompt budget reasonable.
        # Truncation policy: 1200 chars ≈ 350 tokens. Soft cutoff at a
        # whitespace boundary so the model doesn't see a chopped word.
        if len(text) > 1200:
            cut = text.rfind(" ", 0, 1200)
            if cut < 800:
                cut = 1200
            text = text[:cut].rstrip() + " …"
        meta_bits = []
        if ch.get("source_short"):
            meta_bits.append(str(ch["source_short"]))
        if ch.get("section"):
            meta_bits.append(str(ch["section"]))
        if ch.get("page") is not None:
            meta_bits.append(f"стр. {ch['page']}")
        meta = f" (источник: {', '.join(meta_bits)})" if meta_bits else ""
        lines.append(f"[src:{idx}] {text}{meta}")
    return "\n\n".join(lines)


def rag_user_template(
    question: str,
    options: dict[str, str] | None,
    context_chunks: list[dict],
    *,
    context_found: bool = True,
) -> str:
    """Build the user message for either MCQ (options given) or open (options=None)."""
    if context_chunks:
        context_block = _format_context_block(context_chunks)
        header = "<контекст>\n" + context_block + "\n</контекст>"
    else:
        header = (
            "<контекст>\n"
            "(пусто — в индексе не нашлось подходящих фрагментов)\n"
            "</контекст>"
        )

    if not context_found:
        header += (
            "\n\nВНИМАНИЕ: автоматический поиск не уверен, что фрагменты "
            "выше отвечают на вопрос. Если они нерелевантны — используй "
            "общие знания и пометь утверждения `[src:нет]`."
        )

    if options:
        opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        body = f"Вопрос: {question}\n\nВарианты:\n{opts_text}"
    else:
        body = f"Вопрос: {question}"

    return f"{header}\n\n{body}"


# Schema-aware providers (OpenRouter, Yandex json_object, Anthropic tool-use)
# will enforce these; GigaChat will be parsed best-effort.
#
# DA review dd382f8, fix #2a: `citations` is REQUIRED and CONSTRAINED — an
# array of `src:N` / `src:нет` strings (regex-validated), not free-form text.
# This stops the model from emitting empty / hand-wavy citation fields that
# the post-hoc parser then has to guess at.
RAG_MCQ_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "letters": {
            "type": "array",
            "items": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
            "minItems": 1,
        },
        "citations": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^src:(\d+|нет)$"},
            "minItems": 1,
        },
    },
    "required": ["letters", "citations"],
    "additionalProperties": False,
}


# DA review dd382f8, fix #2b: open-ended now also gets a structured schema.
# Same `citations` constraint as MCQ; `letters` is replaced with `answer`
# (free-form string — the actual prose reply).
RAG_OPEN_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^src:(\d+|нет)$"},
            "minItems": 1,
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


__all__ = [
    "RAG_MCQ_ANSWER_SCHEMA",
    "RAG_MCQ_SYSTEM",
    "RAG_OPEN_ANSWER_SCHEMA",
    "RAG_OPEN_SYSTEM",
    "rag_user_template",
]
