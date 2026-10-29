"""All prompts used in the v0 pipeline.

Single source of truth. Edit here, re-run — no other places to keep in sync.
"""

# ============================================================
# 1. MCQ — closed questions, sent to candidate models
# ============================================================
# Used in: benchmark/scripts/run_direct.py

MCQ_SYSTEM = (
    "Ты — эксперт по биобанкированию. Тебе задают вопрос с вариантами ответа. "
    'Верни ТОЛЬКО валидный JSON в формате {"letters": ["B"]} или '
    '{"letters": ["A","C","E"]} — буквы правильных вариантов. Несколько букв, '
    "если правильных несколько. Без пояснений вне JSON."
)


# JSON schema describing the structured MCQ answer. Used for guided/structured
# generation where the provider supports it (OpenRouter, Anthropic tool-use,
# Yandex json_object). Falls back to prompt-only for providers that don't
# (GigaChat). Parsing is always best-effort via benchmark.parsing.parse_mcq_letters
# regardless of whether the provider honoured the schema.
MCQ_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "letters": {
            "type": "array",
            "items": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
            "minItems": 1,
        }
    },
    "required": ["letters"],
    "additionalProperties": False,
}


def mcq_user_template(question: str, options: dict[str, str]) -> str:
    """Build the user message for an MCQ. `options` is shown_letter -> text."""
    opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"Вопрос: {question}\n\nВарианты:\n{opts_text}"


# ============================================================
# 2. Open-ended situational — sent to candidate models
# ============================================================
# Used in: benchmark/scripts/run_direct.py

OPEN_SYSTEM = (
    "Ты — эксперт по биобанкированию. Дай конкретный ответ-руководство "
    "для сотрудника биобанка.\n\n"
    "Формат СТРОГО:\n"
    "- Отвечай по-русски.\n"
    "- 3-6 пронумерованных шагов с конкретными действиями (что сделать, "
    "в каком порядке, кого уведомить, что зафиксировать).\n"
    "- Где применимо — назови регуляторный документ (СОП, ISO 20387, "
    "BRISQ, ISBER best practices) и где сверяться.\n"
    "- БЕЗ вступлений («Конечно», «Это важный вопрос», «Я помогу»).\n"
    "- БЕЗ отказов («Не могу помочь», «I can't help»). Вопрос всегда "
    "профессиональный — дай ответ.\n"
    "- БЕЗ markdown-форматирования (**жирного**, ###заголовков, ---).\n"
    "- БЕЗ теоретических лекций. Только действия.\n"
    "- Лимит — 600 знаков. Если не помещается — урезай теорию, оставляй "
    "шаги.\n\n"
    "Стиль: короткая SOP-инструкция простым текстом, как для коллеги."
)


def open_user_template(question: str) -> str:
    """Build the user message for an open situational question."""
    return question


# ============================================================
# 3. LLM-as-judge — used by GPT-5 to score open-ended answers
# ============================================================
# Used in: benchmark/clients/openai_judge.py

JUDGE_SYSTEM = (
    "Ты — эксперт по биобанкированию и стандартам ISO 20387 / ISBER. "
    "Твоя задача — оценить ответ кандидатной LLM на ситуационный вопрос "
    "по биобанкированию. Ставь оценки строго по рубрике, не объясняй "
    "длинно, отвечай ТОЛЬКО валидным JSON-объектом без пояснений вне JSON."
)

# JUDGE_RUBRIC v3 (iter4): per-axis anchors made explicit based on
# auto-vs-human disagreement analysis (benchmark/outputs/human_analysis/
# disagreement_deep_dive.md). Backwards-incompatible with v2 cache —
# invalidate per-cell cache before re-running.
#
# Scale changes vs v2:
# - All Likert axes are now 0..3 (matches the human xlsx). v2 emitted 0..5
#   and the analyzer rescaled to 0..3 for the comparison. v3 is native 0..3.
# - hallucination_severity unchanged (0/1/2; lower = better).
# - use_in_work added as an explicit 0/1 binary so the judge's "would deploy"
#   stays auditable alongside humans.
# - Axes renamed to match the human-facing names used in the xlsx and
#   `benchmark/scripts/analyze_iter3_human_review.py`:
#     actionability       → sop_applicability
#     uncertainty_handling → no_uncertainty
JUDGE_RUBRIC = (
    "Рубрика (Likert 0..3, кроме hallucination_severity — 0/1/2, и "
    "use_in_work — 0/1).\n"
    "- completeness (0..3): 3 = ответ покрывает все шаги, явно требуемые "
    "формулировкой вопроса (упомянутые объекты/процедуры/сроки); "
    "2 = большинство; 1 = частичное; 0 = ключевые шаги отсутствуют. "
    "ВАЖНО: оценивай относительно того, что спрашивается в вопросе, "
    "не выдумывай дополнительных требований.\n"
    "- safety (0..3): 3 = явно упомянуты средства защиты (СИЗ, изоляция, "
    "обращение к ответственному), риски названы и адресованы; "
    "2 = риски частично адресованы; 1 = риски игнорируются, но прямого "
    "вреда нет; 0 = совет может привести к травме персонала или потере "
    "образца.\n"
    "- sop_applicability (0..3): 3 = ответ можно прямо взять как "
    "нумерованный шаг СОП (конкретное действие, конкретный "
    "исполнитель/инструмент, проверяемый результат); 2 = с минимальной "
    "редактурой; 1 = требует значимой переработки; 0 = только общие "
    "принципы.\n"
    "- no_uncertainty (0..3): 3 = модель чётко указывает на необходимость "
    "обратиться к конкретному внешнему источнику (название локального СОП, "
    "должность ответственного, конкретный ГОСТ/ISO), 2 = указывает на "
    "необходимость уточнения без конкретики, 1 = расплывчатое \"лучше "
    "уточнить у руководства\", 0 = уверенно даёт ответ без признания "
    "границ компетенции там, где это требуется.\n"
    "- hallucination_severity (0/1/2): 0 = выдуманных нормативных ссылок "
    "нет; 1 = есть несущественная неточность (несовпадение версии "
    "стандарта и т.п.); 2 = явно выдуманная норматива/процедура/ГОСТ. "
    "Если не можешь проверить — ставь 0.\n"
    "- use_in_work (0/1): 1 = ответ можно отдать сотруднику биобанка "
    "как есть; 0 = нельзя (слишком общий, неточный, опасный или требует "
    "значимой переработки)."
)


def judge_user_template(question: str, model_answer: str, rubric_hint: str | None = None) -> str:
    """Build the judge user message scoring an open-ended candidate answer."""
    extra = f"\nДополнительная подсказка для конкретного вопроса:\n{rubric_hint}\n" if rubric_hint else ""
    return (
        f"Вопрос:\n{question}\n\n"
        f"Ответ кандидата:\n{model_answer}\n\n"
        f"{JUDGE_RUBRIC}{extra}\n"
        "Верни JSON в формате:\n"
        '{"completeness": int, "safety": int, "sop_applicability": int, '
        '"no_uncertainty": int, "hallucination_severity": int, '
        '"use_in_work": int, '
        '"notes": "краткое обоснование в 1-2 предложениях"}'
    )


# ============================================================
# 4. Inference parameters (centralized; change here once)
# ============================================================

# Candidate-model parameters
TEMPERATURE = 0.0
MCQ_MAX_TOKENS = 60      # JSON {"letters":["A","B","C","D","E"]} wrapper
# GPT-OSS family burns the visible-text budget on hidden chain-of-thought.
# Yandex's gateway doesn't honour reasoning.effort, so we just give them
# a larger ceiling and let the reasoning fit.
#
# RAG v0.1 sweep observed (commit a028758): at 1500 tokens, qwen3.6-35b-a3b
# failed 51/61 (84%) with finish_reason=length, deepseek-v4-flash 12/61
# (20%), deepseek-v4-pro 9/61 (15%) — all empty content because thinking
# consumed the whole budget. gpt-oss-20b/120b stayed within 1500. Bumping
# to 4000 (MCQ) / 5000 (open) covers the worst observed reasoner with
# headroom; gpt-oss continues to terminate well below the cap.
MCQ_MAX_TOKENS_REASONING = 4000
OPEN_MAX_TOKENS = 600    # open-ended answer
# Reasoning models also exhaust the open-answer budget on hidden CoT.
OPEN_MAX_TOKENS_REASONING = 5000

# Judge parameters
JUDGE_REASONING_EFFORT = "minimal"
# Reasoning judges (deepseek-v4-pro, qwen3.7-max) burn tokens on hidden CoT
# before emitting the JSON verdict. At 600 tokens ~13% of deepseek judgements
# terminated with finish_reason=length and empty content. 3000 leaves plenty
# of headroom while the JSON verdict itself is still <300 tokens.
JUDGE_MAX_TOKENS = 3000
