"""Public metadata for each candidate model — sizes / context windows.

Source: vendor docs as of 2026-06; closed models keep tier labels since
parameter counts aren't disclosed.
"""

MODEL_META = {
    # Sber GigaChat (closed)
    "GigaChat-2":           {"size": "Lite tier (closed)", "context": "32K"},
    "GigaChat-2-Pro":       {"size": "Pro tier (closed)",  "context": "32K"},
    "GigaChat-2-Max":       {"size": "Max tier (closed)",  "context": "32K"},

    # Yandex AI Studio (closed)
    "yandexgpt/latest":         {"size": "Pro 5.1 (closed)",     "context": "32K"},
    "yandexgpt-lite/latest":    {"size": "Lite 5.1 (closed)",    "context": "32K"},
    "yandexgpt-32k/latest":     {"size": "long-ctx (closed)",    "context": "32K+"},

    # GPT-OSS hosted by Yandex (OpenAI open weights)
    "gpt-oss-120b/latest":      {"size": "120B (open)",          "context": "?"},
    "gpt-oss-20b/latest":       {"size": "20B (open)",           "context": "?"},

    # OpenRouter — DeepSeek V4
    "deepseek/deepseek-v4-pro":   {"size": "V4 flagship MoE (closed)", "context": "1M"},
    "deepseek/deepseek-v4-flash": {"size": "V4 flash (closed)",        "context": "1M"},

    # OpenRouter — Qwen
    "qwen/qwen3.7-max":           {"size": "3.7 flagship (closed MoE)", "context": "?"},
    "qwen/qwen3.6-35b-a3b":       {"size": "35B-A3B MoE (open)",        "context": "?"},
}


def lookup(model: str) -> dict[str, str]:
    """Return {'size','context'} for a model id; '?' for both if unknown."""
    return MODEL_META.get(model, {"size": "?", "context": "?"})
