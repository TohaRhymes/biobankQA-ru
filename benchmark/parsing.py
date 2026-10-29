"""Lightweight parsing helpers used downstream of candidate-model generation.

`parse_mcq_letters` is the single source of truth for turning a model's
free-form MCQ response into a clean ordered list of A-E letters. It tries the
structured-JSON path first (matching MCQ_ANSWER_SCHEMA), then falls back to a
plain A-E regex over the raw text. Both paths preserve insertion order and
deduplicate.
"""

from __future__ import annotations

import json
import re


_LETTER_RE = re.compile(r"[ABCDE]")


def parse_mcq_letters(text: str | None) -> list[str]:
    """Try JSON {"letters": [...]} first, then fall back to A-E regex.

    Returns letters in the order encountered, deduplicated, uppercase.
    """
    if not text:
        return []
    s = text.strip()
    # strip code fences if present
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip("\n").strip()
    # JSON path: find the first {...} block and try to parse it.
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            arr = obj.get("letters") if isinstance(obj, dict) else None
            if isinstance(arr, list):
                out: list[str] = []
                for x in arr:
                    if isinstance(x, str) and x.upper() in "ABCDE" and x.upper() not in out:
                        out.append(x.upper())
                if out:
                    return out
        except json.JSONDecodeError:
            pass
    # Regex fallback (preserve insertion order, dedupe).
    seen: list[str] = []
    for hit in _LETTER_RE.finditer(s.upper()):
        if hit.group(0) not in seen:
            seen.append(hit.group(0))
    return seen
