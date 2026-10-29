"""Text normalisation for the RAG index and queries.

Two purposes:

1. **Homoglyph normalisation.** The corpus mixes scanned PDFs with manually
   typed Russian, so Cyrillic / Latin lookalikes (`е` ↔ `e`, `а` ↔ `a`,
   `о` ↔ `o`, ...) appear in both queries and indexed text. We map Latin
   lookalikes to their Cyrillic equivalents in any *all-Cyrillic context*
   (heuristic: a Latin char wedged between two Cyrillic chars is almost
   always a homoglyph artefact, not a true English token). This avoids
   false-positives in English passages of ISBER 5ed.

2. **Abbreviation expansion.** Biobanking jargon is dense with acronyms
   (ИДС, СОП, ПЦР, BRISQ, ISO 20387, ...). We *append* the canonical
   expansion to each occurrence so BM25 picks up either the acronym or
   the expansion, and so dense embeddings see both forms in context.
   Both `normalize_for_index` and `normalize_for_query` apply expansion;
   only the index pass is forced through homoglyph canonicalisation.

This module is import-clean (no heavy dependencies) so it can be reused
by the parser, the BM25 lemmatiser, and the dense indexer alike.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. Homoglyph map (Latin → Cyrillic) — only the safe ones for *Russian*
# ---------------------------------------------------------------------------
# Reference: Unicode TR-39 confusables, restricted to characters that occur in
# the visible glyph set of typed Russian (lowercase + uppercase).
_HOMOGLYPHS_LAT_TO_CYR: dict[str, str] = {
    # lower
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "i": "і",  # rare; only inside Cyrillic words. Falls through if no context.
    # upper
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
}
_LATIN_HOMOGLYPHS = set(_HOMOGLYPHS_LAT_TO_CYR.keys())


def _is_cyr(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    # U+0400..U+04FF (Cyrillic), U+0500..U+052F (Cyrillic Supplement)
    return 0x0400 <= cp <= 0x052F


def _is_lat(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def canonicalize_homoglyphs(text: str) -> str:
    """Replace Latin lookalikes wedged inside Cyrillic words with their
    Cyrillic equivalents. Leaves real English tokens (`ISO`, `BRISQ`, ...)
    alone because they have no Cyrillic neighbours on either side.
    """
    if not text:
        return text
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in _LATIN_HOMOGLYPHS:
            left = text[i - 1] if i > 0 else ""
            right = text[i + 1] if i + 1 < n else ""
            # Wedge rule: at least one neighbour is Cyrillic AND no English
            # neighbour. This catches `биoбанк` (Latin o) but spares `BIO`.
            if (_is_cyr(left) or _is_cyr(right)) and not (_is_lat(left) or _is_lat(right)):
                out.append(_HOMOGLYPHS_LAT_TO_CYR[ch])
                continue
        out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. Abbreviation expansion — append canonical form on each match
# ---------------------------------------------------------------------------
# Keep this list short and load-bearing: only acronyms that are also asked
# about in the dataset or appear in the corpus's controlled vocabulary.
# Format: surface form -> canonical expansion (Russian preferred).
ABBREVIATIONS: dict[str, str] = {
    "ИДС": "информированное добровольное согласие",
    "СОП": "стандартная операционная процедура",
    "ПЦР": "полимеразная цепная реакция",
    "ISO 20387": "ISO 20387 биобанкирование общие требования",
    "ISO-20387": "ISO 20387 биобанкирование общие требования",
    "ISO20387": "ISO 20387 биобанкирование общие требования",
    "BRISQ": "Biospecimen Reporting for Improved Study Quality стандарт отчётности по биообразцам",
    "ISBER": "International Society for Biological and Environmental Repositories",
    "ESBB": "European Middle Eastern and African Society for Biopreservation and Biobanking",
    "ДНК": "дезоксирибонуклеиновая кислота",
    "РНК": "рибонуклеиновая кислота",
    "СПО": "стандартный протокол отбора",
    "QC": "контроль качества quality control",
    "QA": "обеспечение качества quality assurance",
}

# Pre-compile a single regex that matches any abbreviation as a whole token.
# Sort by length desc so `ISO 20387` matches before `ISO`.
_ABBR_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-я])(" +
    "|".join(re.escape(k) for k in sorted(ABBREVIATIONS.keys(), key=len, reverse=True)) +
    r")(?![A-Za-zА-Яа-я0-9])"
)


def expand_abbreviations(text: str) -> str:
    """Append the canonical expansion next to each abbreviation match.

    `СОП` -> `СОП (стандартная операционная процедура)`. We append rather
    than replace so the original acronym stays in the BM25 index (people
    *do* search for `СОП`) but the expansion adds dense-embedding context.

    Idempotent: if the next characters after the match are already the
    expected ` (expansion)` suffix, we leave the token alone (DA #53
    finding #2). Without this guard, calling `normalize_text` twice on
    the same string would double-wrap acronyms.
    """
    if not text:
        return text

    def _repl(m: re.Match[str]) -> str:
        token = m.group(1)
        expansion = ABBREVIATIONS.get(token)
        if not expansion:
            return token
        # Guard 1: already expanded — next chars are " (expansion)".
        expected_suffix = f" ({expansion})"
        if text[m.end():m.end() + len(expected_suffix)] == expected_suffix:
            return token
        # Guard 2: this match is INSIDE a previous expansion's parens. The
        # only way that happens is if a previous pass already inserted
        # "token (expansion…)" and `expansion` itself starts with `token`
        # (e.g. ISO 20387 maps to "ISO 20387 биобанкирование…"). Detect by
        # checking that the preceding char is `(` AND the suffix after
        # `token` matches `expansion[len(token):] + ")"`.
        if m.start() > 0 and text[m.start() - 1] == "(":
            tail = expansion[len(token):] + ")"
            if text[m.end():m.end() + len(tail)] == tail:
                return token
        return f"{token} ({expansion})"

    return _ABBR_PATTERN.sub(_repl, text)


# ---------------------------------------------------------------------------
# 3. Whitespace and Unicode normalisation
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# 4. Public entry points
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Generic normalisation used at *both* indexing and query time.

    1. NFKC unicode normalisation
    2. Homoglyph canonicalisation (Latin lookalikes inside Cyrillic words)
    3. Abbreviation expansion
    4. Whitespace collapse
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = canonicalize_homoglyphs(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def normalize_for_index(text: str) -> str:
    """Alias kept for callsite clarity."""
    return normalize_text(text)


def normalize_for_query(text: str) -> str:
    """Alias kept for callsite clarity."""
    return normalize_text(text)


def batch_normalize(texts: Iterable[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


__all__ = [
    "ABBREVIATIONS",
    "batch_normalize",
    "canonicalize_homoglyphs",
    "collapse_whitespace",
    "expand_abbreviations",
    "normalize_for_index",
    "normalize_for_query",
    "normalize_text",
]
