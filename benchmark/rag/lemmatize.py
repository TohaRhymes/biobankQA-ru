"""pymorphy3 Russian lemmatiser used on the BM25 side.

Why a separate module?
----------------------
- BM25 wants normalised single-word tokens; Russian morphology means that
  the same stem appears as `биобанк`, `биобанка`, `биобанку`, `биобанков`
  in our corpus. Without lemmatisation BM25 misses 60-80% of valid hits
  for short queries.
- Dense embeddings (bge-m3) *do not* benefit from lemmatisation — they
  internalise morphology — so we only run this on BM25 documents and
  BM25 queries.

The wrapper instantiates pymorphy3 lazily (process-fork-safe) and
memoises by token so the cost is amortised over the ~2k chunks of the
corpus.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Iterable

# pymorphy3 is a hard requirement for build_index.py / retrieve.py
# but we import lazily so a fresh checkout that hasn't run
# `pip install -r requirements.txt` still lets the rest of the package
# import (e.g. for tests that don't touch BM25).
_morph_lock = threading.Lock()
_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        with _morph_lock:
            if _morph is None:  # double-checked under lock
                import pymorphy3  # type: ignore

                _morph = pymorphy3.MorphAnalyzer(lang="ru")
    return _morph


# ---------------------------------------------------------------------------
# Tokenisation: keep words, drop punctuation, keep Latin tokens as-is
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9][А-Яа-яЁёA-Za-z0-9\-]*")

# Minimal Russian stopword list. Borrowed from common Russian IR work; we
# keep it small because pymorphy3 + BM25 length-norm already do most of
# the work, and aggressive stoplisting hurts short noun-phrase queries.
RUSSIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только её мне было вот от меня ещё нет о из ему теперь когда даже ну вдруг ли
    если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь эту куда зачем всех никогда можно при наконец два об другой
    хоть после над больше тот через эти нас про всего них какая много разве три
    эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им
    более всегда конечно всю между
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Light tokeniser: lowercase + word-character regex."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@lru_cache(maxsize=200_000)
def _lemmatize_one(token: str) -> str:
    # Latin / numeric tokens: leave intact — pymorphy3 is Russian-only.
    if not token or token.isascii():
        return token
    try:
        morph = _get_morph()
        parses = morph.parse(token)
        if not parses:
            return token
        return parses[0].normal_form
    except Exception:
        return token


def lemmatize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokenise -> lemmatise -> (optionally) drop stopwords.

    Returns a list of lemma strings, ready to feed to `BM25Okapi(corpus)`.
    """
    tokens = tokenize(text)
    out: list[str] = []
    for tok in tokens:
        lemma = _lemmatize_one(tok)
        if drop_stopwords and lemma in RUSSIAN_STOPWORDS:
            continue
        # Drop pure punctuation/numbers without alphabetic content
        if not any(c.isalpha() for c in lemma):
            continue
        out.append(lemma)
    return out


def batch_lemmatize(
    texts: Iterable[str], *, drop_stopwords: bool = True
) -> list[list[str]]:
    return [lemmatize(t, drop_stopwords=drop_stopwords) for t in texts]


__all__ = [
    "RUSSIAN_STOPWORDS",
    "batch_lemmatize",
    "lemmatize",
    "tokenize",
]
