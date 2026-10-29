"""Prepare the v0 benchmark dataset from biobank .docx sources.

Produces:
  data/processed/biobankqa_v0_executable.jsonl  — single source of truth

Pipeline:
  1. Parse MCQ + open from sources (final_test_otta_2025-10-10.docx +
     open_situational_biobank.docx).
  2. For MCQ, extract red-marked correct option(s) from docx run color.
  3. Apply manual_overrides.json (e.g. mcq_046, which has no red marking).
  4. Shuffle MCQ options deterministically (seed=42) to mitigate position bias;
     keep both canonical and shuffled layouts + mapping.
  5. Write jsonl + markdown.

Run:
  python -m benchmark.scripts.extract_questions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import docx
import jsonlines


REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_ROOT = REPO_ROOT / "data" / "raw" / "questions"
# Search these subfolders for each named source file, first match wins.
SOURCE_SEARCH_DIRS = [
    QUESTIONS_ROOT / "sources",
    QUESTIONS_ROOT / "sources_additional",
]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUT_JSONL = PROCESSED_DIR / "biobankqa_v0_executable.jsonl"
OVERRIDES_PATH = PROCESSED_DIR / "manual_overrides.json"

# Shuffle salt for deterministic option re-ordering. Pin in code; do not load
# from CLI — changing it invalidates the model cache.
SHUFFLE_SALT = 42


def _find_source(fname: str) -> Path:
    for d in SOURCE_SEARCH_DIRS:
        p = d / fname
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{fname} not found in any of: {[str(d) for d in SOURCE_SEARCH_DIRS]}"
    )

# Source files in deterministic processing order.
# v0 scope (as agreed with reviewer): the school version of the Otta test is
# the canonical wording source — its small text edits (e.g. mcq_002 option B
# "В Китае", mcq_013 option C wider phrasing) take precedence over the older
# final test. We then append MCQ that exist ONLY in the final test (3 items)
# and, for overlapping MCQ where school lacks a gold marker but final has one,
# we rescue the gold marker from the final test (e.g. mcq_046 bold option D).
# The LLM-generated synthetic block and the older draft tests are intentionally
# excluded — they will be revisited in v0.1+.
MCQ_SOURCES = [
    "test_questions_school.docx",
    "final_test_otta_2025-10-10.docx",
]
OPEN_SOURCE = "open_situational_biobank.docx"


# --- text helpers ---------------------------------------------------------

def _clean(s: str) -> str:
    """Strip, collapse whitespace, normalize odd spaces."""
    s = s.replace("\xa0", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+)\s*[.\)]?\s+")


def _strip_leading_question_number(s: str) -> str:
    """Remove leading question numbering like '12.', '12)', '12 ', '12\t'."""
    return _LEADING_NUMBER_RE.sub("", s, count=1)


def _extract_leading_question_number(s: str) -> int | None:
    """Return the integer prefix if the paragraph starts with one, else None."""
    m = _LEADING_NUMBER_RE.match(s)
    return int(m.group(1)) if m else None


def _strip_leading_generated_code(s: str) -> str:
    """Remove leading 'В1.' or 'B1.' style codes from generated test."""
    return re.sub(r"^\s*[ВB]\s*\d+\s*[.\)]?\s+", "", s, count=1)


_NORMALIZE_PUNCT_RE = re.compile(r"[^\w\sа-яё]", flags=re.UNICODE | re.IGNORECASE)


def _normalize_question(text: str) -> str:
    """Normalize a question for fuzzy equality across school vs final.

    Lowercase, ё→е, strip leading numbering, strip non-word non-cyrillic
    punctuation, collapse whitespace, truncate to 120 chars.
    """
    if not text:
        return ""
    s = text.lower().replace("ё", "е")
    s = _strip_leading_question_number(s)
    s = _NORMALIZE_PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


# Map of letters we accept as option labels to their normalized Latin form.
# Latin: A B C D E. Russian capital that visually overlaps: А (=A), В (=B/V),
# С (=C/S), Е (=E). Russian positional: А Б В Г Д -> A B C D E.
# Because Russian capital А, В, С, Е visually overlap with Latin A, B, C, E,
# we resolve ambiguity contextually below.

LATIN_LABELS = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
# Look-alikes that overlap Latin alphabet, treat as Latin position by default.
# Cyrillic: А (U+0410) ~ A, В (U+0412) ~ B, С (U+0421) ~ C, Е (U+0415) ~ E.
LOOKALIKE_LABELS = {"А": "A", "В": "B", "С": "C", "Е": "E"}
# Russian positional labels distinct from Latin look-alikes: Б, Г, Д.
RUSSIAN_POS_DISTINCT = {"Б", "Г", "Д"}
# Mapping used when a block is detected as Russian positional (А Б В Г Д).
RUSSIAN_POS_MAP = {"А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E"}

# Regex pieces.
# All letters we accept as option labels: Latin A-E (both cases) plus the
# Cyrillic capitals used in the sources (А Б В Г Д Е С). Note: С (Cyrillic
# capital S) is U+0421 which is OUTSIDE the А-Г range, so it must be
# enumerated. Lowercase Latin is included because some sources use 'd.'.
ALL_LABEL_CHARS = "A-Ea-eА-ЕС"

OPTION_LINE_RE = re.compile(
    rf"^\s*([{ALL_LABEL_CHARS}])\s*[\.\)]?\s*(.*)$",
    flags=re.UNICODE,
)
ANSWER_LINE_RE = re.compile(
    rf"^\s*Ответ\s*[:\-]?\s*([{ALL_LABEL_CHARS}])\s*[\.\)]?\s*$",
    flags=re.UNICODE | re.IGNORECASE,
)
RATIONALE_LINE_RE = re.compile(
    r"^\s*(?:Обоснование|Пояснение)\s*[:\-]?\s*(.+)$",
    flags=re.UNICODE | re.IGNORECASE,
)
# Block header: "1. Основы биобанкирования" (no question mark, short).
BLOCK_HEADER_RE = re.compile(r"^\s*(\d+)\.\s+([^?]+?)\s*$", flags=re.UNICODE)


def _looks_like_option_line(text: str) -> bool:
    """A paragraph that begins with a single capital letter + delimiter + content."""
    if not text:
        return False
    m = re.match(
        rf"^\s*([{ALL_LABEL_CHARS}])\s*([\.\)]|\s)\s*\S",
        text,
        flags=re.UNICODE,
    )
    return bool(m)


def _normalize_label(letter: str, used_russian_positional: bool) -> str:
    """Normalize a single letter to Latin A/B/C/D/E.

    used_russian_positional: True if we've seen distinctly Russian letters
    (Б/Г/Д) earlier in the same option block, in which case all look-alikes
    should map positionally (А->A, В->C, ...).
    """
    letter = letter.upper()
    if letter in LATIN_LABELS:
        return LATIN_LABELS[letter]
    if used_russian_positional:
        if letter in RUSSIAN_POS_MAP:
            return RUSSIAN_POS_MAP[letter]
    if letter in LOOKALIKE_LABELS:
        return LOOKALIKE_LABELS[letter]
    if letter in RUSSIAN_POS_MAP:
        return RUSSIAN_POS_MAP[letter]
    return letter  # fallback


def _normalize_options(raw_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Take ordered (label, text) pairs and normalize labels to A/B/C/D/E.

    If any distinctly Russian letter (Б/Г/Д) appears we treat the whole block
    as Russian positional, otherwise we treat letters as Latin/look-alike.
    """
    russian = any(lbl in RUSSIAN_POS_DISTINCT for lbl, _ in raw_pairs)
    out: dict[str, str] = {}
    for lbl, txt in raw_pairs:
        norm = _normalize_label(lbl, russian)
        # Skip duplicate labels — keep the first one.
        if norm not in out:
            out[norm] = txt
    return out


# --- core parsers ---------------------------------------------------------

def _read_paragraphs(path: Path) -> list[str]:
    d = docx.Document(str(path))
    return [_clean(p.text) for p in d.paragraphs]


# --- formatting (gold-answer) parsing ------------------------------------
# An option paragraph is treated as "marked correct" if either:
#   (a) any of its runs is rendered in red (FF0000-ish), or
#   (b) any of its runs has bold=True with non-empty text.
# Question paragraphs are universally bold in this test, but option paragraphs
# are bold only when the author marked them as correct (verified empirically
# on final_test_otta_2025-10-10.docx — only mcq_046 option D is bold).

def _is_red_rgb(rgb_color) -> bool:
    if rgb_color is None:
        return False
    try:
        r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    except (TypeError, IndexError, ValueError):
        return False
    return r >= 180 and g <= 80 and b <= 80


def _paragraph_mark_kind(p) -> str | None:
    """Return "red" or "bold" if the paragraph carries that mark, else None."""
    has_bold = False
    for run in p.runs:
        try:
            rgb = run.font.color.rgb
        except Exception:
            rgb = None
        if _is_red_rgb(rgb):
            return "red"
        if run.font.bold and run.text.strip():
            has_bold = True
    return "bold" if has_bold else None


def _extract_marked_gold_letters(path: Path) -> list[dict]:
    """For each contiguous block of MCQ option paragraphs in the docx, return
    `{"marked_letters": [..normalized..], "kind": "single"|"multi"|"none",
      "mark": "red"|"bold"|"mixed"|None}`. One entry per block, in document order.
    """
    d = docx.Document(str(path))
    out: list[dict] = []
    block: list[tuple[str, str | None]] = []  # (raw_label, mark)

    def flush() -> None:
        if not block:
            return
        labels = [lbl for lbl, _ in block]
        russian = any(lbl in RUSSIAN_POS_DISTINCT for lbl in labels)
        marked = [(_normalize_label(lbl, russian), m) for lbl, m in block if m]
        letters = [l for l, _ in marked]
        mark_set = {m for _, m in marked}
        mark = next(iter(mark_set)) if len(mark_set) == 1 else ("mixed" if mark_set else None)
        if len(letters) == 1:
            out.append({"marked_letters": letters, "kind": "single", "mark": mark})
        elif not letters:
            out.append({"marked_letters": [], "kind": "none", "mark": None})
        else:
            out.append({"marked_letters": letters, "kind": "multi", "mark": mark})
        block.clear()

    for p in d.paragraphs:
        text = _clean(p.text)
        if not text:
            flush()
            continue
        if _looks_like_option_line(text):
            m = OPTION_LINE_RE.match(text)
            if m:
                block.append((m.group(1), _paragraph_mark_kind(p)))
                continue
        flush()
    flush()
    return out


def _parse_mcq_paragraphs(
    paragraphs: list[str],
    source_file: str,
    *,
    has_answers: bool = False,
    block_headers: bool = False,
) -> list[dict]:
    """Parse a list of paragraphs into MCQ records.

    Generic state machine:
      - Outside a question: a non-option, non-answer line that looks like a
        question starts a new MCQ.
      - Inside a question: option lines accumulate; an answer/rationale line
        terminates accumulation for generated content.
      - A new question line flushes the current one.
    """
    records: list[dict] = []
    current_topic: str | None = None

    cur_question: str | None = None
    cur_source_number: int | None = None
    cur_options: list[tuple[str, str]] = []
    cur_answer: str | None = None
    cur_rationale: str | None = None

    def flush() -> None:
        nonlocal cur_question, cur_source_number, cur_options, cur_answer, cur_rationale
        if cur_question is None:
            return
        opts = _normalize_options(cur_options) if cur_options else None
        gold = None
        if has_answers and cur_answer is not None:
            russian = any(lbl in RUSSIAN_POS_DISTINCT for lbl, _ in cur_options)
            gold = _normalize_label(cur_answer, russian)
        rec = {
            "_question": cur_question,
            "_source_number": cur_source_number,
            "_options": opts,
            "_gold": gold,
            "_rationale": cur_rationale,
            "_topic": current_topic,
            "_source_file": source_file,
        }
        records.append(rec)
        cur_question = None
        cur_source_number = None
        cur_options = []
        cur_answer = None
        cur_rationale = None

    for raw in paragraphs:
        if not raw:
            continue

        # Block header detection (for generated_test_blocks.docx).
        if block_headers:
            m = BLOCK_HEADER_RE.match(raw)
            if m and not _looks_like_option_line(raw) and "?" not in raw:
                # Make sure this isn't actually a numbered question.
                # Numbered questions usually contain "?" or end with ":".
                # Block headers are short topic phrases.
                topic_name = m.group(2).strip()
                # Reject obvious non-topics (too long, ends with punctuation
                # other than period).
                if len(topic_name) <= 80:
                    flush()
                    current_topic = topic_name
                    continue

        # Answer line (only for generated).
        if has_answers:
            am = ANSWER_LINE_RE.match(raw)
            if am:
                cur_answer = am.group(1)
                continue
            rm = RATIONALE_LINE_RE.match(raw)
            if rm:
                cur_rationale = rm.group(1).strip()
                continue

        # Option line: only if we have an active question.
        if cur_question is not None and _looks_like_option_line(raw):
            m = OPTION_LINE_RE.match(raw)
            if m:
                lbl = m.group(1)
                txt = m.group(2).strip()
                cur_options.append((lbl, txt))
                continue

        # Otherwise: this is (probably) a new question. Flush previous.
        flush()
        cur_source_number = _extract_leading_question_number(raw)
        q = raw
        q = _strip_leading_question_number(q)
        q = _strip_leading_generated_code(q)
        cur_question = q.strip()

    flush()

    # Convert internal records to final schema later (id assignment global).
    return records


def _parse_open_paragraphs(paragraphs: list[str], source_file: str) -> list[dict]:
    recs: list[dict] = []
    for raw in paragraphs:
        if not raw:
            continue
        n = _extract_leading_question_number(raw)
        q = _strip_leading_question_number(raw).strip()
        if not q:
            continue
        recs.append({
            "_question": q,
            "_source_number": n,
            "_options": None,
            "_gold": None,
            "_rationale": None,
            "_topic": None,
            "_source_file": source_file,
        })
    return recs


# --- assembly -------------------------------------------------------------

def _parse_with_marks(fname: str, warnings: list[str]) -> list[dict]:
    """Parse one MCQ docx and attach gold-letter info from in-doc marks.

    Each returned record is enriched with `_red_letters`, `_gold` (None if
    missing or multi-correct), and `_gold_source` ("red_marked", "bold_marked",
    "marked_mixed", "marked", "multiple_correct", or "missing"). Mismatched
    block counts are appended to `warnings` and gold fields are left empty.
    """
    paragraphs = _read_paragraphs(_find_source(fname))
    recs = _parse_mcq_paragraphs(
        paragraphs, fname, has_answers=False, block_headers=False
    )
    blocks = _extract_marked_gold_letters(_find_source(fname))
    if len(blocks) == len(recs):
        for rec, block in zip(recs, blocks):
            rec["_red_letters"] = block["marked_letters"]
            mark = block.get("mark")
            source_map = {"red": "red_marked", "bold": "bold_marked", "mixed": "marked_mixed"}
            if block["kind"] == "single":
                rec["_gold"] = block["marked_letters"][0]
                rec["_gold_source"] = source_map.get(mark, "marked")
            elif block["kind"] == "multi":
                rec["_gold"] = None
                rec["_gold_source"] = "multiple_correct"
            else:
                rec["_gold"] = None
                rec["_gold_source"] = "missing"
    else:
        warnings.append(
            f"{fname}: marked-block count ({len(blocks)}) != MCQ count "
            f"({len(recs)}); skipping format-based gold assignment"
        )
    return recs


def _build_final_items() -> tuple[list[dict], list[str]]:
    """Return (items_in_jsonl_schema, warnings).

    Merge strategy (per reviewer): take ALL MCQ from school as the canonical
    base (its small text edits win), then append only the MCQ that appear in
    the final test but NOT in school. For overlapping MCQ where school has no
    gold mark but final does, rescue the gold mark from final.
    """
    warnings: list[str] = []
    all_open_records: list[dict] = []

    # 1) MCQ sources. Each docx may mark the correct option by colouring its
    # paragraph red OR by making it bold black; we accept either.
    school_fname, final_fname = MCQ_SOURCES[0], MCQ_SOURCES[1]
    school_recs = _parse_with_marks(school_fname, warnings)
    final_recs = _parse_with_marks(final_fname, warnings)

    school_by_norm: dict[str, int] = {}
    for i, rec in enumerate(school_recs):
        key = _normalize_question(rec["_question"])
        if key and key not in school_by_norm:
            school_by_norm[key] = i

    all_mcq_records: list[dict] = list(school_recs)
    n_filled_from_final = 0
    n_new_from_final = 0
    for fr in final_recs:
        key = _normalize_question(fr["_question"])
        idx = school_by_norm.get(key) if key else None
        if idx is not None:
            sr = all_mcq_records[idx]
            if sr.get("_gold") is None and fr.get("_gold") is not None:
                sr["_gold"] = fr["_gold"]
                sr["_red_letters"] = fr.get("_red_letters", [])
                sr["_gold_source"] = f"final_{fr.get('_gold_source')}"
                n_filled_from_final += 1
        else:
            all_mcq_records.append(fr)
            n_new_from_final += 1

    warnings.append(
        f"merged: {len(school_recs)} from school + {n_new_from_final} new "
        f"from final = {len(all_mcq_records)} MCQ"
    )
    warnings.append(
        f"filled gold from final for {n_filled_from_final} overlapping MCQ"
    )

    # 2) Open situational.
    paragraphs = _read_paragraphs(_find_source(OPEN_SOURCE))
    all_open_records.extend(_parse_open_paragraphs(paragraphs, OPEN_SOURCE))

    # Assign IDs.
    items: list[dict] = []
    mcq_counter = 0
    open_counter = 0

    def make_item(rec: dict, qtype: str, qid: str) -> dict:
        return {
            "id": qid,
            "type": qtype,
            "source_file": rec["_source_file"],
            "source_question_number": rec.get("_source_number"),
            "topic": rec["_topic"],
            "question": rec["_question"],
            "options": rec["_options"],
            "gold_answer": rec["_gold"],
            "gold_letters": rec.get("_red_letters", []),
            "gold_source": rec.get("_gold_source"),
            "rationale": rec["_rationale"],
        }

    for rec in all_mcq_records:
        mcq_counter += 1
        qid = f"mcq_{mcq_counter:03d}"
        item = make_item(rec, "mcq", qid)
        items.append(item)
        if item["options"] is None or len(item["options"]) < 3:
            warnings.append(
                f"{qid} ({item['source_file']}): only "
                f"{0 if item['options'] is None else len(item['options'])} "
                f"option(s) detected — Q: {item['question'][:80]!r}"
            )
        elif item["options"] and len(set(item["options"].values())) < len(item["options"]):
            warnings.append(
                f"{qid} ({item['source_file']}): duplicate option text "
                f"(source typo) — options={item['options']}"
            )
        if item["gold_answer"] is None:
            extra = (
                f" multi-correct={item['gold_letters']}"
                if item.get("gold_source") == "multiple_correct"
                else ""
            )
            warnings.append(
                f"{qid} ({item['source_file']}): no single-letter gold "
                f"({item.get('gold_source')}){extra} — "
                f"Q: {item['question'][:80]!r}"
            )

    for rec in all_open_records:
        open_counter += 1
        qid = f"open_{open_counter:03d}"
        items.append(make_item(rec, "open", qid))

    return items, warnings


# (markdown rendering removed — jsonl is the single source of truth)


# --- override + shuffle ---------------------------------------------------

def _apply_overrides(items: list[dict], overrides: dict[str, dict]) -> int:
    """Mutate items in place applying manual overrides (`gold_answer`,
    `gold_letters`, `gold_source`, etc). The `note` key is metadata only.
    Returns number of items touched."""
    n = 0
    for it in items:
        ov = overrides.get(it["id"])
        if not ov:
            continue
        for k, v in ov.items():
            if k == "note":
                continue
            it[k] = v
        n += 1
    return n


def _stable_seed(item_id: str, salt: int) -> int:
    h = hashlib.sha256(f"{item_id}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _shuffle_mcq(item: dict, salt: int) -> dict:
    """Return a new item with options reshuffled. Preserves the canonical
    layout and gold; adds shuffled layout + bidirectional shuffle maps so the
    scorer can decode the model's letter back to the canonical letter space.
    """
    options = item.get("options") or {}
    canonical_letters = sorted(options.keys())
    rng = random.Random(_stable_seed(item["id"], salt))
    order = canonical_letters[:]
    rng.shuffle(order)
    shown_to_canonical = {shown: order[i] for i, shown in enumerate(canonical_letters)}
    canonical_to_shown = {v: k for k, v in shown_to_canonical.items()}

    options_shuffled = {shown: options[canon] for shown, canon in shown_to_canonical.items()}
    gold = item.get("gold_answer")
    gold_letters = item.get("gold_letters") or []

    new = dict(item)
    new["options_canonical"] = options
    new["options_shuffled"] = options_shuffled
    new["shuffle_map_to_canonical"] = shown_to_canonical
    new["shuffle_map_to_shown"] = canonical_to_shown
    new["gold_answer_canonical"] = gold
    new["gold_answer_shuffled"] = canonical_to_shown.get(gold) if gold else None
    new["gold_letters_canonical"] = gold_letters
    new["gold_letters_shuffled"] = [canonical_to_shown[g] for g in gold_letters if g in canonical_to_shown]
    return new


# --- main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Skip MCQ option shuffling (debug only — keeps canonical order)",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    items, warnings = _build_final_items()
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    overrides: dict[str, dict] = {}
    if OVERRIDES_PATH.exists():
        overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    n_overridden = _apply_overrides(items, overrides)

    out_items: list[dict] = []
    n_shuffled = 0
    for it in items:
        if it["type"] == "mcq" and it.get("options") and not args.no_shuffle:
            out_items.append(_shuffle_mcq(it, SHUFFLE_SALT))
            n_shuffled += 1
        else:
            out_items.append(it)

    with jsonlines.open(OUT_JSONL, mode="w") as writer:
        for it in out_items:
            writer.write(it)

    n_mcq = sum(1 for i in out_items if i["type"] == "mcq")
    n_open = sum(1 for i in out_items if i["type"] == "open")
    n_single = sum(1 for i in out_items if i["type"] == "mcq" and i.get("gold_answer"))
    n_multi = sum(1 for i in out_items if i.get("gold_source") == "multiple_correct")
    n_missing = sum(1 for i in out_items if i.get("gold_source") == "missing")

    print(file=sys.stderr)
    print(f"WROTE {OUT_JSONL.name}: {len(out_items)} items "
          f"({n_mcq} MCQ, {n_open} open) — shuffled={n_shuffled}, overrides={n_overridden}", file=sys.stderr)
    print(f"Gold breakdown: {n_single} single  |  {n_multi} multi-correct  |  {n_missing} missing", file=sys.stderr)
    print(f"Accuracy-evaluable MCQ: {n_single + n_multi} of {n_mcq}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
