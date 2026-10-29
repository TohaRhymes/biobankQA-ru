"""V2 materials parser — column-aware extraction via `unstructured`.

Why a rewrite (vs `parse_materials.py`):
  v1 used `pdfplumber.extract_text()`, which reads multi-column PDFs
  line-by-line across columns and glues left-column line N to right-column
  line N. For `cardiovascular_*.pdf` and the Meshkov monograph the result was
  jumbled Russian; running journal/booklet titles also leaked into every
  chunk as section labels and as text prefixes (~672/1989 chunks).

This rewrite:
  - Uses `unstructured.partition.pdf(strategy="fast")`, which clusters PDF
    tokens into columns by x-coordinate and emits typed elements
    (NarrativeText / ListItem / Title / Header / Footer / ...).
  - Strips elements categorized as Header/Footer from main text.
  - Additionally strips any short string that appears verbatim at the top or
    bottom of >=30 % of a source's pages (catches running titles that
    `unstructured` didn't classify as Header).
  - Detects journal article boundaries via the canonical first-page header
    `... doi: 10.15829/...` AND the article title (the NarrativeText element
    that follows the header on the same page).
  - Populates `section` only when a heading is detected with confidence:
        * `unstructured` Title element with low element count on the page; OR
        * line matching "Раздел / Глава / Часть / SECTION X / [A-Z]\\d\\." etc.
    Otherwise null (target null rate 30-50 %).
  - References-list detection unchanged from v1 logic but applied per chunk.
  - Chunking: 512-token target, 64-token overlap, never breaks a paragraph
    (same as v1, slightly different overlap).
  - DOC handling unchanged (textutil to UTF-8).

Run:
  python -m benchmark.scripts.parse_materials_v2
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from statistics import median
from datetime import date
from pathlib import Path
from typing import Any, Iterable

# --- silence the very noisy pdfminer color-warnings before importing unstructured
os.environ.setdefault("PDFMINER_LOG_LEVEL", "ERROR")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)
logging.getLogger("pdfminer.cmapdb").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfdocument").setLevel(logging.ERROR)
logging.getLogger("pdfminer.layout").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfpage").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from unstructured.partition.pdf import partition_pdf  # noqa: E402
import pdfplumber  # noqa: E402


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "data" / "raw" / "materials" / "sources"
OUT_DIR = REPO_ROOT / "data" / "processed" / "materials"

CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
MIN_CHUNK_CHARS = 30

# A string that appears on >= this share of pages of a source (in top or
# bottom region) is considered a running header/footer and stripped.
RUNNING_HEADER_PAGE_RATIO = 0.30

# Each source: (filename, short_slug, expected_lang, kind)
SOURCES: list[tuple[str, str, str, str]] = [
    ("ISBER-Best-Practice-2023-5th-edition.pdf", "isber-en-5ed", "en", "standard"),
    ("isber_bp_4ed_russian.pdf", "isber-ru-4ed", "ru", "standard"),
    ("программа по биобанкированию_ред 09.07.2025.doc", "program-ru-2025", "ru", "program"),
    ("25-07-Meshkov.pdf", "meshkov-monograph-2022", "ru", "monograph"),
    ("cardiovascular_2023.pdf", "cardiovasc-2023", "ru", "journal"),
    ("cardiovascular_2024.pdf", "cardiovasc-2024", "ru", "journal"),
    ("cardiovascular_2025.pdf", "cardiovasc-2025", "ru", "journal"),
]


# -----------------------------------------------------------------------------
# Lightweight text helpers
# -----------------------------------------------------------------------------

_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-z]")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
_WS = re.compile(r"[ \t]+")
_DOI_JOURNAL = re.compile(r"doi:\s*10\.15829/", re.IGNORECASE)
_DOI_GENERIC = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_ISSN = re.compile(r"ISSN\s*1728-8800", re.IGNORECASE)
_BARE_PAGENUM_TAIL = re.compile(r"\s\d{1,4}\s*$")
_BARE_PAGENUM_LEAD = re.compile(r"^\s*\d{1,4}\s+")

# Headings recognised as confident section markers.
_SECTION_PATTERNS = [
    re.compile(r"^\s*(?:Раздел|Глава|Часть|Приложение|Введение|Заключение|Список\s+литературы)\b.{0,80}$"),
    re.compile(r"^\s*SECTION\s+[A-Z][A-Z0-9\-:.\s,]{0,80}$"),
    re.compile(r"^\s*[A-Z]\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[A-ZА-ЯЁ][^.!?]{2,80}$"),
    re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[A-ZА-ЯЁ][^.!?]{2,80}$"),
]


def detect_lang(s: str) -> str:
    cyr = len(_CYR.findall(s))
    lat = len(_LAT.findall(s))
    if cyr == 0 and lat == 0:
        return "ru"
    return "ru" if cyr >= lat else "en"


def est_tokens(text: str) -> int:
    return int(round(len(text.split()) * 1.3))


def _dedup_doubled_chars(s: str) -> str:
    """Collapse "CCaarrddiioo" -> "Cardio" style artefacts."""
    def fix(tok: str) -> str:
        if len(tok) < 6 or len(tok) % 2:
            return tok
        if not tok.isalpha():
            return tok
        if all(tok[i] == tok[i + 1] for i in range(0, len(tok), 2)):
            return tok[::2]
        return tok
    return " ".join(fix(t) for t in s.split(" "))


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _dedup_doubled_chars(s)
    s = s.replace("­", "").replace("​", "").replace(" ", " ")
    # Soft-wrap hyphenation: "слово-\nслово" -> "словослово"
    s = re.sub(r"(\w)[-‐]\s+(\w)", r"\1\2", s)
    lines = [_WS.sub(" ", ln).strip() for ln in s.splitlines()]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def is_garbage_chunk(text: str) -> bool:
    if len(text) < MIN_CHUNK_CHARS:
        return True
    alpha = len(_ALPHA.findall(text))
    nonspace = sum(1 for c in text if not c.isspace())
    if nonspace == 0:
        return True
    if alpha / nonspace < 0.30:
        return True
    # Rotated / reversed-character table OCR comes through as long sequences of
    # 1-2-char tokens ("2 а ц и л б а Т" = "Таблица 2" reversed). Real Russian
    # / English text averages 4-7 chars per word. Use two conditions so we drop
    # the OCR junk but keep legitimate dense-formula / TOC chunks: average word
    # length < 2.0 OR (avg < 2.5 AND single-char tokens dominate >60% of words).
    words = text.split()
    if len(words) >= 30:
        avg_word_len = sum(len(w) for w in words) / len(words)
        single_char_share = sum(1 for w in words if len(w) == 1) / len(words)
        if avg_word_len < 2.0:
            return True
        if avg_word_len < 2.5 and single_char_share > 0.60:
            return True
    return False


_CITATION_PAT = re.compile(r"\b\d{4};\s*\d+\s*\(\d+\)\s*:\s*\d+")


def looks_like_references(text: str) -> bool:
    if _DOI_GENERIC.search(text) is None and "doi:" not in text.lower():
        return False
    lines = text.splitlines()
    numbered = sum(1 for ln in lines if re.match(r"^\s*(?:\[\d+\]|\d+\.)\s+\S", ln))
    if numbered >= 3 and numbered / max(1, len(lines)) > 0.10:
        return True
    # Fallback: many DOIs or many "Year;Vol(N):page" citation patterns in a
    # single chunk also signals a references block (catches references that
    # arrive as one NarrativeText element with no per-line numbering).
    doi_hits = len(_DOI_GENERIC.findall(text))
    cite_hits = len(_CITATION_PAT.findall(text))
    return doi_hits >= 3 or cite_hits >= 3


# -----------------------------------------------------------------------------
# Element model
# -----------------------------------------------------------------------------

@dataclass
class Elem:
    page: int
    category: str          # e.g. NarrativeText / Title / ListItem / Header / Footer
    text: str
    top: float | None      # y-coord of top (lower = higher on page; PDF origin is bottom)
    bottom: float | None
    y_norm_top: float | None    # 0 = page top, 1 = page bottom
    y_norm_bottom: float | None
    section_hint: str | None = None  # typography evidence; never injected into body text


# -----------------------------------------------------------------------------
# Per-source extraction via unstructured
# -----------------------------------------------------------------------------

def _y_coords(coord_dict: dict | None, page_height: float | None) -> tuple[float | None, float | None]:
    """Return (top, bottom) y-coordinates in the same coordinate system unstructured emits.

    `coord_dict` looks like: {"points": [[x,y],[x,y],[x,y],[x,y]], "system": "PixelSpace", ...}
    With "PixelSpace" / coords coming from pdfminer, smaller y is the page top.
    """
    if not coord_dict:
        return None, None
    pts = coord_dict.get("points") or []
    if not pts:
        return None, None
    ys = [float(p[1]) for p in pts]
    return min(ys), max(ys)


def _cluster_x_into_columns(x0s: list[float], page_width: float) -> list[float]:
    """Return ascending list of column-boundary x-cutoffs.

    Histogram x0s in 20-wide bins; find peaks above 10% of max. Returns the
    x boundaries between detected peaks. If 1 peak -> []; 2 peaks -> [midpoint].
    """
    if not x0s:
        return []
    from collections import Counter
    binsz = 20
    hist = Counter(int(x // binsz) * binsz for x in x0s)
    if not hist:
        return []
    max_count = max(hist.values())
    threshold = max(2, int(0.15 * max_count))
    # Peaks: bins above threshold, taking the local max in each contiguous run
    bins_sorted = sorted(hist)
    peaks: list[int] = []
    run: list[int] = []
    for b in bins_sorted:
        if hist[b] >= threshold:
            if run and (b - run[-1] <= binsz):
                run.append(b)
            else:
                if run:
                    # take bin in run with max count
                    peaks.append(max(run, key=lambda x: hist[x]))
                run = [b]
        else:
            if run:
                peaks.append(max(run, key=lambda x: hist[x]))
                run = []
    if run:
        peaks.append(max(run, key=lambda x: hist[x]))

    # Only keep peaks that are well-separated (> page_width/4 apart).
    peaks.sort()
    keep: list[int] = []
    for p in peaks:
        if not keep or p - keep[-1] > page_width / 4:
            keep.append(p)
    if len(keep) < 2:
        return []
    # Boundaries are midpoints between successive peaks.
    return [(keep[i] + keep[i + 1]) / 2 for i in range(len(keep) - 1)]


def _column_idx(x0: float, boundaries: list[float]) -> int:
    for i, b in enumerate(boundaries):
        if x0 < b:
            return i
    return len(boundaries)


def _words_to_lines(words: list[dict], y_tol: float = 3.0) -> list[dict]:
    """Group words into lines by similar `top`. Returns list of dicts:
    {"top": float, "bottom": float, "text": str, "x0": float}.
    """
    if not words:
        return []
    sw = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[dict] = []
    cur: list[dict] = [sw[0]]
    for w in sw[1:]:
        if abs(w["top"] - cur[-1]["top"]) <= y_tol:
            cur.append(w)
        else:
            cur.sort(key=lambda x: x["x0"])
            lines.append({
                "top": min(x["top"] for x in cur),
                "bottom": max(x["bottom"] for x in cur),
                "x0": min(x["x0"] for x in cur),
                "x1": max(x.get("x1", x["x0"]) for x in cur),
                "text": " ".join(x["text"] for x in cur),
            })
            cur = [w]
    if cur:
        cur.sort(key=lambda x: x["x0"])
        lines.append({
            "top": min(x["top"] for x in cur),
            "bottom": max(x["bottom"] for x in cur),
            "x0": min(x["x0"] for x in cur),
            "x1": max(x.get("x1", x["x0"]) for x in cur),
            "text": " ".join(x["text"] for x in cur),
        })
    return lines


def _lines_to_paragraphs(lines: list[dict]) -> list[str]:
    """Join consecutive lines into paragraphs, breaking on big gaps."""
    if not lines:
        return []
    # Median line height as the unit gap.
    heights = sorted(l["bottom"] - l["top"] for l in lines)
    h = heights[len(heights) // 2] if heights else 10.0
    paras: list[list[str]] = [[lines[0]["text"]]]
    for prev, cur in zip(lines, lines[1:]):
        gap = cur["top"] - prev["bottom"]
        if gap > h * 0.9:
            paras.append([cur["text"]])
        else:
            # Soft-wrap: join with space; if prev ends with hyphen, glue.
            ptxt = paras[-1]
            last = ptxt[-1]
            if last.endswith("-"):
                ptxt[-1] = last[:-1] + cur["text"].lstrip()
            else:
                ptxt.append(cur["text"])
    return ["\n".join(p) for p in paras]


def _is_typographic_heading(text: str, size: float, body_size: float, bold_share: float) -> bool:
    """Conservative short standalone heading, backed by PDF font evidence."""
    text = clean_text(text).strip()
    if not 4 <= len(text) <= 120 or not text[:1].isupper():
        return False
    if re.match(r"^(?:Что |What |Как |How )", text) or text == "Литература/References":
        return False
    if re.search(r"\S.+\sОбсуждение$", text) and text != "Результаты и обсуждение":
        return False
    if re.search(r"\b(?:[A-Z]{3,}\s+\d\w*|GG\s+GG)\b", text):
        return False
    if text.lower() in {"ключевые моменты", "key messages", "ключевые моменты key messages", "оригинальные статьи", "оригинальн ые статьи", "original articles", "обзор литературы", "literature review", "передовая статья", "кардиоваскулярная", "содержание", "contents"}:
        return False
    if text.split()[-1].lower() in {"для", "и", "в", "на", "по", "с", "из", "of", "for", "and", "in", "the", "with", "by"}:
        return False
    if text.endswith((".", ",", ";", ":", "—", "-")) or "@" in text:
        return False
    if re.search(r"\.{4,}\s*\d+\s*$", text):
        return False
    if re.match(r"^(?:Таблица|Рис(?:унок|\.)?|Figure|Table)\s+\d", text, re.I):
        return False
    if any(t in text.lower() for t in ("кардиоваскулярная терапия", "issn", "doi:", "10.15829", "isber head office")):
        return False
    if re.search(r"[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.", text):
        return False
    words = text.split()
    if not 1 <= len(words) <= 14:
        return False
    return body_size > 0 and size >= .98 * body_size and bold_share >= .8


def _typographic_heading_lines(page: Any, lines: list[dict]) -> list[tuple[str, float]]:
    chars = [c for c in page.chars if c.get("text", "").strip() and c.get("size", 0) > 0]
    if not chars:
        return []
    body_size = median(c["size"] for c in chars)
    result = []
    for line in lines:
        # Match only characters belonging to this column's line.  Bounding-box
        # evidence avoids changing extract_words tokenization or paragraph text.
        right = line.get("x1", float(page.width))
        matched = [c for c in chars if abs(float(c["top"]) - line["top"]) <= 3 and c["x0"] >= line["x0"] - 1 and c["x1"] <= right + 1]
        if not matched:
            continue
        # A column boundary may cut a printed line in the middle of a heading
        # or bold lead-in. Do not promote such fragments to persistent labels.
        if any(abs(float(c["top"]) - line["top"]) <= 3 and 0 <= c["x0"] - right <= max(8, body_size) for c in chars):
            continue
        size = median(c["size"] for c in matched)
        bold = sum("bold" in c.get("fontname", "").lower() for c in matched) / len(matched)
        if _is_typographic_heading(line["text"], size, body_size, bold):
            result.append((clean_text(line["text"]), line["top"] / float(page.height)))
    return result


def extract_pdf_pdfplumber(path: Path) -> list[Elem]:
    """Column-aware pdfplumber extraction fallback (when unstructured fails).

    For each page:
      - extract words
      - detect column boundaries from word-x histogram
      - bucket words by column, then by line, then concat top-to-bottom
        within each column, then concat columns left-to-right
      - emit each paragraph as a NarrativeText Elem with normalised y-coords.
    """
    out: list[Elem] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_height = float(page.height)
            page_width = float(page.width)
            try:
                words = page.extract_words(use_text_flow=False) or []
            except Exception:
                words = []
            if not words:
                continue
            x0s = [float(w["x0"]) for w in words]
            boundaries = _cluster_x_into_columns(x0s, page_width)
            # Bucket words by column.
            col_words: dict[int, list[dict]] = defaultdict(list)
            for w in words:
                col_words[_column_idx(float(w["x0"]), boundaries)].append({
                    "top": float(w["top"]),
                    "bottom": float(w["bottom"]),
                    "x0": float(w["x0"]),
                    "text": w["text"],
                    "x1": float(w["x1"]),
                })
            for col_idx in sorted(col_words):
                cws = col_words[col_idx]
                lines = _words_to_lines(cws)
                heading_lines = _typographic_heading_lines(page, lines)
                paragraphs = _lines_to_paragraphs(lines)
                for para_text in paragraphs:
                    if not para_text.strip():
                        continue
                    cleaned = clean_text(para_text)
                    if not cleaned:
                        continue
                    top = min(l["top"] for l in lines) if lines else 0.0
                    bot = max(l["bottom"] for l in lines) if lines else page_height
                    # Use the first line of this paragraph for y-norm to keep
                    # header/footer detection working.
                    # Approximate top of this paragraph from its first line.
                    # We don't actually have per-paragraph line list here; use
                    # a conservative midpoint of the column.
                    y_top = top / page_height
                    y_bot = bot / page_height
                    hints = [(h, y) for h, y in heading_lines if " ".join(h.split()) in " ".join(cleaned.split())]
                    hint = hints[0][0] if hints else None
                    out.append(
                        Elem(
                            page=i,
                            category="NarrativeText",
                            text=cleaned,
                            top=top,
                            bottom=bot,
                            y_norm_top=y_top,
                            y_norm_bottom=y_bot,
                            section_hint=hint,
                        )
                    )
    # Re-derive per-paragraph y_norm using actual paragraph top/bottom would
    # need re-emitting; for header/footer detection we instead match by text.
    return out


def extract_pdf(path: Path) -> list[Elem]:
    """Run unstructured on a PDF; fall back to pdfplumber column-aware if empty."""
    try:
        raw = partition_pdf(str(path), strategy="fast")
    except Exception as e:
        print(f"  unstructured failed ({e!s}), falling back to pdfplumber", file=sys.stderr)
        raw = []
    # Fallback triggers: empty element list OR near-empty total text (covers
    # cases where unstructured returns stub elements with effectively no text,
    # e.g. cardiovascular_2025.pdf).
    total_chars = sum(len(str(el).strip()) for el in raw)
    if not raw:
        print(f"  unstructured returned 0 elements, falling back to pdfplumber (column-aware)", file=sys.stderr)
        return extract_pdf_pdfplumber(path)
    if total_chars < 500:
        print(
            f"  unstructured returned {len(raw)} elements but only {total_chars} chars of text,"
            f" falling back to pdfplumber (column-aware)",
            file=sys.stderr,
        )
        return extract_pdf_pdfplumber(path)
    out: list[Elem] = []
    # Compute per-page max-y so we can normalise top/bottom (0 = top of page).
    by_page_max: dict[int, float] = {}
    by_page_min: dict[int, float] = {}
    tmp: list[tuple[int, str, str, dict | None]] = []
    for el in raw:
        try:
            md = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}
        except Exception:
            md = {}
        page = md.get("page_number") or 1
        coords = md.get("coordinates")
        txt = str(el).strip()
        if not txt:
            continue
        tmp.append((int(page), el.category, txt, coords))
        if coords:
            top, bot = _y_coords(coords, None)
            if top is not None and bot is not None:
                by_page_max[page] = max(by_page_max.get(page, bot), bot)
                by_page_min[page] = min(by_page_min.get(page, top), top)
    for page, category, txt, coords in tmp:
        top, bot = _y_coords(coords, None)
        ymax = by_page_max.get(page)
        ymin = by_page_min.get(page)
        if ymax is not None and ymin is not None and ymax > ymin:
            span = ymax - ymin
            y_top = (top - ymin) / span if top is not None else None
            y_bot = (bot - ymin) / span if bot is not None else None
        else:
            y_top, y_bot = None, None
        out.append(
            Elem(
                page=page,
                category=category,
                text=clean_text(txt),
                top=top,
                bottom=bot,
                y_norm_top=y_top,
                y_norm_bottom=y_bot,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Running header/footer detection
# -----------------------------------------------------------------------------

_NUM_ONLY = re.compile(r"^\s*\d{1,4}\s*$")


def find_running_headers_footers(elems: list[Elem]) -> tuple[set[str], set[re.Pattern]]:
    """Identify strings (or patterns) appearing on >= RATIO of pages.

    Returns:
        exact: literal strings to drop entirely (case-sensitive after clean_text).
        patterns: regex patterns to drop (used for prefixes like
                  "Кардиоваскулярная терапия и профилактика. 2024;23(...): ...").
    """
    pages = sorted({e.page for e in elems})
    if not pages:
        return set(), set()

    # 1) `unstructured`-classified Header/Footer: dedupe by literal text.
    h_counts: Counter[str] = Counter()
    for e in elems:
        if e.category in ("Header", "Footer"):
            h_counts[e.text] += 1

    # 2) Top-most and bottom-most line on each page (rough header/footer band).
    by_page: dict[int, list[Elem]] = defaultdict(list)
    for e in elems:
        by_page[e.page].append(e)

    band_top: Counter[str] = Counter()
    band_bot: Counter[str] = Counter()
    for p, els in by_page.items():
        with_y = [e for e in els if e.y_norm_top is not None]
        if not with_y:
            continue
        # top band: y_norm_top < 0.10
        top = [e for e in with_y if (e.y_norm_top or 1.0) < 0.10]
        bot = [e for e in with_y if (e.y_norm_bottom or 0.0) > 0.92]
        for e in top:
            band_top[e.text] += 1
        for e in bot:
            band_bot[e.text] += 1

    threshold = max(2, int(RUNNING_HEADER_PAGE_RATIO * len(pages)))

    exact: set[str] = set()
    for text, n in h_counts.items():
        # Any literal Header/Footer the library identified at least twice
        # is a running element; drop verbatim.
        if n >= 2:
            exact.add(text)
    for text, n in band_top.items():
        if n >= threshold:
            exact.add(text)
    for text, n in band_bot.items():
        if n >= threshold:
            exact.add(text)

    # 3) Patterns: drop bare page numbers and journal running title prefixes.
    patterns: set[re.Pattern] = set()
    patterns.add(re.compile(r"^\s*\d{1,4}\s*$"))
    # Truncated banner without DOI (DA #52 R3: cardiovasc-2025 p16-c1 etc.).
    patterns.add(
        re.compile(
            r"^\s*Кардиоваскулярная\s+терапия\s+и\s+профилактика\.?\s*\d{4};\d+(?:\([0-9]+\))?\.?\s*$",
            re.IGNORECASE,
        )
    )
    patterns.add(
        re.compile(
            r"^\s*Кардиоваскулярная\s+терапия\s+и\s+профилактика\.?\s*\d{4};\d+\([0-9]+\):\d+\.?\s*doi:.*$",
            re.IGNORECASE,
        )
    )
    # ISBER running footer banner (long string, vary slightly in whitespace).
    patterns.add(
        re.compile(
            r"^ISBER\s+Head\s+Office:.*Contact\s+Information:.*$",
            re.IGNORECASE | re.DOTALL,
        )
    )
    # Page-number lines like "Page 12" or "- 12 -"
    patterns.add(re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$"))
    return exact, patterns


def strip_running(elem: Elem, exact: set[str], patterns: list[re.Pattern]) -> str | None:
    t = elem.text
    if not t:
        return None
    if elem.category in ("Header", "Footer", "PageBreak"):
        # Always strip categorised header/footer.
        return None
    if t in exact:
        return None
    for p in patterns:
        if p.match(t):
            return None
    return t


# -----------------------------------------------------------------------------
# Section detection (per page, with confidence)
# -----------------------------------------------------------------------------

def detect_section_for_page(page_elems: list[Elem]) -> str | None:
    """Find a confident section heading among Title elements in the top half.

    Only returns a heading if it matches one of the strong patterns or is a
    short Title element clearly above the body text.
    """
    cands: list[Elem] = []
    for e in page_elems:
        if e.section_hint is not None:
            e = replace(e, category="Title", text=e.section_hint)
        if e.category != "Title":
            continue
        t = e.text.strip() if e.text else ""
        if not t or len(t) < 4 or len(t) > 120:
            continue
        low = t.lower()
        if any(
            tok in low
            for tok in (
                "кардиоваскулярная терапия",
                "issn",
                "doi:",
                "10.15829",
                "isber head office",
            )
        ):
            continue
        # Reject table/figure captions — they are page-local labels, not
        # section headings, and would corrupt carry-forward (DA #51 SS2).
        if re.match(r"^\s*(?:Таблица|Рис(?:унок|\.)?|Figure|Table)\s+\d", t, re.IGNORECASE):
            continue
        # Reject TOC dot-leader lines like "SECTION A: ... ......10" — these
        # look like headings but they're index entries that would be carried
        # forward into real chapter content (DA #52 R1).
        if re.search(r"\.{4,}\s*\d+\s*$", t):
            continue
        # Reject fragments: must start with capital, must not end with heavy
        # punctuation unless it matches a strong pattern.
        starts_ok = t[:1].isupper() or t[:1].isdigit() or t[:1] in "«\""
        if not starts_ok:
            continue
        if t.endswith((".", ",", ";", ":", "—", "-")):
            if not any(p.match(t) for p in _SECTION_PATTERNS):
                continue
        # Reject author-list-ish fragments.
        if t.count(",") >= 3 and re.search(r"[A-ZА-ЯЁ]\.", t):
            continue
        # Reject too-short headings (1-2 alpha words like "А g I" from rotated
        # OCR or initials).
        alpha_words = [w for w in t.split() if re.search(r"[A-Za-zА-ЯЁа-яё]{2,}", w)]
        if len(alpha_words) < 3:
            # Allow only if it matches a strong section pattern (e.g. "9.1 X").
            if not any(p.match(t) for p in _SECTION_PATTERNS):
                continue
        cands.append(e)

    # Conventional journal section labels are often short and mid-page.
    for original in page_elems:
        if original.section_hint and re.fullmatch(r"(?:Введение|Материал(?:ы)? и методы|Результаты(?: и обсуждение)?|Обсуждение|Заключение|Выводы|Литература/References)", original.section_hint):
            return original.section_hint

    # Strong patterns first.
    for e in cands:
        if any(p.match(e.text) for p in _SECTION_PATTERNS):
            return e.text.strip()

    # Top-of-page Title with sensible word count.
    for e in cands:
        if e.y_norm_top is not None and e.y_norm_top < 0.30:
            words = e.text.split()
            if 2 <= len(words) <= 14:
                return e.text.strip()

    return None


# -----------------------------------------------------------------------------
# Chunking (paragraph-level)
# -----------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-ZА-ЯЁ0-9«\"„])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split_long_sentence(sent: str) -> list[str]:
    """Split a single oversize sentence into target-sized windows on whitespace.

    Used when a sentence's estimated tokens exceed CHUNK_TARGET_TOKENS — e.g.
    table-OCR junk, rotated text, or genuinely unbroken long lines. Prevents
    chunk_paragraphs from emitting multi-thousand-token chunks.
    """
    words = sent.split()
    if not words:
        return []
    # est_tokens(text) = round(len(text.split()) * 1.3), so the per-window word
    # cap is target / 1.3.
    words_per_window = max(1, int(CHUNK_TARGET_TOKENS / 1.3))
    out: list[str] = []
    for i in range(0, len(words), words_per_window):
        out.append(" ".join(words[i:i + words_per_window]))
    return out


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    carry: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_tokens, carry
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(text)
        sents = _split_sentences(text)
        carry = []
        ctok = 0
        # Walk sentences right-to-left, accumulating overlap up to budget.
        # Two cases:
        #   - Normal text with sentence punctuation: carry the last 1-N
        #     sentences up to CHUNK_OVERLAP_TOKENS, but always carry at least
        #     one (R5 fix for zero-overlap on long-sentence flush).
        #   - Degenerate single-"sentence" text (no terminal punctuation:
        #     table cells, abbrev lists, glossaries): _split_sentences returns
        #     the whole chunk as one sentence. Carrying it would re-feed the
        #     entire chunk and cause unbounded accumulation across flushes
        #     (DA #51 SS1). In that case, fall back to a char-window of
        #     ~CHUNK_OVERLAP_TOKENS worth of trailing text.
        if len(sents) <= 1 and est_tokens(text) > 2 * CHUNK_OVERLAP_TOKENS:
            # Tail-char overlap: take the last ~CHUNK_OVERLAP_TOKENS*1.3 words
            # (NOT the whole text). Keeps some token continuity without
            # blowing up.
            words = text.split()
            tail_words = max(1, int(CHUNK_OVERLAP_TOKENS / 1.3))
            tail = " ".join(words[-tail_words:])
            carry = [tail]
            ctok = est_tokens(tail)
        else:
            for s in reversed(sents):
                t = est_tokens(s)
                if carry and ctok + t > CHUNK_OVERLAP_TOKENS:
                    break
                carry.insert(0, s)
                ctok += t
        buf = []
        buf_tokens = 0
        if carry:
            j = " ".join(carry)
            buf.append(j)
            buf_tokens = est_tokens(j)

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        ptok = est_tokens(p)
        if ptok > CHUNK_TARGET_TOKENS:
            flush()
            for sent in (_split_sentences(p) or [p]):
                stok = est_tokens(sent)
                # S2: if even a single sentence exceeds the target, hard-split
                # it on whitespace into target-sized windows before appending,
                # so the buffer cannot grow unboundedly large.
                if stok > CHUNK_TARGET_TOKENS:
                    for piece in _hard_split_long_sentence(sent):
                        ptok2 = est_tokens(piece)
                        if buf_tokens + ptok2 > CHUNK_TARGET_TOKENS and buf:
                            flush()
                        buf.append(piece)
                        buf_tokens += ptok2
                    continue
                if buf_tokens + stok > CHUNK_TARGET_TOKENS and buf:
                    flush()
                buf.append(sent)
                buf_tokens += stok
            continue
        if buf_tokens + ptok > CHUNK_TARGET_TOKENS and buf:
            flush()
        buf.append(p)
        buf_tokens += ptok

    if buf:
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(text)
    return chunks


# -----------------------------------------------------------------------------
# DOC fallback (textutil)
# -----------------------------------------------------------------------------

def extract_doc(path: Path) -> list[Elem]:
    out = subprocess.run(
        ["textutil", "-convert", "txt", "-encoding", "UTF-8", "-stdout", str(path)],
        check=True,
        capture_output=True,
    )
    text = clean_text(out.stdout.decode("utf-8", errors="replace"))
    # Split on blank lines into "paragraphs" and store as NarrativeText.
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    elems: list[Elem] = []
    for p in paras:
        elems.append(
            Elem(
                page=1,
                category="NarrativeText",
                text=p,
                top=None,
                bottom=None,
                y_norm_top=None,
                y_norm_bottom=None,
            )
        )
    return elems


# -----------------------------------------------------------------------------
# Journal article boundaries
# -----------------------------------------------------------------------------

@dataclass
class Article:
    article_id: str
    title: str | None
    start_page: int
    end_page: int | None = None
    doi: str | None = None


# Article first pages also commonly use these abstract-section keywords; we
# refuse them as titles.
_TITLE_REJECT_PREFIXES = (
    "Цель.",
    "Цель ",
    "Цель:",
    "Результаты.",
    "Результаты ",
    "Заключение.",
    "Заключение ",
    "Материал и методы",
    "Материалы и методы",
    "Ключевые слова",
    "Отношения и деятельность",
    "Для цитирования",
    "Поступила",
    "Aim.",
    "Conclusion.",
    "Results.",
    "Key words",
    "For citation",
)


def _looks_like_title(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 5 or len(s) > 220:
        return False
    if s.endswith((".", ":", ";")):
        return False
    if any(s.startswith(pref) for pref in _TITLE_REJECT_PREFIXES):
        return False
    low = s.lower()
    if "кардиоваскулярная терапия и профилактика" in low:
        return False
    if "issn" in low or "doi:" in low or "10.15829" in low:
        return False
    # Reject e-mail / contact strings that sometimes appear before the real
    # title on author-info pages (DA #52 R4: cardiovasc-2023 a06 picked up
    # "e-mail: [email removed]").
    if "@" in s or re.search(r"\be-?mail\b", low):
        return False
    # Author list signal: many commas + initials.
    if s.count(",") >= 3 and re.search(r"[A-ZА-ЯЁ]\.", s):
        return False
    if not s[:1].isalpha():
        return False
    return True


def detect_journal_articles(elems_by_page: dict[int, list[Elem]], short: str) -> list[Article]:
    arts: list[Article] = []
    for page in sorted(elems_by_page):
        els = elems_by_page[page]
        # Header element must contain doi: 10.15829/
        header_hits = [e for e in els if _DOI_JOURNAL.search(e.text)]
        if not header_hits:
            continue
        # Confirm with ISSN presence anywhere on page.
        if not any(_ISSN.search(e.text) for e in els):
            continue
        # Find title: the first plausible NarrativeText/Title element below
        # the header, in the top 40% of the page.
        title: str | None = None
        for e in els:
            if e.category in ("Header", "Footer", "PageBreak"):
                continue
            if not _looks_like_title(e.text):
                continue
            # Prefer ones above y_norm_top < 0.35 if we have coords.
            if e.y_norm_top is not None and e.y_norm_top > 0.45:
                continue
            title = e.text.strip()
            break
        # Fallback: any plausible title element on page.
        if not title:
            for e in els:
                if _looks_like_title(e.text):
                    title = e.text.strip()
                    break

        # DOI value
        doi = None
        m = _DOI_GENERIC.search(" ".join(e.text for e in els))
        if m:
            doi = m.group(0).rstrip(".,)")

        aid = f"{short}-a{len(arts) + 1:02d}"
        arts.append(Article(article_id=aid, title=title, start_page=page, doi=doi))

    for i, a in enumerate(arts):
        a.end_page = (arts[i + 1].start_page - 1) if i + 1 < len(arts) else max(elems_by_page)
    return arts


# -----------------------------------------------------------------------------
# Per-source pipeline
# -----------------------------------------------------------------------------

@dataclass
class SourceStats:
    file: str
    short: str
    lang: str
    pages: int
    chunks: int
    articles: int
    avg_chunk_tokens: float
    dropped_chunks: int
    headers_stripped: int
    null_section_chunks: int


def process_source(
    filename: str,
    short: str,
    lang_hint: str,
    kind: str,
    out_chunks: list[dict],
    out_articles: list[dict],
) -> SourceStats:
    path = SRC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        elems = extract_pdf(path)
    elif path.suffix.lower() == ".doc":
        elems = extract_doc(path)
    else:
        raise ValueError(f"Unsupported: {path.suffix}")

    # Detect running headers/footers.
    exact, patterns = find_running_headers_footers(elems)
    pat_list = list(patterns)
    n_pages = max((e.page for e in elems), default=1)

    # Group by page in original order (unstructured already returns reading
    # order within each page, column-aware).
    by_page: dict[int, list[Elem]] = defaultdict(list)
    for e in elems:
        by_page[e.page].append(e)

    # Article boundaries.
    if kind == "journal":
        articles = detect_journal_articles(by_page, short)
    else:
        # Single virtual article covering the whole doc.
        articles = [
            Article(
                article_id=short,
                title=None,
                start_page=min(by_page) if by_page else 1,
                end_page=max(by_page) if by_page else 1,
            )
        ]

    page_to_article: dict[int, str | None] = {}
    for a in articles:
        for pg in range(a.start_page, (a.end_page or a.start_page) + 1):
            page_to_article[pg] = a.article_id

    # Build chunks per (page, section) group.
    dropped = 0
    headers_stripped = 0
    chunk_token_counts: list[int] = []
    null_sections = 0

    # R1: carry the last confidently-detected section forward to subsequent
    # pages of the SAME article (chapter headings on long-form sources appear
    # only on chapter-start pages). Reset when article_id changes, when a
    # References chunk fires (DA #51 SS2), or after CARRY_MAX_PAGES of
    # carry-only pages to avoid stale labels on later pages.
    CARRY_MAX_PAGES = 10
    last_section: str | None = None
    last_section_article: str | None = None
    last_section_page: int | None = None

    for page in sorted(by_page):
        page_elems = by_page[page]
        article_id = page_to_article.get(page)
        if kind == "journal" and article_id is None:
            # cover, TOC, indices etc. -> skip outside-article pages
            continue

        # Strip running headers/footers and Header/Footer-categorised elements.
        kept: list[Elem] = []
        for e in page_elems:
            stripped = strip_running(e, exact, pat_list)
            if stripped is None:
                headers_stripped += 1
                continue
            # On article first pages the journal Header element is already
            # categorised Header and stripped above; the title NarrativeText
            # survives and feeds into the first chunk of the article body.
            kept.append(e)

        if not kept:
            continue

        # Section heading detection from the kept Title elements.
        detected = detect_section_for_page(kept)
        if detected is not None:
            last_section = detected
            last_section_article = article_id
            last_section_page = page
            section = detected
        elif (
            last_section is not None
            and last_section_article == article_id
            and last_section_page is not None
            and (page - last_section_page) <= CARRY_MAX_PAGES
        ):
            section = last_section
        else:
            section = None

        # Build paragraph stream from kept elements.
        paragraphs: list[str] = []
        for e in kept:
            t = e.text
            # Heuristic: drop ListItem entries that are only bullets like '•' or '»'.
            if e.category == "ListItem" and len(t) <= 3:
                continue
            paragraphs.append(t)

        if not paragraphs:
            continue

        page_chunks = chunk_paragraphs(paragraphs)
        for idx, ctext in enumerate(page_chunks, start=1):
            if is_garbage_chunk(ctext):
                dropped += 1
                continue
            # References label overrides page section. Also reset the
            # carry-forward state so the next page doesn't inherit a section
            # label across a References break (DA #51 SS2).
            if looks_like_references(ctext):
                sect = "References"
                last_section = None
                last_section_article = None
                last_section_page = None
            else:
                sect = section
            if sect is None:
                null_sections += 1
            tok = est_tokens(ctext)
            out_chunks.append(
                {
                    "chunk_id": f"{short}-p{page}-c{idx}",
                    "source_file": filename,
                    "source_short": short,
                    "article_id": article_id if article_id is not None else short,
                    "page": page,
                    "section": sect,
                    "text": ctext,
                    "lang": detect_lang(ctext),
                    "tokens_est": tok,
                }
            )
            chunk_token_counts.append(tok)

    for a in articles:
        out_articles.append(
            {
                "article_id": a.article_id,
                "source_file": filename,
                "source_short": short,
                "title": a.title,
                "doi": a.doi,
                "start_page": a.start_page,
                "end_page": a.end_page,
                "lang": lang_hint,
                "kind": kind,
            }
        )

    avg = sum(chunk_token_counts) / len(chunk_token_counts) if chunk_token_counts else 0.0
    return SourceStats(
        file=filename,
        short=short,
        lang=lang_hint,
        pages=n_pages,
        chunks=len(chunk_token_counts),
        articles=len(articles) if kind == "journal" else 1,
        avg_chunk_tokens=round(avg, 1),
        dropped_chunks=dropped,
        headers_stripped=headers_stripped,
        null_section_chunks=null_sections,
    )


# -----------------------------------------------------------------------------
# Validation (printed to stderr at end)
# -----------------------------------------------------------------------------

def validate(chunks: list[dict]) -> None:
    total = len(chunks)
    hdr_pat = re.compile(r"(Кардиоваскулярная терапия|FIFTH EDITION|Биобанкирование)")
    # R2: anchored count for backwards comparability; substring count for the
    # real bleed (header text leaks mid-chunk after paragraph joining).
    hdr_contam_anchored = sum(1 for c in chunks if hdr_pat.match(c["text"]))
    hdr_contam_search = sum(1 for c in chunks if hdr_pat.search(c["text"]))
    foot_pat = re.compile(r"\b\d{2,4}\s*$")
    foot_contam = sum(1 for c in chunks if foot_pat.search(c["text"]))
    # Additional metric: any chunk that's not labeled References but contains a
    # 'doi:' string in the body — a softer indicator of footer/citation bleed.
    doi_bleed = sum(
        1 for c in chunks
        if c.get("section") != "References" and "doi:" in c["text"].lower()
    )
    sec_null = sum(1 for c in chunks if not c.get("section"))
    refs_pat = re.compile(r"\[\s*1\s*\].*\[\s*2\s*\]")
    ref_bleed = sum(
        1 for c in chunks
        if c.get("section") != "References" and refs_pat.search(c["text"])
    )
    pct = (sec_null / total * 100) if total else 0.0
    print(f"\n--- validation ---", file=sys.stderr)
    print(f"total chunks: {total}", file=sys.stderr)
    print(
        f"header_contam: anchored={hdr_contam_anchored} search={hdr_contam_search}"
        f" (was 672, target <100 anchored, <200 search)",
        file=sys.stderr,
    )
    print(f"footer_contam: {foot_contam} (was 814, target <100)", file=sys.stderr)
    print(f"doi_in_body (non-Refs): {doi_bleed}", file=sys.stderr)
    print(f"section_null:  {sec_null}/{total} ({pct:.0f}%, target 30-50%)", file=sys.stderr)
    print(f"ref_bleed in non-References: {ref_bleed} (was 139, target <50)", file=sys.stderr)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: source dir not found: {SRC_DIR}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    all_articles: list[dict] = []
    per_source: list[SourceStats] = []

    t0 = time.time()
    for filename, short, lang, kind in SOURCES:
        ts = time.time()
        print(f"-> {filename} ({kind})", flush=True)
        stats = process_source(filename, short, lang, kind, all_chunks, all_articles)
        per_source.append(stats)
        print(
            f"   pages={stats.pages} chunks={stats.chunks} articles={stats.articles}"
            f" avg_tok={stats.avg_chunk_tokens} dropped={stats.dropped_chunks}"
            f" stripped={stats.headers_stripped}"
            f" null_sec={stats.null_section_chunks}"
            f" ({time.time()-ts:.1f}s)",
            flush=True,
        )

    chunks_path = OUT_DIR / "chunks.jsonl"
    articles_path = OUT_DIR / "articles.jsonl"
    manifest_path = OUT_DIR / "manifest.json"

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with articles_path.open("w", encoding="utf-8") as f:
        for a in all_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    lang_counts: dict[str, int] = {}
    for c in all_chunks:
        lang_counts[c["lang"]] = lang_counts.get(c["lang"], 0) + 1

    elapsed = time.time() - t0
    manifest = {
        "generated_at": date.today().isoformat(),
        "parser": "parse_materials_v2",
        "extraction_backend": "unstructured[pdf] strategy=fast",
        "sources": [asdict(s) for s in per_source],
        "total_chunks": len(all_chunks),
        "total_articles": len(all_articles),
        "languages": lang_counts,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "wall_seconds": round(elapsed, 1),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print()
    print(f"Wrote {len(all_chunks)} chunks   -> {chunks_path}")
    print(f"Wrote {len(all_articles)} articles -> {articles_path}")
    print(f"Wrote manifest                -> {manifest_path}")
    print(f"Language breakdown: {lang_counts}")
    print(f"Total wall: {elapsed:.1f}s")
    validate(all_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
