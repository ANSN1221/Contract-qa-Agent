"""PDF 取文、页眉页脚清洗、标题识别。

只做「把 PDF 变成带页码坐标的行」这一件事，条款切分交给 splitter。
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pymupdf

from contract_qa.models import Line, PageStat

LOW_TEXT_THRESHOLD = 20

# 页眉页脚判定：出现在足够多的页面上、且足够短
RUNNING_LINE_MIN_PAGES = 5
RUNNING_LINE_RATIO = 0.5
RUNNING_LINE_MAX_LEN = 40

_CN_NUM = "一二三四五六七八九十百千零〇两"
_RE_PAGE_NUMBER = re.compile(
    rf"^(?:[-—–－]\s*)?(?:第\s*)?\d{{1,4}}(?:\s*页)?(?:\s*共\s*\d{{1,4}}\s*页)?(?:\s*[-—–－])?$"
)
# 目录行：点线引导 + 页码，或任何含 4 个以上连续点的行
_RE_TOC_LINE = re.compile(r"(?:\.{4,}|…{2,}|·{4,}|．{4,})|(?:\.{2,}\s*\d{1,3}\s*$)")
# 「1.5 万元」不是编号 1.5，而是金额：数字后面紧跟计量单位时一律不认作标题
_RE_MEASURE_AFTER_NUMBER = re.compile(
    r"^\d{1,2}(?:\.\d{1,2}){1,3}\s*(?:万元|亿元|元|米|千米|公里|吨|台|套|个|人|次|天|日|年|月|%|％|‰)"
)

# 层级编号：数字越小越靠上。判断顺序即优先级，必须先匹配多级编号再匹配单级，
# 否则「1.1、」会被「1.」抢走。
_LEVEL_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (1, re.compile(rf"^第[{_CN_NUM}\d]{{1,4}}(?:部分|篇)")),
    (2, re.compile(rf"^第[{_CN_NUM}\d]{{1,4}}章")),
    (3, re.compile(rf"^第[{_CN_NUM}\d]{{1,4}}节")),
    (4, re.compile(rf"^第[{_CN_NUM}\d]{{1,4}}条")),
    (5, re.compile(rf"^[{_CN_NUM}]{{1,3}}、")),
    (6, re.compile(rf"^[（(][{_CN_NUM}]{{1,3}}[）)]")),
    (8, re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\s*[、.]?\s*\S")),
    (7, re.compile(r"^\d{1,2}\s*[.、．]\s*(?!\d)\S")),
    (9, re.compile(r"^[（(]\d{1,2}[）)]")),
)

_RE_HEADING_LABEL = re.compile(
    rf"^(第[{_CN_NUM}\d]{{1,4}}(?:部分|篇|章|节|条)|[{_CN_NUM}]{{1,3}}、"
    rf"|[（(][{_CN_NUM}\d]{{1,3}}[）)]|\d{{1,2}}(?:\.\d{{1,2}}){{1,3}}|\d{{1,2}}[.、．])"
)


def normalize_for_match(text: str) -> str:
    """只用于比较/匹配的归一化，不写回正文。"""
    s = unicodedata.normalize("NFKC", text)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)
    return s


def extract_pages(pdf_path: Path) -> tuple[list[tuple[int, str]], int]:
    """返回 [(页码, 页文本)], 总页数。页码从 1 开始。"""
    doc = pymupdf.open(pdf_path)
    try:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        return pages, doc.page_count
    finally:
        doc.close()


def page_stats(pages: list[tuple[int, str]]) -> list[PageStat]:
    stats: list[PageStat] = []
    for page_no, text in pages:
        stripped = text.strip()
        stats.append(
            PageStat(
                page_no=page_no,
                chars=len(stripped),
                low_text=len(stripped) < LOW_TEXT_THRESHOLD,
            )
        )
    return stats


def _running_keys(pages: list[tuple[int, str]]) -> set[str]:
    """找出跨页重复出现的短行（页眉、页脚、水印、密级标识）。"""
    if len(pages) < RUNNING_LINE_MIN_PAGES:
        return set()
    seen: dict[str, set[int]] = defaultdict(set)
    for page_no, text in pages:
        for raw in text.splitlines():
            key = normalize_for_match(raw)
            if not key or len(key) > RUNNING_LINE_MAX_LEN:
                continue
            seen[key].add(page_no)
    threshold = max(RUNNING_LINE_MIN_PAGES, int(len(pages) * RUNNING_LINE_RATIO))
    return {key for key, pages_hit in seen.items() if len(pages_hit) >= threshold}


def classify_heading(text: str) -> tuple[int, str]:
    """返回 (级别, 标题标签)。0 表示不是标题。

    级别约定（数字越小越靠上）：
    1 第X部分/篇 · 2 第X章 · 3 第X节 · 4 第X条
    5 一、 · 6 (一) · 7 1./1、 · 8 1.1 · 9 (1)

    标签（clause_no）取自原文而非归一化文本，保留全角括号等原始写法，
    因为它是最终引用给用户看的东西。
    """
    probe = unicodedata.normalize("NFKC", text).strip()
    if _RE_TOC_LINE.search(probe[:120]):
        return 0, ""
    if _RE_MEASURE_AFTER_NUMBER.match(probe):
        return 0, ""
    for level, pattern in _LEVEL_PATTERNS:
        if pattern.match(probe):
            label = _RE_HEADING_LABEL.match(text.strip())
            return level, (label.group(1) if label else "")
    return 0, ""


def _toc_pages(pages: list[tuple[int, str]]) -> set[int]:
    """整页都是「标题......页码」的，判定为目录页，整页丢弃。

    逐行丢会留下只有「第一章」三个字的残行，它会污染后续所有条款的 chapter_path。
    """
    toc: set[int] = set()
    for page_no, text in pages:
        hits = sum(1 for line in text.splitlines() if _RE_TOC_LINE.search(line[:120]))
        if hits >= 4:
            toc.add(page_no)
    return toc


def build_lines(pages: list[tuple[int, str]]) -> tuple[list[Line], int]:
    """把整份文档拍平成带页码与偏移量的行序列，顺带丢掉页眉页脚与目录页。"""
    running = _running_keys(pages)
    toc_pages = _toc_pages(pages)
    raw: list[tuple[str, int]] = []
    dropped = 0

    for page_no, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            key = normalize_for_match(stripped)
            if not key:
                continue
            if key in running or _RE_PAGE_NUMBER.match(key):
                dropped += 1
                continue
            if page_no in toc_pages or _RE_TOC_LINE.search(stripped[:120]):
                dropped += 1
                continue
            raw.append((stripped, page_no))

    raw, merged_bare = _merge_bare_headings(raw)

    lines: list[Line] = []
    offset = 0
    for text, page_no in raw:
        level, label = classify_heading(text)
        lines.append(
            Line(
                text=text,
                page_no=page_no,
                offset=offset,
                is_heading=level > 0,
                heading_level=level,
                heading_label=label,
            )
        )
        offset += len(text) + 1

    return lines, dropped + merged_bare


def _merge_bare_headings(raw: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    """「第一章」单独一行、章名在下一行时，把两行拼回去。

    否则 chapter_path 里全是「第一章」「第二章」，引用时看不出是哪一章。
    """
    out: list[tuple[str, int]] = []
    merged = 0
    index = 0
    while index < len(raw):
        text, page_no = raw[index]
        level, label = classify_heading(text)
        bare = level > 0 and label and len(normalize_for_match(text)) <= len(
            normalize_for_match(label)
        ) + 1
        if bare and index + 1 < len(raw):
            nxt, _ = raw[index + 1]
            if classify_heading(nxt)[0] == 0 and len(nxt) <= 40:
                out.append((f"{text} {nxt}", page_no))
                merged += 1
                index += 2
                continue
        out.append((text, page_no))
        index += 1
    return out, merged


def join_lines(texts: list[str]) -> str:
    """把 PDF 硬换行拼回自然段：中文边界不加空格，英文/数字边界补一个空格。"""
    out = ""
    for piece in texts:
        piece = piece.strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        prev, nxt = out[-1], piece[0]
        if _is_cjk(prev) or _is_cjk(nxt):
            out += piece
        elif prev in "-—_/·":
            out += piece
        else:
            out += " " + piece
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )
